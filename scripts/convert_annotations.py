"""Build data/annotations.csv from data/source/Khota_Model_Final_Binary.xlsx (AGENTS.md section 3). Version 2.

The Excel (version 2, binary severity) has:
  - Boxes:           one row per box. Box_Index "B01" = line 1 of the Roboflow YOLO label file, "B02" = line 2, ...
                     Class_ID (0 curb, 1 surface), normalised x_center/y_center/width/height, Severity.
  - Batch_1/2/3:     one row per image + class + severity, Box_Index lists its boxes ("B01, B03"), Reason_Codes.
  - Image_Index:     every image used by the models, with its Roboflow image and label filenames and Box_Count.
  - Empty_Images:    images with no target defect (background images, empty label file).
  - Excluded_Images: Not Assessable and Duplicate images. They are removed from both models.

The script checks that the Batch sheets and the Boxes sheet agree (every box in exactly one row, same class and
severity), and with --raw that every box matches the Roboflow label files. Any problem stops the script.

Usage:
    python scripts/convert_annotations.py --raw raw/                # full check (labels + images)
    python scripts/convert_annotations.py --raw raw/ --labels-only  # when only the label files are downloaded
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from khota_ml.config import (DETECTION_CLASSES, EXCEL_CLASS_NAMES, EXCEL_SEVERITY_NAMES,  # noqa: E402
                             NO_DEFECT, SEVERITY_CLASSES)

EXCEL = ROOT / "data/source/Khota_Model_Final_Binary.xlsx"
LOCATIONS = ROOT / "data/locations.csv"
OUT = ROOT / "data/annotations.csv"
COLUMNS = ["image_id", "batch", "image_file", "box_id", "box_label", "location_id", "defect_class", "severity",
           "x_center", "y_center", "width", "height", "reason_codes"]


def box_number(label) -> int:
    """'B03' -> 2 (0-based line index in the label file)."""
    m = re.fullmatch(r"B(\d{2})", str(label).strip())
    if not m:
        raise ValueError(f"bad Box_Index '{label}'")
    return int(m.group(1)) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, help="folder containing the Roboflow YOLO exports")
    ap.add_argument("--labels-only", action="store_true", help="with --raw: do not require the image files")
    ap.add_argument("--tol", type=float, default=1e-4, help="coordinate tolerance for the --raw check")
    args = ap.parse_args()

    xl = pd.read_excel(EXCEL, sheet_name=None)
    excluded = set(xl["Excluded_Images"].Image_ID)
    index = xl["Image_Index"]
    empty = set(xl["Empty_Images"].Image_ID)
    boxes = xl["Boxes"].copy()
    rows = pd.concat([xl[f"Batch_{b}"].assign(Batch=b) for b in (1, 2, 3)], ignore_index=True)
    errors = []

    if excluded & set(index.Image_ID):
        errors.append(f"excluded images listed in Image_Index: {sorted(excluded & set(index.Image_ID))}")
    if empty - set(index.Image_ID):
        errors.append(f"Empty_Images not in Image_Index: {sorted(empty - set(index.Image_ID))}")

    # Boxes sheet: classes, severities, coordinates.
    boxes["box_id"] = boxes.Box_Index.map(box_number)
    for _, r in boxes.iterrows():
        tag = f"{r.Image_ID} {r.Box_Index}"
        name = EXCEL_CLASS_NAMES.get(r.Class_Name)
        if name is None or DETECTION_CLASSES.index(name) != r.Class_ID:
            errors.append(f"{tag}: Class_ID {r.Class_ID} / Class_Name {r.Class_Name}")
        if r.Severity not in EXCEL_SEVERITY_NAMES:
            errors.append(f"{tag}: severity '{r.Severity}'")
        for c in ("x_center", "y_center", "width", "height"):
            if not 0 <= r[c] <= 1:
                errors.append(f"{tag}: {c}={r[c]} outside [0, 1]")
    if boxes.duplicated(["Image_ID", "box_id"]).any():
        errors.append("duplicate boxes in Boxes sheet")

    # Box_Count and order per image.
    got = boxes.groupby("Image_ID").box_id.agg(sorted)
    for _, r in index.iterrows():
        ids = got.get(r.Image_ID, [])
        if ids != list(range(int(r.Box_Count))):
            errors.append(f"{r.Image_ID}: Box_Count {r.Box_Count} but boxes {ids}")
        if (r.Box_Count == 0) != (r.Image_ID in empty):
            errors.append(f"{r.Image_ID}: Box_Count {r.Box_Count} disagrees with Empty_Images")
    if set(boxes.Image_ID) - set(index.Image_ID):
        errors.append(f"boxes for images not in Image_Index: {sorted(set(boxes.Image_ID) - set(index.Image_ID))}")

    # Batch sheets must agree with Boxes: every box in exactly one row of the same class and severity.
    owner = {}
    for _, r in rows.iterrows():
        if pd.isna(r.Box_Index) or not str(r.Box_Index).strip():
            errors.append(f"{r.Image_ID}: row {r.Class_Name} / {r.Severity} has no boxes")
            continue
        for lab in str(r.Box_Index).split(","):
            key = (r.Image_ID, box_number(lab))
            if key in owner:
                errors.append(f"{r.Image_ID} {lab.strip()}: in two rows")
            owner[key] = (r.Class_Name, r.Severity, r.Reason_Codes)
    for _, r in boxes.iterrows():
        o = owner.get((r.Image_ID, r.box_id))
        if o is None:
            errors.append(f"{r.Image_ID} {r.Box_Index}: in Boxes but in no Batch row")
        elif o[:2] != (r.Class_Name, r.Severity):
            errors.append(f"{r.Image_ID} {r.Box_Index}: Boxes says {r.Class_Name}/{r.Severity}, Batch row says {o[:2]}")
    if len(owner) != len(boxes):
        errors.append(f"Batch rows list {len(owner)} boxes, Boxes sheet has {len(boxes)}")

    if args.raw:
        errors += check_raw(args.raw, index, boxes, args.tol, args.labels_only)

    if errors:
        print(f"STOP: {len(errors)} problem(s)")
        for e in errors[:50]:
            print("  -", e)
        sys.exit(1)

    if not LOCATIONS.exists():
        sys.exit("STOP: data/locations.csv not found. Run scripts/make_locations.py first.")
    loc = pd.read_csv(LOCATIONS).set_index("image_id").location_id
    no_loc = sorted(set(index.Image_ID) - set(loc.index))
    if no_loc:
        sys.exit(f"STOP: no location_id for {no_loc[:20]}")

    files = index.set_index("Image_ID").Image_Filename
    batch = index.set_index("Image_ID").Batch
    out = []
    for _, r in boxes.sort_values(["Image_ID", "box_id"]).iterrows():
        codes = owner[(r.Image_ID, r.box_id)][2]
        out.append([r.Image_ID, batch[r.Image_ID], files[r.Image_ID], int(r.box_id), r.Box_Index, loc[r.Image_ID],
                    EXCEL_CLASS_NAMES[r.Class_Name], EXCEL_SEVERITY_NAMES[r.Severity],
                    r.x_center, r.y_center, r.width, r.height, "" if pd.isna(codes) else str(codes)])
    for _, r in index[index.Box_Count == 0].iterrows():
        out.append([r.Image_ID, r.Batch, r.Image_Filename, "", "", loc[r.Image_ID], NO_DEFECT, "",
                    "", "", "", "", ""])
    out = pd.DataFrame(out, columns=COLUMNS).sort_values(["image_id", "box_id"], key=lambda s: s.astype(str))
    out.to_csv(OUT, index=False)

    b = out[out.defect_class != NO_DEFECT]
    print(f"Wrote {OUT.relative_to(ROOT)}: {out.image_id.nunique()} images "
          f"({(out.defect_class == NO_DEFECT).sum()} no_defect), {len(b)} boxes, {len(excluded)} images excluded")
    print(pd.crosstab(b.defect_class, b.severity, margins=True).reindex(columns=SEVERITY_CLASSES + ["All"]))


def check_raw(raw: Path, index, boxes, tol, labels_only):
    """Compare the Excel with the Roboflow YOLO label files (found by filename anywhere under raw/)."""
    errors = []
    labels = {p.name: p for p in raw.rglob("*.txt") if "__MACOSX" not in p.parts}
    images = {p.name for p in raw.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    by_img = boxes.set_index(["Image_ID", "box_id"])
    for _, r in index.iterrows():
        if not labels_only and r.Image_Filename not in images:
            errors.append(f"{r.Image_ID}: image file {r.Image_Filename} not found under {raw}")
        p = labels.get(r.Label_Filename)
        if p is None:
            errors.append(f"{r.Image_ID}: label file {r.Label_Filename} not found (an old export?)")
            continue
        lines = [ln.split() for ln in p.read_text().splitlines() if ln.strip()]
        if len(lines) != r.Box_Count:
            errors.append(f"{r.Image_ID}: {len(lines)} lines in label file, Box_Count {r.Box_Count}")
            continue
        for i, ln in enumerate(lines):
            x = by_img.loc[(r.Image_ID, i)]
            vals = [float(v) for v in ln[1:5]]
            ref = [x.x_center, x.y_center, x.width, x.height]
            if int(ln[0]) != x.Class_ID or max(abs(a - b) for a, b in zip(vals, ref)) > tol:
                errors.append(f"{r.Image_ID} B{i + 1:02d}: label file {ln[:5]} vs Excel "
                              f"{[int(x.Class_ID)] + [round(float(v), 6) for v in ref]}")
    print(f"--raw check: {len(index)} images, {len(errors)} problem(s)")
    return errors


if __name__ == "__main__":
    main()
