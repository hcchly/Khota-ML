"""Build data/annotations.csv from data/source/Khota_Model_Final.xlsx (AGENTS.md section 3).

The Excel is the team's final label table:
  - Batch_1 / Batch_2 / Batch_3: one row per box. Box_Index = 0-based line order in the Roboflow YOLO label file.
    Each row has Class_ID (0 curb, 1 surface), normalised x_center/y_center/width/height, Severity, Reason_Codes.
  - Image_Index:    every image used by the models, with its Roboflow image and label filenames and Box_Count.
  - Empty_Images:   images with no target defect (background images, empty label file).
  - Excluded_Images: Not Assessable and Duplicate images. They are removed from both models.

With --raw, the script also checks every box against the Roboflow export: the label file must exist, have
Box_Count lines, and each line must match the Excel class and coordinates. Run it with --raw once after
downloading the exports; any mismatch stops the script.

Usage:
    python scripts/convert_annotations.py                 # Excel only
    python scripts/convert_annotations.py --raw raw/      # Excel + check against the Roboflow exports
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from khota_ml.config import DETECTION_CLASSES, EXCEL_CLASS_NAMES, NO_DEFECT, SEVERITY_CLASSES  # noqa: E402

EXCEL = ROOT / "data/source/Khota_Model_Final.xlsx"
LOCATIONS = ROOT / "data/locations.csv"
OUT = ROOT / "data/annotations.csv"
COLUMNS = ["image_id", "batch", "image_file", "box_id", "location_id", "defect_class", "severity",
           "x_center", "y_center", "width", "height", "reason_codes"]


def clean_codes(text) -> str:
    """Normalise reason codes, e.g. 'CR;H1' -> 'CR-H1', '\\tSD-L1; SD-L2;SD-L4' -> 'SD-L1; SD-L2; SD-L4'."""
    if pd.isna(text):
        return ""
    text = str(text).strip().replace("CR;H", "CR-H")
    parts = [p.strip() for p in re.split(r"[;,]", text) if p.strip()]
    return "; ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, help="folder containing the Roboflow YOLO exports")
    ap.add_argument("--tol", type=float, default=1e-4, help="coordinate tolerance for the --raw check")
    args = ap.parse_args()

    xl = pd.read_excel(EXCEL, sheet_name=None)
    excluded = set(xl["Excluded_Images"].Image_ID)
    index = xl["Image_Index"]
    empty = set(xl["Empty_Images"].Image_ID)
    boxes = pd.concat([xl[f"Batch_{b}"].assign(Batch=b) for b in (1, 2, 3)], ignore_index=True)

    errors = []
    if excluded & set(index.Image_ID):
        errors.append(f"excluded images listed in Image_Index: {sorted(excluded & set(index.Image_ID))}")
    boxes = boxes[~boxes.Image_ID.isin(excluded)]

    # Class and severity checks.
    for _, r in boxes.iterrows():
        name = EXCEL_CLASS_NAMES.get(r.Class_Name)
        if name is None or DETECTION_CLASSES.index(name) != r.Class_ID:
            errors.append(f"{r.Image_ID} box {r.Box_Index}: Class_ID {r.Class_ID} / Class_Name {r.Class_Name}")
        if str(r.Severity).strip().lower() not in SEVERITY_CLASSES:
            errors.append(f"{r.Image_ID} box {r.Box_Index}: severity '{r.Severity}'")
        for c in ("x_center", "y_center", "width", "height"):
            if not 0 <= r[c] <= 1:
                errors.append(f"{r.Image_ID} box {r.Box_Index}: {c}={r[c]} outside [0, 1]")

    # Box counts and Box_Index order per image.
    counts = boxes.groupby("Image_ID").Box_Index.agg(list)
    for _, r in index.iterrows():
        got = sorted(counts.get(r.Image_ID, []))
        if got != list(range(r.Box_Count)):
            errors.append(f"{r.Image_ID}: Box_Count {r.Box_Count} but Box_Index {got}")
        if (r.Box_Count == 0) != (r.Image_ID in empty):
            errors.append(f"{r.Image_ID}: Box_Count {r.Box_Count} disagrees with Empty_Images")
    missing = set(boxes.Image_ID) - set(index.Image_ID)
    if missing:
        errors.append(f"boxes for images not in Image_Index: {sorted(missing)}")

    if args.raw:
        errors += check_raw(args.raw, index, boxes, args.tol)

    if errors:
        print(f"STOP: {len(errors)} problem(s)")
        for e in errors[:50]:
            print("  -", e)
        sys.exit(1)

    # Locations.
    if not LOCATIONS.exists():
        sys.exit("STOP: data/locations.csv not found. Run scripts/make_locations.py first.")
    loc = pd.read_csv(LOCATIONS).set_index("image_id").location_id
    no_loc = sorted(set(index.Image_ID) - set(loc.index))
    if no_loc:
        sys.exit(f"STOP: no location_id for {no_loc[:20]}")

    files = index.set_index("Image_ID").Image_Filename
    rows = []
    for _, r in boxes.sort_values(["Image_ID", "Box_Index"]).iterrows():
        rows.append([r.Image_ID, r.Batch, files[r.Image_ID], int(r.Box_Index), loc[r.Image_ID],
                     EXCEL_CLASS_NAMES[r.Class_Name], str(r.Severity).strip().lower(),
                     r.x_center, r.y_center, r.width, r.height, clean_codes(r.Reason_Codes)])
    for _, r in index[index.Box_Count == 0].iterrows():
        rows.append([r.Image_ID, r.Batch, r.Image_Filename, "", loc[r.Image_ID], NO_DEFECT, "",
                     "", "", "", "", ""])
    out = pd.DataFrame(rows, columns=COLUMNS).sort_values(["image_id", "box_id"], key=lambda s: s.astype(str))
    out.to_csv(OUT, index=False)

    b = out[out.defect_class != NO_DEFECT]
    print(f"Wrote {OUT.relative_to(ROOT)}: {out.image_id.nunique()} images "
          f"({(out.defect_class == NO_DEFECT).sum()} no_defect), {len(b)} boxes, "
          f"{len(excluded)} images excluded")
    print(pd.crosstab(b.defect_class, b.severity, margins=True).reindex(columns=SEVERITY_CLASSES + ["All"]))


def check_raw(raw: Path, index, boxes, tol):
    """Compare the Excel with the Roboflow YOLO label files."""
    errors = []
    labels = {p.name: p for p in raw.rglob("*.txt")}
    images = {p.name for p in raw.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    by_img = boxes.set_index(["Image_ID", "Box_Index"])
    for _, r in index.iterrows():
        if r.Image_Filename not in images:
            errors.append(f"{r.Image_ID}: image file {r.Image_Filename} not found under {raw}")
        p = labels.get(r.Label_Filename)
        if p is None:
            if r.Box_Count:
                errors.append(f"{r.Image_ID}: label file {r.Label_Filename} not found")
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
                errors.append(f"{r.Image_ID} box {i}: label file {ln[:5]} vs Excel {[int(x.Class_ID)] + [round(float(v), 6) for v in ref]}")
    print(f"--raw check: {len(index)} images, {len(errors)} problem(s)")
    return errors


if __name__ == "__main__":
    main()
