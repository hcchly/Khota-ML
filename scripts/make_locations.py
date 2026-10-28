"""Build data/locations.csv: one row per image with a location_id for the grouped split (AGENTS.md section 4).

GPS comes from:
  1. data/source/gps_batch1_2.csv  (Images sheets of the batch 1 and batch 2 Excel files), and
  2. optionally, EXIF GPS read from original photos:  --originals <folder>  (HEIC or JPG named by Image_ID,
     e.g. RYD-0701.HEIC). Use this for batch 3 and for any image that has no GPS yet.

Images within LOCATION_RADIUS_M of each other (chained) get the same location_id.
An image that still has no GPS gets location_id = "nogps_<block>", where a block is a run of consecutive
Image_IDs without GPS (consecutive photos are taken at the same place). The script prints these so the team
can check them.

Usage:
    python scripts/make_locations.py
    python scripts/make_locations.py --originals /path/to/batch3_originals
"""
import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from khota_ml.config import LOCATION_RADIUS_M  # noqa: E402

EXCEL = ROOT / "data/source/Khota_Model_Final.xlsx"
GPS_CSV = ROOT / "data/source/gps_batch1_2.csv"
OUT = ROOT / "data/locations.csv"


def exif_gps(path: Path):
    """Return (lat, lon) from a photo's EXIF, or None."""
    try:
        from PIL import Image
        if path.suffix.lower() in (".heic", ".heif"):
            from pillow_heif import register_heif_opener  # pip install pillow-heif
            register_heif_opener()
        exif = Image.open(path).getexif()
        gps = exif.get_ifd(0x8825)
        if not gps or 2 not in gps or 4 not in gps:
            return None

        def dms(v):
            return float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600

        lat, lon = dms(gps[2]), dms(gps[4])
        if gps.get(1) == "S":
            lat = -lat
        if gps.get(3) == "W":
            lon = -lon
        return lat, lon
    except Exception as exc:  # unreadable file or no EXIF
        print(f"  no EXIF GPS in {path.name}: {exc}")
        return None


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 6371000 * 2 * np.arcsin(np.sqrt(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--originals", type=Path, help="folder with original photos named by Image_ID")
    args = ap.parse_args()

    index = pd.read_excel(EXCEL, sheet_name="Image_Index")[["Batch", "Image_ID"]]
    gps = pd.read_csv(GPS_CSV)[["Image_ID", "Latitude", "Longitude"]]
    df = index.merge(gps, on="Image_ID", how="left")
    df["gps_source"] = np.where(df.Latitude.notna(), "batch_excel", "")

    if args.originals:
        files = {re.sub(r"\.[^.]+$", "", p.name).upper(): p for p in args.originals.rglob("*") if p.is_file()}
        for i, row in df[df.Latitude.isna()].iterrows():
            p = files.get(row.Image_ID.upper())
            if p is None:
                continue
            ll = exif_gps(p)
            if ll:
                df.loc[i, ["Latitude", "Longitude", "gps_source"]] = [ll[0], ll[1], "exif"]

    # Cluster images that have GPS (single linkage, radius LOCATION_RADIUS_M).
    has = df[df.Latitude.notna()].index.to_list()
    lat = df.loc[has, "Latitude"].to_numpy()
    lon = df.loc[has, "Longitude"].to_numpy()
    label = -np.ones(len(has), dtype=int)
    n = 0
    for i in range(len(has)):
        if label[i] >= 0:
            continue
        stack, label[i] = [i], n
        while stack:
            j = stack.pop()
            near = np.where((haversine_m(lat[j], lon[j], lat, lon) <= LOCATION_RADIUS_M) & (label < 0))[0]
            label[near] = n
            stack.extend(near.tolist())
        n += 1
    df["location_id"] = ""
    df.loc[has, "location_id"] = [f"loc_{k:03d}" for k in label]

    # Images without GPS: one block per run of consecutive Image_IDs.
    missing = df[df.location_id == ""].copy()
    if len(missing):
        nums = missing.Image_ID.str.extract(r"(\d+)")[0].astype(int)
        block = (nums.diff().fillna(1) != 1).cumsum()
        df.loc[missing.index, "location_id"] = [f"nogps_{b:02d}" for b in block]
        print("Images without GPS (grouped by consecutive IDs):")
        for loc, grp in df.loc[missing.index].groupby("location_id"):
            print(f"  {loc}: {grp.Image_ID.iloc[0]} .. {grp.Image_ID.iloc[-1]} ({len(grp)} images)")

    df = df.rename(columns={"Batch": "batch", "Image_ID": "image_id", "Latitude": "latitude", "Longitude": "longitude"})
    df[["image_id", "batch", "latitude", "longitude", "gps_source", "location_id"]].to_csv(OUT, index=False)
    sizes = df.location_id.value_counts()
    print(f"Wrote {OUT.relative_to(ROOT)}: {len(df)} images, {df.location_id.nunique()} locations, "
          f"largest location {sizes.max()} images")


if __name__ == "__main__":
    main()
