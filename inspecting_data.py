"""
Inspecting the dataset

"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect the NeuroSight MRI dataset")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root folder containing class subfolders (e.g. 'Astrocytoma T1/')")
    parser.add_argument("--metadata",  type=str, required=True,
                        help="Path to metadata.json")
    parser.add_argument("--output",    type=str, default="./inspection_report",
                        help="Output directory for all inspection artefacts")
    return parser.parse_args()

NORMAL_CLASSES = {"Normal T1", "Normal T1C+", "Normal T2"}

WEIGHTING_MAP = {
    "T1C+": "T1C+",   
    "T1":   "T1",
    "T2":   "T2",
}

def extract_weighting(class_label: str) -> str:
    """Pull T1 / T1C+ / T2 from the class string."""
    for key in WEIGHTING_MAP:
        if class_label.endswith(key):
            return WEIGHTING_MAP[key]
    return "Unknown"

def extract_tumor_type(class_label: str) -> str:
    """Strip the weighting suffix to get the base tumor type."""
    for suffix in ["T1C+", "T1", "T2"]:
        if class_label.endswith(suffix):
            return class_label[: -len(suffix)].strip()
    return class_label

def normalise_key(raw_key: str) -> Path:
    """
    JSON keys use Windows-style backslashes: 'Astrocytoma T1\\file.jpg'
    Normalise to forward slashes for cross-platform Path handling.
    """
    return Path(raw_key.replace("\\", "/"))



def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = []  
    def log(msg=""):
        print(msg)
        lines.append(str(msg))

  
    log("=" * 60)
    log("  NEUROSIGHT — DATASET INSPECTION")
    log("=" * 60)

    with open(args.metadata, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    log(f"\n Metadata entries     : {len(metadata):,}")

    disk_files = set()
    for root, _, files in os.walk(data_root):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                rel = Path(root).relative_to(data_root) / fname
                disk_files.add(rel)

    log(f"  Files found on disk  : {len(disk_files):,}")

 
    json_paths  = {normalise_key(k) for k in metadata.keys()}
    in_json_not_disk = json_paths  - disk_files
    in_disk_not_json = disk_files  - json_paths

    log(f"\n Cross-reference check:")
    log(f"   In JSON, missing on disk : {len(in_json_not_disk)}")
    log(f"   On disk, missing in JSON : {len(in_disk_not_json)}")

    if in_json_not_disk:
        missing_df = pd.DataFrame(sorted(str(p) for p in in_json_not_disk),
                                  columns=["missing_from_disk"])
        missing_df.to_csv(output_dir / "missing_files.csv", index=False)
        log(f"     Saved missing list → missing_files.csv")

    records = []
    for raw_key, meta in metadata.items():
        rel_path = normalise_key(raw_key)
        abs_path = data_root / rel_path

        class_label  = meta.get("class", "Unknown")
        tumor_type   = extract_tumor_type(class_label)
        weighting    = extract_weighting(class_label)
        is_normal    = class_label in NORMAL_CLASSES

        point        = meta.get("point", {})
        px           = point.get("x", None)
        py           = point.get("y", None)

        # Normalised point (0–1) relative to 512×512
        w, h         = meta.get("width", 512), meta.get("height", 512)
        px_norm      = round(px / w, 4) if px is not None else None
        py_norm      = round(py / h, 4) if py is not None else None

        location_tags = meta.get("location", [])

        records.append({
            "rel_path"      : str(rel_path),
            "abs_path"      : str(abs_path),
            "file_exists"   : abs_path.exists(),
            "class"         : class_label,
            "tumor_type"    : tumor_type,
            "weighting"     : weighting,
            "is_normal"     : is_normal,
            "width"         : w,
            "height"        : h,
            "point_x"       : px,
            "point_y"       : py,
            "point_x_norm"  : px_norm,
            "point_y_norm"  : py_norm,
            "has_location"  : len(location_tags) > 0,
            "location_tags" : "|".join(location_tags) if location_tags else "",
            "description_len": len(meta.get("description", "")),
        })

    df = pd.DataFrame(records)

    # Class distribution
    log("\n" + "─" * 60)
    log("  CLASS DISTRIBUTION (30 classes)")
    log("─" * 60)

    class_dist = (
        df.groupby(["tumor_type", "weighting", "class"])
          .size()
          .reset_index(name="count")
          .sort_values(["tumor_type", "weighting"])
    )

    log(class_dist.to_string(index=False))
    class_dist.to_csv(output_dir / "class_distribution.csv", index=False)

    # High-level summary numbers
    log("\n" + "─" * 60)
    log("  HIGH-LEVEL SUMMARY")
    log("─" * 60)

    log(f"  Total images            : {len(df):,}")
    log(f"  Unique classes          : {df['class'].nunique()}")
    log(f"  Unique tumor types      : {df['tumor_type'].nunique()}")
    log(f"  MRI weightings          : {sorted(df['weighting'].unique())}")
    log(f"  Normal images           : {df['is_normal'].sum():,}")
    log(f"  Abnormal images         : {(~df['is_normal']).sum():,}")
    log(f"  Images with location tag: {df['has_location'].sum():,}")
    log(f"  Files verified on disk  : {df['file_exists'].sum():,} / {len(df):,}")

   
    log("\n" + "─" * 60)
    log("  IMAGE DIMENSIONS")
    log("─" * 60)
    dim_counts = df.groupby(["width", "height"]).size().reset_index(name="count")
    log(dim_counts.to_string(index=False))
    if len(dim_counts) > 1:
        log("    WARNING: Not all images are the same size!")
    else:
        log("   All images are uniformly 512×512")

    # ── 8. Point statistics ───────────────────────────────────────────────────
    log("\n" + "─" * 60)
    log("  LESION POINT STATISTICS (abnormal classes only)")
    log("─" * 60)

    abnormal = df[~df["is_normal"]]
    log(f"  X  — mean: {abnormal['point_x'].mean():.1f}  "
        f"std: {abnormal['point_x'].std():.1f}  "
        f"min: {abnormal['point_x'].min()}  "
        f"max: {abnormal['point_x'].max()}")
    log(f"  Y  — mean: {abnormal['point_y'].mean():.1f}  "
        f"std: {abnormal['point_y'].std():.1f}  "
        f"min: {abnormal['point_y'].min()}  "
        f"max: {abnormal['point_y'].max()}")

   # (possible Normal mislabels?)
    centre_x, centre_y = 256, 256
    threshold = 10  # pixels
    suspicious = abnormal[
        (abs(abnormal["point_x"] - centre_x) < threshold) &
        (abs(abnormal["point_y"] - centre_y) < threshold)
    ]
    log(f"\n  Points within {threshold}px of centre in abnormal classes: {len(suspicious)}")
    if len(suspicious):
        log("    These may need manual review:")
        log(suspicious[["rel_path", "class", "point_x", "point_y"]].to_string(index=False))

    # Per-class point stats
    point_stats = (
        abnormal.groupby("class")[["point_x_norm", "point_y_norm"]]
        .agg(["mean", "std"])
        .round(3)
    )
    point_stats.to_csv(output_dir / "points_stats.csv")
    log(f"\n  Per-class point stats saved → points_stats.csv")

    log("\n" + "─" * 60)
    log("  CLASS BALANCE")
    log("─" * 60)

    counts = df["class"].value_counts()
    log(f"  Most common class  : {counts.index[0]}  ({counts.iloc[0]} images)")
    log(f"  Least common class : {counts.index[-1]}  ({counts.iloc[-1]} images)")
    log(f"  Imbalance ratio    : {counts.iloc[0] / counts.iloc[-1]:.1f}x")

    # Warn if any class has < 100 images
    small_classes = counts[counts < 100]
    if len(small_classes):
        log(f"\n    Classes with < 100 images (may need oversampling):")
        for cls, cnt in small_classes.items():
            log(f"     {cls}: {cnt}")

    log("\n" + "─" * 60)
    log("  WEIGHTING BALANCE (T1 / T1C+ / T2)")
    log("─" * 60)
    log(df["weighting"].value_counts().to_string())

    log("\n" + "─" * 60)
    log("  LOCATION TAGS")
    log("─" * 60)

    all_tags = []
    for tags in df["location_tags"]:
        if tags:
            all_tags.extend(tags.split("|"))

    if all_tags:
        tag_counts = pd.Series(all_tags).value_counts()
        log(tag_counts.to_string())
    else:
        log("  No location tags found — location field is empty for all entries.")

    df.to_csv(output_dir / "full_manifest.csv", index=False)
    log("\n" + "─" * 60)
    log(f"   Full manifest saved → full_manifest.csv  ({len(df):,} rows)")

    log("\n" + "─" * 60)
    log("  RECOMMENDED SPLIT SIZES (80 / 10 / 10 stratified)")
    log("─" * 60)
    total = len(df)
    log(f"  Train : ~{int(total * 0.8):,}")
    log(f"  Val   : ~{int(total * 0.1):,}")
    log(f"  Test  : ~{int(total * 0.1):,}  (lock this — never touch until final eval)")

    log("\n" + "=" * 60)
    log("  INSPECTION COMPLETE")
    log("=" * 60)

    # Save summary text
    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n All outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()