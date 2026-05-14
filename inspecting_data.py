"""
inspect_dataset.py
------------------

"""

import re
import json
import argparse
import unicodedata
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── CONSTANTS 

CLASS_NAMES = [
    "Astrocytoma T1",        "Astrocytoma T1C+",        "Astrocytoma T2",
    "Ependymoma T1",         "Ependymoma T1C+",          "Ependymoma T2",
    "Glioma T1",             "Glioma T1C+",              "Glioma T2",
    "Hemangiopericytoma T1", "Hemangiopericytoma T1C+",  "Hemangiopericytoma T2",
    "Meningioma T1",         "Meningioma T1C+",          "Meningioma T2",
    "Neurocytoma T1",        "Neurocytoma T1C+",          "Neurocytoma T2",
    "Normal T1",             "Normal T1C+",               "Normal T2",
    "Oligodendroglioma T1",  "Oligodendroglioma T1C+",   "Oligodendroglioma T2",
    "Other T1",              "Other T1C+",                "Other T2",
    "Schwannoma T1",         "Schwannoma T1C+",           "Schwannoma T2",
]

TUMOR_TYPES = [
    "Astrocytoma", "Ependymoma", "Glioma", "Hemangiopericytoma",
    "Meningioma",  "Neurocytoma", "Normal", "Oligodendroglioma",
    "Other",       "Schwannoma",
]

CLASS_TO_IDX  = {c: i for i, c in enumerate(CLASS_NAMES)}
TUMOR_TO_IDX  = {t: i for i, t in enumerate(TUMOR_TYPES)}
NORMAL_CLASSES = frozenset({"Normal T1", "Normal T1C+", "Normal T2"})
IMAGE_SIZE     = 512


_WEIGHTING_RE = re.compile(r"\b(T1C\+|T1|T2)\b")
 

def extract_weighting(class_label: str) -> str:
    m = _WEIGHTING_RE.search(class_label)
    return m.group(1) if m else "Unknown"


def extract_tumor_type(class_label: str) -> str:
    return _WEIGHTING_RE.sub("", class_label).strip()


def normalise_key(raw_key: str) -> str:
    """Windows backslash → forward slash."""
    return raw_key.replace("\\", "/")


def resolve_path(data_root: Path, rel_path: str) -> Optional[Path]:
    rel_path = (
        rel_path.replace("—", "-")
                .replace("–", "-")
                .replace("−", "-")
    )
    
    for norm in ["NFC", "NFD", "NFKC", "NFKD"]:
        candidate = data_root / unicodedata.normalize(norm, rel_path)
        if candidate.exists():
            return candidate
    # Direct attempt last (no normalisation)
    direct = data_root / rel_path
    if direct.exists():
        return direct
    return None


# ── MANIFEST BUILDER ──────────────────────────────────────────────────────────

def build_manifest(metadata_path: str, data_root: str) -> tuple[pd.DataFrame, list]:
    
    data_root = Path(data_root)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    records = []
    missing = []
    skipped = []

    for raw_key, meta in metadata.items():
        rel_path    = normalise_key(raw_key)
        class_label = meta.get("class", "").strip()

        if class_label not in CLASS_TO_IDX:
            skipped.append((rel_path, class_label))
            continue

        abs_path = resolve_path(data_root, rel_path)
        if abs_path is None:
            missing.append(rel_path)
            continue

        point   = meta.get("point", {})
        px      = point.get("x")
        py      = point.get("y")
        w       = meta.get("width",  IMAGE_SIZE)
        h       = meta.get("height", IMAGE_SIZE)

        px_norm = round(float(px) / w, 6) if px is not None else 0.5
        py_norm = round(float(py) / h, 6) if py is not None else 0.5

        is_normal  = class_label in NORMAL_CLASSES
        tumor_type = extract_tumor_type(class_label)

        records.append({
            "rel_path"       : rel_path,
            "abs_path"       : str(abs_path),
            "class"          : class_label,
            "class_idx"      : CLASS_TO_IDX[class_label],
            "tumor_type"     : tumor_type,
            "tumor_type_idx" : TUMOR_TO_IDX.get(tumor_type, -1),
            "weighting"      : extract_weighting(class_label),
            "is_normal"      : is_normal,
            "loc_mask"       : 0.0 if is_normal else 1.0,
            "point_x"        : px,
            "point_y"        : py,
            "point_x_norm"   : px_norm,
            "point_y_norm"   : py_norm,
            "width"          : w,
            "height"         : h,
            "location_tags"  : "|".join(meta.get("location", [])),
            "description"    : meta.get("description", "").strip(),
        })

    if skipped:
        warnings.warn(f"{len(skipped)} entries skipped (unknown class): {skipped[:3]}")

    df = pd.DataFrame(records)
    return df, missing


def run_inspection(df: pd.DataFrame, missing: list, output_dir: Path) -> None:
    
    lines = []
    def log(msg=""):
        print(msg)
        lines.append(str(msg))

    log("=" * 62)
    log("  NEUROSIGHT DATASET INSPECTION + MANIFEST BUILD")
    log("=" * 62)

   #for missing files - I am hoping we dont have Jesu
    log(f"\n Metadata entries    : {len(df) + len(missing):,}")
    log(f" Loaded into manifest: {len(df):,}")
    log(f" Missing on disk     : {len(missing)}")

    if missing:
        missing_df = pd.DataFrame({"missing_from_disk": missing})
    missing_df.to_csv(output_dir / "missing_files.csv", index=False)

    # DIAGNOSTIC
    print("\n===== PATH DIAGNOSTIC =====")

    missing_example = missing[0]

    print("\nMETADATA STRING:")
    print(repr(missing_example))

    print("\nPOTENTIAL MATCHES ON DISK:")

    data_root = Path(df["abs_path"].iloc[0]).parent

    for p in data_root.rglob("*001.jpg"):
        if "Diffuse midline glioma" in p.name:
            print(repr(p.name))

    # ORIGINAL LOGS
    log(f"   → Saved to missing_files.csv")
    log(f"   Known cause: em-dash (–) vs hyphen (-) in filenames")
    log(f"   Affected (first 3): {missing[:3]}")

   
    log("\n" + "─" * 62)
    log("  IMAGE DIMENSIONS")
    log("─" * 62)
    dim_counts = df.groupby(["width", "height"]).size().reset_index(name="count")
    log(dim_counts.to_string(index=False))
    if len(dim_counts) == 1:
        log("   All images uniformly 512×512")
    else:
        log("    WARNING: Mixed dimensions detected!")

   
    log("\n" + "─" * 62)
    log("  CLASS DISTRIBUTION (30 classes)")
    log("─" * 62)
    class_dist = (
        df.groupby(["tumor_type", "weighting", "class"])
          .size()
          .reset_index(name="count")
          .sort_values(["tumor_type", "weighting"])
    )
    log(class_dist.to_string(index=False))
    class_dist.to_csv(output_dir / "class_distribution.csv", index=False)

    log("\n" + "─" * 62)
    log("  HIGH-LEVEL SUMMARY")
    log("─" * 62)
    log(f"  Total images            : {len(df):,}")
    log(f"  Unique classes          : {df['class'].nunique()}")
    log(f"  Unique tumor types      : {df['tumor_type'].nunique()}")
    log(f"  Normal images           : {df['is_normal'].sum():,}")
    log(f"  Abnormal images         : {(~df['is_normal']).sum():,}")
    log(f"  Images with location tag: {(df['location_tags'] != '').sum():,}")


    log("\n" + "─" * 62)
    log("  WEIGHTING BALANCE")
    log("─" * 62)
    log(df["weighting"].value_counts().to_string())

  
    log("\n" + "─" * 62)
    log("  CLASS BALANCE")
    log("─" * 62)
    counts = df["class"].value_counts()
    log(f"  Most common  : {counts.index[0]}  ({counts.iloc[0]})")
    log(f"  Least common : {counts.index[-1]}  ({counts.iloc[-1]})")
    log(f"  Imbalance    : {counts.iloc[0] / counts.iloc[-1]:.1f}x")

    small = counts[counts < 100]
    if len(small):
        log(f"\n    Classes with < 100 images (will be heavily upsampled):")
        for cls, cnt in small.items():
            log(f"     {cls}: {cnt}")


    log("\n" + "─" * 62)
    log("  LESION POINT STATISTICS (abnormal only)")
    log("─" * 62)
    ab = df[~df["is_normal"]]
    log(f"  X — mean: {ab['point_x'].mean():.1f}  std: {ab['point_x'].std():.1f}"
        f"  min: {ab['point_x'].min()}  max: {ab['point_x'].max()}")
    log(f"  Y — mean: {ab['point_y'].mean():.1f}  std: {ab['point_y'].std():.1f}"
        f"  min: {ab['point_y'].min()}  max: {ab['point_y'].max()}")

    threshold = 10
    suspicious = ab[
        (abs(ab["point_x"] - 256) < threshold) &
        (abs(ab["point_y"] - 256) < threshold)
    ]
    log(f"\n  Points within {threshold}px of centre in abnormal classes: {len(suspicious)}")
    log(f"  Note: these are anatomically valid midline tumors (Colloid cysts,")
    log(f"        Diffuse midline gliomas, Central neurocytomas) — do NOT exclude.")

    point_stats = (
        ab.groupby("class")[["point_x_norm", "point_y_norm"]]
          .agg(["mean", "std"])
          .round(4)
    )
    point_stats.to_csv(output_dir / "points_stats.csv")
    log(f"\n  Per-class point stats → points_stats.csv")

    
    log("\n" + "─" * 62)
    log("  LOCATION TAGS")
    log("─" * 62)
    all_tags = []
    for tags in df["location_tags"]:
        if tags:
            all_tags.extend(tags.split("|"))
    if all_tags:
        log(pd.Series(all_tags).value_counts().to_string())
    else:
        log("  No location tags found.")

 
    log("\n" + "─" * 62)
    log("  RECOMMENDED SPLIT SIZES (80 / 10 / 10 stratified)")
    log("─" * 62)
    n = len(df)
    log(f"  Train : ~{int(n * 0.8):,}")
    log(f"  Val   : ~{int(n * 0.1):,}")
    log(f"  Test  : ~{int(n * 0.1):,}  ← lock until final eval")

    log("\n" + "=" * 62)
    log(f"   full_manifest.csv saved  ({len(df):,} rows)")
    log("  Pass this file to dataset.py — nothing else reads metadata.json")
    log("=" * 62)



    # Save summary
    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── MAIN ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="NeuroSight — Dataset inspection + manifest builder"
    )
    parser.add_argument("--data_root", required=True,
                        help="Root folder containing class subfolders")
    parser.add_argument("--metadata",  required=True,
                        help="Path to metadata.json")
    parser.add_argument("--output",    default="./inspection_report",
                        help="Output directory for all artefacts")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, missing = build_manifest(args.metadata, args.data_root)

  
    manifest_path = output_dir / "full_manifest.csv"
    df.to_csv(manifest_path, index=False)

    
    run_inspection(df, missing, output_dir)

    print(f"\n All outputs → {output_dir.resolve()}")
    print(f" Manifest    → {manifest_path.resolve()}")
    print(f"\nNext step:")
    print(f"  python dataset.py --manifest {manifest_path} --splits ./data/splits")


if __name__ == "__main__":
    main()