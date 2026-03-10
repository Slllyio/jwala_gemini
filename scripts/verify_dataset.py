#!/usr/bin/env python3
"""
scripts/verify_dataset.py
==========================
Sanity-check suite for the VanAgni training dataset.

Checks performed:
  1.  Manifest exists and has correct columns
  2.  Train / Val / Test split counts and ratios
  3.  Tier distribution
  4.  Random patch .npz integrity (shapes, dtypes, value ranges)
  5.  Label quality (fire-pixel density per tier)
  6.  HLS temporal stack (no all-NaN frames)
  7.  Weather features (no extreme outliers after normalisation)
  8.  Burn-age channel statistics
  9.  Terrain channel statistics
  10. Class-imbalance ratio check

Usage:
  python scripts/verify_dataset.py                         # all checks
  python scripts/verify_dataset.py --n-samples 20         # check 20 patches
  python scripts/verify_dataset.py --split val            # only val patches
  python scripts/verify_dataset.py --strict               # raise on any warning
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT         = Path(__file__).resolve().parent.parent
DATA_LAKE    = ROOT / "data_lake"
OUT_DIR      = DATA_LAKE / "training_patches"
MANIFEST_CSV = OUT_DIR / "manifest.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

PASS  = "[PASS]"
WARN  = "[WARN]"
FAIL  = "[FAIL]"

_issues: list = []

def check(cond: bool, msg: str, strict: bool = False) -> bool:
    tag = PASS if cond else (FAIL if strict else WARN)
    print(f"  {tag}  {msg}")
    if not cond:
        _issues.append(msg)
    return cond


# ─────────────────────────────────────────────────────────────────────────────
# 1. Manifest
# ─────────────────────────────────────────────────────────────────────────────

def check_manifest(strict: bool) -> pd.DataFrame:
    print("\n[1] Manifest file")
    check(MANIFEST_CSV.exists(), f"manifest.csv exists at {MANIFEST_CSV}", strict=strict)
    if not MANIFEST_CSV.exists():
        print("    Cannot continue — no manifest found.")
        sys.exit(1)

    df = pd.read_csv(MANIFEST_CSV)
    expected_cols = {"path", "patch_id", "tile", "fire_date", "split", "tier",
                     "fire_pixels", "is_fire"}
    missing = expected_cols - set(df.columns)
    check(len(missing) == 0, f"All expected columns present (missing: {missing})", strict=strict)
    check(len(df) > 0, f"Manifest has {len(df)} rows", strict=strict)

    print(f"\n    Total patches : {len(df)}")
    for col in ["split", "tier", "is_fire"]:
        print(f"\n    {col} distribution:")
        for k, v in df[col].value_counts().items():
            pct = v / len(df) * 100
            print(f"      {str(k):15s}: {v:5d}  ({pct:.1f}%)")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Split Ratios
# ─────────────────────────────────────────────────────────────────────────────

def check_splits(df: pd.DataFrame, strict: bool) -> None:
    print("\n[2] Split ratios")
    splits = df["split"].value_counts()
    total  = len(df)

    check("train" in splits, "Train split present", strict=strict)
    check("val"   in splits, "Val   split present", strict=strict)
    check("test"  in splits, "Test  split present", strict=strict)

    if "train" in splits:
        tr_pct = splits["train"] / total * 100
        check(tr_pct >= 50, f"Train >= 50% of patches  ({tr_pct:.1f}%)", strict=False)

    # Fire / negative ratio per split
    for sp in ["train", "val", "test"]:
        sub = df[df["split"] == sp]
        if len(sub) == 0:
            continue
        n_fire = int(sub["is_fire"].sum())
        n_neg  = len(sub) - n_fire
        ratio  = n_neg / max(n_fire, 1)
        check(
            0 < ratio < 10,
            f"{sp:5s}: {n_fire} fire + {n_neg} neg  (ratio neg:pos = {ratio:.1f}x)",
            strict=False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Patch .npz Integrity
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_SHAPES = {
    "hls":        lambda d: d.ndim == 4 and d.shape[0] == 6 and d.shape[1] >= 1,
    "indices":    lambda d: d.ndim == 4 and d.shape[0] == 3,
    "weather":    lambda d: d.ndim == 1 and d.shape[0] == 10,
    "weather_7d": lambda d: d.ndim == 2 and d.shape[0] == 10 and d.shape[1] == 7,
    "terrain":    lambda d: d.ndim == 3 and d.shape[0] == 4,
    "burn_age":   lambda d: d.ndim == 3 and d.shape[0] == 3,
    "landcover":  lambda d: d.ndim == 3 and d.shape[0] == 11,
    "label":      lambda d: d.ndim == 2,
    "weight":     lambda d: d.ndim == 2,
}

EXPECTED_RANGES = {
    "hls":        (-6.0,  6.0),   # normalised bands
    "indices":    (-1.0,  1.0),   # NDVI/NBR/BSI
    "weather":    (-5.0,  5.0),   # normalised scalars
    "weather_7d": (-5.0,  5.0),
    "terrain":    (-1.0,  1.5),   # elev/slope [0,1], sin/cos[-1,1]
    "burn_age":   ( 0.0,  1.0),   # sigmoid-based encodings
    "landcover":  ( 0.0,  1.0),   # one-hot
}


def check_patch(path: Path, strict: bool) -> dict:
    """Load one patch and run integrity checks. Returns summary stats."""
    data = np.load(str(path), allow_pickle=True)

    stats = {}
    for key, shape_ok in EXPECTED_SHAPES.items():
        if key not in data:
            check(False, f"{path.name}: missing array '{key}'", strict=strict)
            continue
        arr = data[key]
        check(shape_ok(arr), f"{path.name}: {key} shape={arr.shape}", strict=strict)

        if key in EXPECTED_RANGES:
            lo, hi = EXPECTED_RANGES[key]
            arr_f  = arr.astype(np.float32)
            actual_lo = float(np.nanmin(arr_f))
            actual_hi = float(np.nanmax(arr_f))
            in_range  = (actual_lo >= lo - 0.5) and (actual_hi <= hi + 0.5)
            check(
                in_range,
                f"{path.name}: {key} range [{actual_lo:.2f}, {actual_hi:.2f}]  "
                f"expected [{lo}, {hi}]",
                strict=False,
            )

    # Label checks
    if "label" in data:
        lbl     = data["label"].astype(np.int64)
        vals    = np.unique(lbl)
        fire_px = int((lbl == 1).sum())
        bg_px   = int((lbl == 0).sum())
        ok_vals = set(vals).issubset({0, 1, 255, -1})
        check(ok_vals, f"{path.name}: label values = {vals}", strict=strict)
        stats["fire_pixels"] = fire_px
        stats["bg_pixels"]   = bg_px

    # HLS temporal stack — check no frame is entirely zero (missing frame fill-zero is OK)
    if "hls" in data:
        hls = data["hls"]   # (6, T, H, W)
        for t in range(hls.shape[1]):
            frame_std = float(np.std(hls[:, t]))
            check(
                frame_std > 0.01,
                f"{path.name}: HLS frame t={t} std={frame_std:.4f} (>0.01 expected)",
                strict=False,
            )

    # Weight sanity
    if "weight" in data:
        w = data["weight"].astype(np.float32)
        check(float(w.max()) <= 3.5, f"{path.name}: max weight={w.max():.2f} (<=3.5)", strict=False)
        check(float(w.min()) >= 0.0, f"{path.name}: min weight={w.min():.2f} (>=0.0)", strict=strict)

    # Meta JSON
    if "meta" in data:
        try:
            m = json.loads(str(data["meta"]))
            check("tier"       in m, f"{path.name}: meta has 'tier'",  strict=False)
            check("fire_date"  in m, f"{path.name}: meta has 'fire_date'", strict=False)
            check("pred_horizon" in m, f"{path.name}: meta has 'pred_horizon'", strict=False)
        except Exception as e:
            check(False, f"{path.name}: meta parse error: {e}", strict=False)

    return stats


def check_patches(df: pd.DataFrame, n_samples: int, split: Optional[str],
                  strict: bool) -> None:
    print(f"\n[3] Patch .npz integrity  (sampling {n_samples} patches)")

    sub = df if split is None else df[df["split"] == split]
    if len(sub) == 0:
        print(f"    No patches found for split='{split}'")
        return

    # Sample evenly across tiers
    sample = sub.groupby("tier", group_keys=False).apply(
        lambda g: g.sample(min(len(g), max(1, n_samples // sub["tier"].nunique())),
                            random_state=42)
    ).head(n_samples)

    fire_px_by_tier: dict = {}
    for _, row in sample.iterrows():
        path = ROOT / row["path"]
        if not path.exists():
            check(False, f"Patch file missing: {path}", strict=strict)
            continue
        stats = check_patch(path, strict)
        tier  = str(row["tier"])
        if tier not in fire_px_by_tier:
            fire_px_by_tier[tier] = []
        fire_px_by_tier[tier].append(stats.get("fire_pixels", 0))

    print(f"\n    Mean fire pixels per tier:")
    for tier, counts in sorted(fire_px_by_tier.items()):
        mean_px = np.mean(counts) if counts else 0
        print(f"      {tier:12s}: {mean_px:8.0f} px  (n={len(counts)})")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Class Imbalance
# ─────────────────────────────────────────────────────────────────────────────

def check_imbalance(df: pd.DataFrame, strict: bool) -> None:
    print("\n[4] Class imbalance (fire-pixel density)")
    fire_df = df[df["is_fire"] == 1]
    neg_df  = df[df["is_fire"] == 0]
    ratio   = len(neg_df) / max(len(fire_df), 1)
    check(ratio < 10, f"Patch-level neg:pos ratio = {ratio:.1f}x  (target <10x)", strict=False)
    check(len(fire_df) >= 100, f"At least 100 fire patches ({len(fire_df)})", strict=False)
    check(len(neg_df)  >= 50,  f"At least 50 negative patches ({len(neg_df)})", strict=False)


# ─────────────────────────────────────────────────────────────────────────────
# 5. File-Count Reconciliation
# ─────────────────────────────────────────────────────────────────────────────

def check_file_counts(df: pd.DataFrame, strict: bool) -> None:
    print("\n[5] File count reconciliation")
    n_missing = 0
    n_checked = min(len(df), 500)
    sample    = df.sample(n_checked, random_state=0)
    for _, row in sample.iterrows():
        p = ROOT / row["path"]
        if not p.exists():
            n_missing += 1
    check(
        n_missing == 0,
        f"{n_missing}/{n_checked} patch files missing (sampled check)",
        strict=strict,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

# (Optional imported at top of file)

def main() -> None:
    global MANIFEST_CSV
    parser = argparse.ArgumentParser(description="Verify VanAgni training dataset")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Number of patches to inspect (default: 10)")
    parser.add_argument("--split", type=str, default=None,
                        choices=["train", "val", "test"],
                        help="Only verify patches from this split")
    parser.add_argument("--strict", action="store_true",
                        help="Treat warnings as failures")
    parser.add_argument("--manifest", type=str, default=str(MANIFEST_CSV),
                        help=f"Path to manifest.csv (default: {MANIFEST_CSV})")
    args = parser.parse_args()

    MANIFEST_CSV = Path(args.manifest)

    print("=" * 65)
    print("VanAgni Dataset Verification")
    print("=" * 65)

    df = check_manifest(args.strict)
    check_splits(df, args.strict)
    check_patches(df, args.n_samples, args.split, args.strict)
    check_imbalance(df, args.strict)
    check_file_counts(df, args.strict)

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    if _issues:
        print(f"VERIFICATION COMPLETE — {len(_issues)} issue(s) found:")
        for issue in _issues:
            print(f"  - {issue}")
        if args.strict:
            sys.exit(1)
    else:
        print("VERIFICATION COMPLETE — all checks passed")
    print("=" * 65)


if __name__ == "__main__":
    main()
