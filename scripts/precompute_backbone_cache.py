#!/usr/bin/env python3
"""
scripts/precompute_backbone_cache.py
=====================================
Pre-compute frozen backbone features (blocks 0-27) for all training patches.

This eliminates ~79% of forward-pass computation during training by caching
the deterministic output of the frozen prefix. At training time, only blocks
28-31 + decoder need to run.

Cached per patch:
  block27_out  float32  (N_tokens, embed_dim)  = (197, 1280) for 600M
               N_tokens = 1 (CLS) + 196 (14x14 spatial)
               ~1.0 MB per patch

Total cache:  711 patches x ~1.0 MB = ~700 MB

Usage:
  py -3.12 scripts/precompute_backbone_cache.py
  py -3.12 scripts/precompute_backbone_cache.py --config models/vanaagni/wfire_config.yaml
  py -3.12 scripts/precompute_backbone_cache.py --cache-dir outputs/wfire/backbone_cache
"""

import os
import sys
import logging
import argparse
import time
from pathlib import Path

# Add repo root to sys.path so "from src.model..." imports work
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))

# PROJ fix
def _fix_proj():
    try:
        import pyproj as _pp
        _proj_data = str(Path(_pp.datadir.get_data_dir()))
        os.environ["PROJ_DATA"] = _proj_data
        os.environ["PROJ_LIB"] = _proj_data
        os.environ.pop("GDAL_DATA", None)
    except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Ignored exception: {e}")
_fix_proj()

import numpy as np
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


def get_device():
    try:
        import torch_directml
        return torch_directml.device(0), "DirectML"
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda:0"), "CUDA"
    return torch.device("cpu"), "CPU"


def build_backbone(cfg: dict):
    """Build and return the Prithvi backbone (no decoder)."""
    from src.model.prithvi_eo2_backbone import build_prithvi_eo2

    model_cfg = cfg.get("model", {})
    backbone = build_prithvi_eo2(
        variant=model_cfg.get("eo2_variant", "600M-TL"),
        num_frames=model_cfg.get("num_frames", 3),
        in_chans=model_cfg.get("in_chans", 6),
        img_size=model_cfg.get("img_size", 224),
    )
    return backbone


def precompute(cfg: dict, cache_dir: Path, device: torch.device):
    """Run all patches through frozen blocks 0-27 and cache the output."""

    # Build backbone
    backbone = build_backbone(cfg)
    backbone.to(device)

    # Access the internal backbone's blocks
    inner = backbone._backbone
    if not hasattr(inner, "blocks"):
        raise RuntimeError("Backbone has no 'blocks' attribute")

    depth = len(inner.blocks)
    detach_at = cfg.get("llrd", {}).get("detach_prefix", 28)
    detach_at = min(detach_at, depth)

    log.info(f"Backbone: {depth} blocks, caching output of block {detach_at - 1}")
    log.info(f"Cache dir: {cache_dir}")

    # Build dataset (no augmentation, no sampling)
    from src.data.vanaagni_dataset import VanAgniDataset, vanaagni_collate
    from torch.utils.data import DataLoader

    data_cfg = cfg.get("data", {})
    manifest = ROOT / data_cfg.get("manifest_path", "data_lake/training_patches/manifest.csv")

    all_patches = []
    for split in ["train", "val", "test"]:
        ds = VanAgniDataset(
            manifest_path=manifest,
            split=split,
            augment=False,
            min_fire_pixels=0,
            binary_mode=True,
        )
        all_patches.append(ds)

    total = sum(len(ds) for ds in all_patches)
    log.info(f"Total patches to cache: {total}")

    cache_dir.mkdir(parents=True, exist_ok=True)

    cached = 0
    skipped = 0
    t0 = time.time()

    for ds in all_patches:
        loader = DataLoader(
            ds, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=vanaagni_collate,
        )

        for batch in loader:
            patch_id = batch["patch_id"][0]
            out_path = cache_dir / f"{patch_id}.npy"

            # Skip if already cached
            if out_path.exists():
                skipped += 1
                continue

            hls = batch["hls"].to(device)  # (1, 6, T, H, W)

            with torch.no_grad():
                # Run the TerraTorch forward_features which iterates all blocks
                # and returns list of all block outputs.
                # We only need the output after block (detach_at - 1).
                all_block_outs = inner.forward_features(hls)

                # forward_features returns list[Tensor] of length=depth
                # Each is (B, N_tokens, embed_dim)
                if isinstance(all_block_outs, list):
                    # Take the output of the last frozen block
                    feat = all_block_outs[detach_at - 1]  # (1, 197, 1280)
                else:
                    # If forward_features returns a single tensor (final output),
                    # we can't extract intermediate. Fall back to manual iteration.
                    feat = _manual_forward_prefix(inner, hls, detach_at)

            # Save as numpy (CPU, float32)
            feat_np = feat.squeeze(0).cpu().numpy()  # (197, 1280)
            np.save(str(out_path), feat_np)
            cached += 1

            if (cached + skipped) % 50 == 0:
                elapsed = time.time() - t0
                rate = (cached + skipped) / max(elapsed, 0.01)
                eta = (total - cached - skipped) / max(rate, 0.01)
                log.info(
                    f"  [{cached + skipped:4d}/{total}]  "
                    f"cached={cached}  skipped={skipped}  "
                    f"{rate:.1f} patches/sec  ETA={eta:.0f}s"
                )

    elapsed = time.time() - t0
    log.info(f"\nPrecompute complete: {cached} cached, {skipped} skipped")
    log.info(f"Time: {elapsed:.1f}s ({elapsed/max(total,1):.2f}s/patch)")
    log.info(f"Cache size: {_dir_size_mb(cache_dir):.0f} MB")


def _manual_forward_prefix(inner, x, detach_at):
    """Manually run patch_embed + blocks 0..(detach_at-1)."""
    if x.ndim == 4:
        x = x.unsqueeze(2)  # add time dim if needed

    sample_shape = x.shape[-3:]
    x = inner.patch_embed(x)

    # Position embedding
    if hasattr(inner, "interpolate_pos_encoding"):
        pos_embed = inner.interpolate_pos_encoding(sample_shape)
    else:
        pos_embed = inner.pos_embed

    x = x + pos_embed[:, 1:, :]

    # CLS token
    if hasattr(inner, "cls_token"):
        cls_token = inner.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

    # Run frozen blocks
    for i in range(detach_at):
        x = inner.blocks[i](x)

    return x


def _dir_size_mb(d: Path) -> float:
    total = sum(f.stat().st_size for f in d.glob("*.npy") if f.is_file())
    return total / 1e6


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute frozen backbone features")
    parser.add_argument("--config", type=str,
                        default="models/vanaagni/wfire_config.yaml",
                        help="Training config YAML")
    parser.add_argument("--cache-dir", type=str,
                        default="outputs/wfire/backbone_cache",
                        help="Directory to store cached features")
    args = parser.parse_args()

    cfg_path = ROOT / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device, dev_name = get_device()
    log.info(f"Device: {device} ({dev_name})")

    cache_dir = ROOT / args.cache_dir
    precompute(cfg, cache_dir, device)
