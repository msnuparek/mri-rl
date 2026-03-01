"""
Centralized CLI argument definitions for all scripts.

Each script should import and call the appropriate parse_* function
instead of defining its own ArgumentParser. This keeps defaults and help
texts consistent across the repository.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    import torch  # for default device detection
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    _HAS_TORCH = False


def _default_device() -> str:
    if _HAS_TORCH and getattr(torch, "cuda", None) and torch.cuda.is_available():
        return "cuda"
    return "cpu"


# -----------------------------
# Shared/common option helpers
# -----------------------------
def add_common_device(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--device", default=_default_device())


def _resolve_global_args_path() -> Path | None:
    """Pick a global args JSON file if present.

    Priority:
    - env MRI_RL_ARGS (file path)
    - repo root next to this file: args.json, args_global.json (first found)
    """
    env_p = os.getenv("MRI_RL_ARGS")
    if env_p:
        p = Path(env_p)
        return p if p.is_file() else None
    here = Path(__file__).resolve().parent
    for name in ("args.json", "args_global.json"):
        cand = here / name
        if cand.is_file():
            return cand
    return None


def _load_global_overrides() -> dict[str, object]:
    path = _resolve_global_args_path()
    if not path:
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):  # pragma: no cover
            return {}
        # normalize keys to argparse dest format (dashes -> underscores)
        return {str(k).replace("-", "_"): v for k, v in data.items()}
    except Exception:
        return {}


def _apply_global_overrides(ap: argparse.ArgumentParser) -> None:
    """Apply overrides from a global JSON file to this parser's defaults.

    CLI options still take precedence over defaults.
    Only keys that match this parser's destinations are applied.
    """
    overrides = _load_global_overrides()
    if not overrides:
        return
    # Collect dest names present in this parser
    dests = {a.dest for a in ap._actions if getattr(a, "dest", None)}
    filtered = {k: v for k, v in overrides.items() if k in dests}
    if filtered:
        ap.set_defaults(**filtered)


# -----------------------------
# Script-specific parsers
# -----------------------------
def parse_train_rl_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Train RL (MaskablePPO) selector over k-space columns")
    ap.add_argument("--train-list", default="data/splits/train.txt")
    ap.add_argument("--val-list", default="data/splits/val.txt")
    ap.add_argument("--budget", type=int, default=37)
    ap.add_argument("--timesteps", type=int, default=100000)
    ap.add_argument("--recon", default="checkpoints/recon_largeunet_ssiml1_sag_6R.pth")
    ap.add_argument("--acs", type=int, default=16)
    ap.add_argument("--start-with-acs", type=int, default=1)
    _apply_global_overrides(ap)
    return ap.parse_args()


def parse_train_recon_ssim_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Train reconstructor with SSIM/SSIM+L1 loss")
    ap.add_argument("--train-list", default="data/splits/train.txt")
    ap.add_argument("--val-list",   default="data/splits/val.txt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--accel", type=int, default=6)
    ap.add_argument("--acs",   type=int, default=16)
    ap.add_argument("--loss",  choices=["ssim", "ssim_l1"], default="ssim_l1")
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--save",  default="checkpoints/recon_largeunet_ssiml1_6R.pth")
    add_common_device(ap)
    _apply_global_overrides(ap)
    return ap.parse_args()

def parse_train_recon_dicom_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Train reconstructor on DICOM slices with optional caching")
    ap.add_argument("--train-list", default="data/splits/train.txt")
    ap.add_argument("--val-list",   default="data/splits/val.txt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--accel", type=int, default=6)
    ap.add_argument("--acs",   type=int, default=16)
    ap.add_argument("--loss",  choices=["ssim", "ssim_l1"], default="ssim_l1")
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--save",  default="checkpoints/recon_dicom_ssiml1_sag_8R.pth")
    ap.add_argument("--center-crop", type=int, default=320)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--cache-items", type=int, default=4096, help="Maximum cached slices per worker (0 disables)")
    ap.add_argument("--no-cache", action="store_true", help="Disable in-memory slice caching")
    ap.add_argument("--return-kspace", action="store_true", help="Return full k-space tensors from the loader")
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision training")
    ap.add_argument("--warm-cache", action="store_true", help="Iterate datasets once to populate caches before training")
    add_common_device(ap)
    _apply_global_overrides(ap)
    return ap.parse_args()



def parse_visualize_rl_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Visualize RL episode on a validation slice")
    ap.add_argument("--val-list", default='data/splits/val.txt')
    ap.add_argument("--index", type=int, default=202, help="slice index in the validation set")
    ap.add_argument("--rl", default='checkpoints/ppo_maskable_sag_8R.zip', help="path to MaskablePPO .zip checkpoint")
    ap.add_argument("--recon", default="checkpoints/recon_largeunet_ssiml1_sag_8R.pth", help="optional reconstructor checkpoint (.pth)")
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--device", default=_default_device())
    ap.add_argument("--acs", type=int, default=16)
    ap.add_argument("--start-with-acs", type=int, default=1)
    ap.add_argument("--use-pseudo-ref", type=int, default=1,
                    help="Použít pseudo-referenci místo GT pro reward (1=ano, 0=ne).")
    ap.add_argument("--pseudo-ref-acs", type=int, default=16,
                    help="Velikost ACS regionu pro pseudo-referenci.")
    ap.add_argument("--out-gif", default="outputs/rl_episode_sag.gif")
    ap.add_argument("--out-grid", default="outputs/rl_episode_grid_sag.png")
    ap.add_argument("--fps", type=int, default=2)
    _apply_global_overrides(ap)
    return ap.parse_args()


def parse_make_scan_rl_dir_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Reconstruct DICOM series from an explicit directory path")
    ap.add_argument("--dicom-dir", default=r"C:\Users\uzivatel\Plocha\experiment zipy\DICOMDIR-Snuparek-Matej-none\DATA\DICOM", help="Directory containing DICOM slices (recursively scanned)")
    ap.add_argument("--index", type=int, default=0, help="slice index within the provided directory")
    ap.add_argument("--rl", default="checkpoints/ppo_maskable_sag_4R.zip", help="MaskablePPO .zip checkpoint")
    ap.add_argument("--recon", default="checkpoints/recon_largeunet_ssiml1_sag_4R.pth", help="reconstructor checkpoint (.pth)")
    ap.add_argument("--budget", type=int, default=160)
    ap.add_argument("--acs", type=int, default=16)
    ap.add_argument("--start-with-acs", type=int, default=1)
    ap.add_argument("--use-pseudo-ref", type=int, default=1,
                    help="Use pseudo-reference instead of GT for reward (1=enabled, 0=disabled)")
    ap.add_argument("--pseudo-ref-acs", type=int, default=16,
                    help="ACS region size for pseudo-reference (larger=better quality)")
    add_common_device(ap)
    ap.add_argument("--out-dir", default=r"C:\Users\uzivatel\Plocha\DP_test_set\4R\02_4R")
    ap.add_argument("--pctl", type=float, default=99.5, help="display percentile for PNGs")
    _apply_global_overrides(ap)
    return ap.parse_args()


def parse_visualize_compare_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Side-by-side: RL policy vs baseline k-space sampling")
    ap.add_argument("--val-list", default="data/splits/val.txt")
    ap.add_argument("--index", type=int, default=202,
                    help="Slice index in the validation dataset")
    ap.add_argument("--rl", default="checkpoints/ppo_maskable_sag_6R.zip",
                    help="MaskablePPO .zip checkpoint")
    ap.add_argument("--recon", default="checkpoints/recon_largeunet_ssiml1_sag_6R.pth",
                    help="Reconstructor checkpoint (.pth)")
    ap.add_argument("--budget", type=int, default=53)
    ap.add_argument("--acs", type=int, default=16)
    ap.add_argument("--start-with-acs", type=int, default=1)
    ap.add_argument("--use-pseudo-ref", type=int, default=1,
                    help="Use pseudo-reference for RL reward during the episode (1=yes, 0=no)")
    ap.add_argument("--pseudo-ref-acs", type=int, default=16,
                    help="ACS region size used to build the pseudo-reference")
    ap.add_argument("--baseline", choices=["uniform", "random", "lowfreq"],
                    default="uniform",
                    help="Baseline sampling strategy: uniform (equidistant), "
                         "random (shuffled, seed=42), lowfreq (centre-out)")
    ap.add_argument("--out-gif", default="outputs/compare_rl_vs_baseline.gif")
    ap.add_argument("--out-grid", default="outputs/compare_rl_vs_baseline_grid.png")
    ap.add_argument("--fps", type=int, default=2)
    add_common_device(ap)
    _apply_global_overrides(ap)
    return ap.parse_args()


def parse_generate_splits_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Generate train/val list files for DICOM directory datasets")
    ap.add_argument("--train-root", default="D:/DP/data/knee_singlecoil_train")
    ap.add_argument("--val-root", default="D:/DP/data/knee_singlecoil_val")
    ap.add_argument("--out-dir", default="data/splits")
    _apply_global_overrides(ap)
    return ap.parse_args()

