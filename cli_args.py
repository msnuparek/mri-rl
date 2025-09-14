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
    ap.add_argument("--val-list", default="data/splits/train.txt")
    ap.add_argument("--budget", type=int, default=144)
    ap.add_argument("--timesteps", type=int, default=100000)
    ap.add_argument("--recon", default="checkpoints/recon_largeunet_ssiml1.pth")
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
    ap.add_argument("--accel", type=int, default=3)
    ap.add_argument("--acs",   type=int, default=16)
    ap.add_argument("--loss",  choices=["ssim", "ssim_l1"], default="ssim_l1")
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--save",  default="checkpoints/recon_largeunet_ssiml1.pth")
    add_common_device(ap)
    _apply_global_overrides(ap)
    return ap.parse_args()



def parse_recon_test_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Inference a trained reconstructor on a validation slice.")
    ap.add_argument("--val-list", default="data/splits/val.txt")
    ap.add_argument("--index", type=int, default=20, help="Global slice index in the validation dataset")
    ap.add_argument("--recon", default="checkpoints/recon_largeunet_ssiml1.pth", help="Path to reconstructor checkpoint (.pth)")
    add_common_device(ap)
    ap.add_argument("--out", default="outputs/recon_vis.png", help="Output figure path")
    ap.add_argument("--accel", type=int, default=3)
    ap.add_argument("--acs", type=int, default=16)
    ap.add_argument("--arch", choices=["unet_large", "cascaded", "small"], default="unet_large")
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--use-se", type=int, default=1)
    ap.add_argument("--p-drop", type=float, default=0.0)
    ap.add_argument("--stages", type=int, default=3, help="for 'cascaded' arch")
    ap.add_argument("--tie-weights", type=int, default=0, help="for 'cascaded' arch")
    ap.add_argument("--disp", choices=["gt","joint","per"], default="joint")
    ap.add_argument("--pctl", type=float, default=99.5)
    _apply_global_overrides(ap)
    return ap.parse_args()




def parse_visualize_rl_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Visualize RL episode on a validation slice")
    ap.add_argument("--val-list", default='data/splits/val.txt')
    ap.add_argument("--index", type=int, default=20, help="slice index in the validation set")
    ap.add_argument("--rl", default='checkpoints/ppo_maskable.zip', help="path to MaskablePPO .zip checkpoint")
    ap.add_argument("--recon", default="checkpoints/recon_largeunet_ssiml1.pth", help="optional reconstructor checkpoint (.pth)")
    ap.add_argument("--budget", type=int, default=144)
    ap.add_argument("--device", default=_default_device())
    ap.add_argument("--acs", type=int, default=16)
    ap.add_argument("--start-with-acs", type=int, default=1)
    ap.add_argument("--out-gif", default="outputs/rl_episode.gif")
    ap.add_argument("--out-grid", default="outputs/rl_episode_grid.png")
    ap.add_argument("--fps", type=int, default=2)
    _apply_global_overrides(ap)
    return ap.parse_args()


def parse_make_scan_rl_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Reconstruct full DICOM series (all slices of file containing --index)")
    ap.add_argument("--val-list", default="data/splits/val.txt")
    ap.add_argument("--index", type=int, default=50, help="slice index in the validation set (to choose file)")
    ap.add_argument("--rl", default="checkpoints/ppo_maskable.zip", help="MaskablePPO .zip checkpoint")
    ap.add_argument("--recon", default="checkpoints/recon_largeunet_ssiml1.pth", help="reconstructor checkpoint (.pth)")
    ap.add_argument("--budget", type=int, default=144)
    ap.add_argument("--acs", type=int, default=16)
    ap.add_argument("--start-with-acs", type=int, default=1)
    add_common_device(ap)
    ap.add_argument("--out-dir", default="outputs/scan_rl")
    ap.add_argument("--pctl", type=float, default=99.5, help="display percentile for PNGs")
    _apply_global_overrides(ap)
    return ap.parse_args()


def parse_generate_splits_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Generate train/val split lists from directories containing .h5 files")
    ap.add_argument("--train-root", default="D:/DP/data/knee_singlecoil_test")
    ap.add_argument("--val-root", default="D:/DP/data/knee_singlecoil_test")
    ap.add_argument("--out-dir", default="data/splits")
    _apply_global_overrides(ap)
    return ap.parse_args()
