"""Visualize a full RL episode on a validation slice.

Shows how the RL policy progressively selects k-space columns and how the
reconstruction improves step-by-step. Saves a GIF (if imageio available)
plus an optional static grid PNG.

Usage:
  python visualize_rl.py \
    --val-list data/splits/val.txt \
    --index 0 \
    --rl checkpoints/ppo_maskable.zip \
    --recon checkpoints/recon_smallunet.pth \
    --budget 32 \
    --out-gif outputs/rl_episode.gif \
    --out-grid outputs/rl_episode_grid.png
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import imageio
except Exception:  # pragma: no cover
    imageio = None

from stable_baselines3.common.base_class import BaseAlgorithm

from rl.fastmri_loader import SingleCoilKneeDataset
from rl.reconstructor import build_reconstructor, ReconWrapper
from rl.utils_fft import ifft2_centered as ifft2, fft2_centered as fft2
from rl.metrics import nmse, psnr
from rl.env_ssim import KspaceEnv


def magnitude(x: torch.Tensor) -> torch.Tensor:
    """x: [B,2,H,W] -> [B,1,H,W] magnitude."""
    return torch.sqrt(x[:, :1] ** 2 + x[:, 1:2] ** 2 + 1e-12)

def load_reconstructor(path: str, device: torch.device) -> ReconWrapper:
    # vytvoříme stejnou architekturu jako při tréninku
    core = build_reconstructor("unet_large", base=64, use_se=True, p_drop=0.0).to(device)
    model = ReconWrapper(core).to(device)

    if path:
        ckpt = torch.load(path, map_location=device)
        state = ckpt.get("state_dict", ckpt)

        keys = list(state.keys())
        if all(k.startswith("model.") for k in keys):
            # checkpoint uložený z ReconWrapper(model)  -> nahrajeme do wrapperu
            model.load_state_dict(state)
        else:
            # checkpoint uložený z jádra (bez prefixu) -> nahrajeme do core
            core.load_state_dict(state)

        print(f"Loaded recon checkpoint: {path}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_policy(path: str) -> BaseAlgorithm:
    from sb3_contrib.ppo_mask import MaskablePPO

    model = MaskablePPO.load(path, device="auto")
    print(f"Loaded RL policy: {path}")
    return model


def _disp(img: np.ndarray) -> np.ndarray:
    v = np.percentile(img, 99.5) if np.isfinite(img).all() else 1.0
    return np.clip(img / (v + 1e-8), 0, 1)

def render_frame(
    gt_mag: np.ndarray,
    zf_mag: np.ndarray,
    pr_mag: np.ndarray,
    mask_1d: np.ndarray,
    step: int,
    budget_left: int,
    _mse_unused: float | None = None,  # necháme signaturu beze změny
) -> np.ndarray:
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    axs[0].imshow(_disp(gt_mag), cmap="gray");  axs[0].set_title("GT");          axs[0].axis("off")
    axs[1].imshow(_disp(zf_mag), cmap="gray");  axs[1].set_title("Zero-filled"); axs[1].axis("off")
    axs[2].imshow(_disp(pr_mag), cmap="gray");  axs[2].set_title("Recon");       axs[2].axis("off")

    mse_zf = float(np.mean((zf_mag - gt_mag) ** 2))
    mse_pr = float(np.mean((pr_mag - gt_mag) ** 2))
    measured = int(mask_1d.sum())

    axs[3].imshow(mask_1d[np.newaxis, :], cmap="gray", aspect="auto", vmin=0, vmax=1)
    axs[3].set_yticks([])
    axs[3].set_title(
        f"mask | step {step} | left {budget_left}\n"
        f"MSE ZF {mse_zf:.4f}  MSE R {mse_pr:.4f}  meas {measured}"
    )
    axs[3].set_xlabel("k-space columns")

    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()  # RGBA -> RGB
    plt.close(fig)
    return frame



def make_grid(frames: List[np.ndarray], cols: int = 4) -> np.ndarray:
    """Create a tiled grid image from a list of frames."""
    rows = int(np.ceil(len(frames) / cols))
    # resize all to min size to tile nicely
    h = min(f.shape[0] for f in frames)
    w = min(f.shape[1] for f in frames)
    fr = [f[:h, :w] for f in frames]
    grid = np.ones((rows * h, cols * w, 3), dtype=np.uint8) * 255
    for i, f in enumerate(fr):
        r, c = divmod(i, cols)
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = f
    return grid


def parse_args() -> argparse.Namespace:
    from cli_args import parse_visualize_rl_args
    return parse_visualize_rl_args()


def main():
    # Mild safety for Windows OpenMP mixing
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    args = parse_args()
    device = torch.device(args.device)

    # Data + env
    ds = SingleCoilKneeDataset(args.val_list)
    recon = load_reconstructor(args.recon, device)
    env = KspaceEnv(ds, recon, budget=args.budget, device=args.device, acs=int(args.acs), start_with_acs=bool(args.start_with_acs))

    
    # Seed/Reset
    env._i = int(args.index) % len(ds)
    obs, _ = env.reset()

    # Ground truth magnitude from env buffer (already on device)
    gt = env.target_img.permute(2, 0, 1).unsqueeze(0)  # [1,2,H,W]
    gt_mag = magnitude(gt).squeeze().cpu().numpy()

    # Policy
    model = load_policy(args.rl)

    frames: List[np.ndarray] = []
    step = 0
    done = False

    while not done:
        # Recon from current observation (zf)
        zf = torch.from_numpy(obs["zf"]).unsqueeze(0).to(device)  # [1,2,H,W]
        with torch.no_grad():
            pred = recon(zf)  # [1,2,H,W]

            # Apply data-consistency (same as in training)
            W = zf.shape[-1]
            mask4 = torch.from_numpy(obs["mask"].astype(np.float32)).to(device).view(1, 1, 1, W)  # [1,1,1,W]
            k_full4 = env.kspace_full.permute(2, 0, 1).unsqueeze(0)                                  # [1,2,H,W]
            k_pred = fft2(pred.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)                               # [1,2,H,W]
            k_dc   = torch.where(mask4.bool(), k_full4, k_pred)
            pred   = ifft2(k_dc.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)                               # [1,2,H,W]

        zf_mag = magnitude(zf).squeeze().cpu().numpy()
        pr_mag = magnitude(pred).squeeze().cpu().numpy()

        # MSE vs GT (on magnitude)
        mse_now = float(np.mean((pr_mag - gt_mag) ** 2))

        # Render a frame with the current mask
        mask_np = obs["mask"].astype(np.float32)
        frames.append(render_frame(gt_mag, zf_mag, pr_mag, mask_np, step, int(obs["budget"][0] if np.ndim(obs["budget"]) else obs["budget"]), mse_now))

        # Action mask + predict next action
        action_masks = env.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = bool(terminated or truncated)
        step += 1

    # Save GIF
    out_gif = Path(args.out_gif)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    if imageio is not None:
        imageio.mimsave(out_gif, frames, fps=max(1, args.fps))
        print(f"Saved GIF -> {out_gif}")
    else:
        print("imageio not available; skipping GIF. Install with `pip install imageio`.Saving grid instead…")

    # Also save a grid (e.g., 12 frames sampled uniformly)
    K = min(12, len(frames))
    idxs = np.linspace(0, len(frames) - 1, K).astype(int)
    grid = make_grid([frames[i] for i in idxs], cols=4)
    out_grid = Path(args.out_grid)
    out_grid.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_grid, grid)
    print(f"Saved grid -> {out_grid}")


if __name__ == "__main__":
    main()
