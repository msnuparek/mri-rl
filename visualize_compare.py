"""Side-by-side comparison: RL policy vs baseline k-space sampling.

Runs both the RL agent and a simple baseline (uniform / random / low-freq) on
the same slice, records the reconstruction and SSIM at every acquisition step,
and renders a synchronized GIF + static grid PNG.

Each frame contains:
  - Ground Truth
  - RL Recon  |  RL k-space mask  (blue tones)
  - Baseline Recon  |  Baseline k-space mask  (orange tones)
  - SSIM progress curve for both methods (updated up to the current step)

Usage:
  python visualize_compare.py \\
    --val-list data/splits/val.txt \\
    --index 0 \\
    --rl checkpoints/ppo_maskable_sag_8R.zip \\
    --recon checkpoints/recon_largeunet_ssiml1_sag_8R.pth \\
    --budget 40 \\
    --baseline uniform \\
    --out-gif outputs/compare_rl_vs_uniform.gif \\
    --out-grid outputs/compare_rl_vs_uniform_grid.png
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

try:
    import imageio
except Exception:  # pragma: no cover
    imageio = None

from rl.fastmri_loader import DicomSingleCoilDataset
from rl.reconstructor import build_reconstructor, ReconWrapper
from rl.env_ssim import KspaceEnv


# ---------------------------------------------------------------------------
# Data container for one episode step
# ---------------------------------------------------------------------------

@dataclass
class StepState:
    """Snapshot of reconstruction quality at one acquisition step."""
    recon_mag: np.ndarray   # [H, W] magnitude image
    mask_1d:   np.ndarray   # [W]    k-space column mask (float 0/1)
    ssim_gt:   float        # SSIM vs ground truth
    budget_left: int        # remaining acquisition budget


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_reconstructor(path: str, device: torch.device) -> ReconWrapper:
    core  = build_reconstructor("unet_large", base=64, use_se=True, p_drop=0.0).to(device)
    model = ReconWrapper(core).to(device)
    if path:
        ckpt  = torch.load(path, map_location=device)
        state = ckpt.get("state_dict", ckpt)
        if list(state.keys()) and all(k.startswith("model.") for k in state.keys()):
            model.load_state_dict(state, strict=False)
        else:
            core.load_state_dict(state, strict=False)
        print(f"[info] Loaded recon checkpoint: {path}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_policy(path: str):
    from sb3_contrib.ppo_mask import MaskablePPO
    model = MaskablePPO.load(path, device="auto")
    print(f"[info] Loaded RL policy: {path}")
    return model


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _disp(img: np.ndarray) -> np.ndarray:
    """Clip to 99.5-th percentile and normalise to [0, 1]."""
    v = np.percentile(img, 99.5) if np.isfinite(img).all() else 1.0
    return np.clip(img / (v + 1e-8), 0, 1)


def _magnitude(x: torch.Tensor) -> torch.Tensor:
    """[B,2,H,W] -> [B,1,H,W] magnitude."""
    return torch.sqrt(x[:, :1] ** 2 + x[:, 1:2] ** 2 + 1e-12)


# ---------------------------------------------------------------------------
# Baseline column selection
# ---------------------------------------------------------------------------

def baseline_columns(
    W: int,
    already_measured: set,
    budget: int,
    strategy: str,
) -> List[int]:
    """Return ordered list of k-space columns for a simple baseline strategy.

    Args:
        W:                Total number of k-space columns.
        already_measured: Columns already in the mask (e.g. ACS region).
        budget:           Number of additional columns to select.
        strategy:         One of 'uniform', 'random', 'lowfreq'.

    Returns:
        Ordered list of column indices (length == min(budget, free columns)).
    """
    free = [c for c in range(W) if c not in already_measured]

    if strategy == "uniform":
        if budget >= len(free):
            return free
        idxs = np.round(np.linspace(0, len(free) - 1, budget)).astype(int)
        return [free[i] for i in idxs]

    elif strategy == "random":
        rng = np.random.default_rng(42)
        arr = np.array(free, dtype=int)
        rng.shuffle(arr)
        return arr[:budget].tolist()

    elif strategy == "lowfreq":
        # pick closest-to-centre columns first (low spatial frequencies)
        center = W // 2
        free.sort(key=lambda c: abs(c - center))
        return free[:budget]

    else:
        raise ValueError(f"Unknown baseline strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

def _collect_step_state(env: KspaceEnv, obs: dict) -> StepState:
    """Read reconstruction and GT-SSIM from the current env state."""
    rec_mag = env._reconstruct_current().squeeze().cpu().numpy()  # [H, W]
    ssim_gt = env._current_ssim_gt()
    budget_left = int(obs["budget"][0] if np.ndim(obs["budget"]) else obs["budget"])
    mask_np = obs["mask"].astype(np.float32)
    return StepState(recon_mag=rec_mag, mask_1d=mask_np, ssim_gt=ssim_gt, budget_left=budget_left)


def run_episode_rl(
    env: KspaceEnv,
    model,
    slice_idx: int,
) -> List[StepState]:
    """Run a full RL episode and collect a StepState at every step."""
    env._i = slice_idx
    obs, _ = env.reset()
    states: List[StepState] = []
    done = False

    while not done:
        states.append(_collect_step_state(env, obs))
        action_masks = env.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = bool(terminated or truncated)

    return states


def run_episode_baseline(
    env: KspaceEnv,
    cols_ordered: List[int],
    slice_idx: int,
) -> List[StepState]:
    """Run a baseline episode with a pre-determined column order."""
    env._i = slice_idx
    obs, _ = env.reset()
    states: List[StepState] = []
    col_iter = iter(cols_ordered)
    done = False

    while not done:
        states.append(_collect_step_state(env, obs))

        # Find the next unmeasured column from our list
        action = None
        while action is None:
            try:
                c = next(col_iter)
                if not env.mask[c]:      # skip if somehow already measured
                    action = c
            except StopIteration:
                done = True
                break

        if done:
            break

        obs, _, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

    return states


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def render_compare_frame(
    gt_mag:       np.ndarray,
    rl:           StepState,
    bl:           StepState,
    ssim_hist_rl: List[float],
    ssim_hist_bl: List[float],
    step:         int,
    total_steps:  int,
    baseline_name: str,
) -> np.ndarray:
    """Render one comparison frame (RL vs baseline)."""
    fig = plt.figure(figsize=(24, 9))
    gs  = gridspec.GridSpec(
        2, 5,
        height_ratios=[3, 1.8],
        hspace=0.50,
        wspace=0.30,
    )

    ax_gt      = fig.add_subplot(gs[0, 0])
    ax_rl_rec  = fig.add_subplot(gs[0, 1])
    ax_rl_mask = fig.add_subplot(gs[0, 2])
    ax_bl_rec  = fig.add_subplot(gs[0, 3])
    ax_bl_mask = fig.add_subplot(gs[0, 4])
    ax_curve   = fig.add_subplot(gs[1, :])

    # --- image panels -------------------------------------------------------
    ax_gt.imshow(_disp(gt_mag), cmap="gray")
    ax_gt.set_title("Ground Truth", fontsize=10)
    ax_gt.axis("off")

    rl_color = "#1f77b4"   # matplotlib blue
    bl_color = "#ff7f0e"   # matplotlib orange

    ax_rl_rec.imshow(_disp(rl.recon_mag), cmap="gray")
    ax_rl_rec.set_title(f"RL  SSIM={rl.ssim_gt:.4f}", fontsize=10, color=rl_color)
    ax_rl_rec.axis("off")

    ax_rl_mask.imshow(rl.mask_1d[np.newaxis, :], cmap="Blues",
                      aspect="auto", vmin=0, vmax=1)
    ax_rl_mask.set_yticks([])
    ax_rl_mask.set_title(f"RL mask  (budget left: {rl.budget_left})",
                          fontsize=9, color=rl_color)
    ax_rl_mask.set_xlabel("k-space columns", fontsize=8)

    ax_bl_rec.imshow(_disp(bl.recon_mag), cmap="gray")
    ax_bl_rec.set_title(f"{baseline_name}  SSIM={bl.ssim_gt:.4f}",
                        fontsize=10, color=bl_color)
    ax_bl_rec.axis("off")

    ax_bl_mask.imshow(bl.mask_1d[np.newaxis, :], cmap="Oranges",
                      aspect="auto", vmin=0, vmax=1)
    ax_bl_mask.set_yticks([])
    ax_bl_mask.set_title(f"{baseline_name} mask  (budget left: {bl.budget_left})",
                          fontsize=9, color=bl_color)
    ax_bl_mask.set_xlabel("k-space columns", fontsize=8)

    # --- SSIM progress curve ------------------------------------------------
    steps_x_rl = list(range(len(ssim_hist_rl)))
    steps_x_bl = list(range(len(ssim_hist_bl)))
    ax_curve.plot(steps_x_rl, ssim_hist_rl,
                  color=rl_color, linewidth=2.0, label="RL")
    ax_curve.plot(steps_x_bl, ssim_hist_bl,
                  color=bl_color, linewidth=2.0, linestyle="--",
                  label=baseline_name)
    ax_curve.axvline(step, color="gray", linestyle=":", linewidth=1.5, alpha=0.8)

    # mark current SSIM values
    if ssim_hist_rl:
        ax_curve.scatter([step], [ssim_hist_rl[-1]], color=rl_color, s=40, zorder=5)
    if ssim_hist_bl:
        ax_curve.scatter([step], [ssim_hist_bl[-1]], color=bl_color, s=40, zorder=5)

    ax_curve.set_xlim(-0.5, total_steps + 0.5)
    ax_curve.set_ylim(0, 1)
    ax_curve.set_xlabel("Acquisition step", fontsize=9)
    ax_curve.set_ylabel("SSIM vs GT", fontsize=9)
    ax_curve.set_title(
        f"SSIM progress  —  step {step + 1} / {total_steps}", fontsize=9
    )
    ax_curve.legend(fontsize=9, loc="lower right")
    ax_curve.grid(True, alpha=0.35)

    fig.suptitle(
        f"RL vs {baseline_name} k-space sampling",
        fontsize=12, y=1.01,
    )

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


# ---------------------------------------------------------------------------
# Grid helper
# ---------------------------------------------------------------------------

def make_grid(frames: List[np.ndarray], cols: int = 3) -> np.ndarray:
    rows = int(np.ceil(len(frames) / cols))
    h = min(f.shape[0] for f in frames)
    w = min(f.shape[1] for f in frames)
    fr   = [f[:h, :w] for f in frames]
    grid = np.ones((rows * h, cols * w, 3), dtype=np.uint8) * 255
    for i, f in enumerate(fr):
        r, c = divmod(i, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = f
    return grid


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    from cli_args import parse_visualize_compare_args
    return parse_visualize_compare_args()


def main():
    os.environ.setdefault("OMP_NUM_THREADS",    "1")
    os.environ.setdefault("MKL_NUM_THREADS",    "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    args   = parse_args()
    device = torch.device(args.device)

    use_pseudo_ref = bool(getattr(args, "use_pseudo_ref", 0))
    pseudo_ref_acs = int(getattr(args, "pseudo_ref_acs", 48))

    baseline_label = {
        "uniform":  "Uniform",
        "random":   "Random",
        "lowfreq":  "Low-freq",
    }.get(args.baseline, args.baseline.capitalize())

    # ------------------------------------------------------------------
    # Dataset + model
    # ------------------------------------------------------------------
    ds    = DicomSingleCoilDataset(args.val_list)
    recon = load_reconstructor(args.recon, device)

    env_kwargs = dict(
        budget         = args.budget,
        device         = args.device,
        acs            = int(args.acs),
        start_with_acs = bool(args.start_with_acs),
        use_pseudo_ref = use_pseudo_ref,
        pseudo_ref_acs = pseudo_ref_acs,
    )
    env_rl = KspaceEnv(ds, recon, **env_kwargs)
    env_bl = KspaceEnv(ds, recon, **env_kwargs)

    slice_idx = int(args.index) % len(ds)

    # Probe slice to get W and initial mask (ACS region)
    env_rl._i = slice_idx
    obs_probe, _ = env_rl.reset()
    W                = obs_probe["mask"].shape[0]
    initial_mask_set = set(int(c) for c in np.where(obs_probe["mask"].astype(bool))[0])
    remaining_budget = int(
        obs_probe["budget"][0] if np.ndim(obs_probe["budget"]) else obs_probe["budget"]
    )

    # Ground truth magnitude (from the probed env)
    gt_tensor = env_rl.target_img.permute(2, 0, 1).unsqueeze(0)  # [1,2,H,W]
    gt_mag    = _magnitude(gt_tensor).squeeze().cpu().numpy()

    # Pre-compute baseline columns
    cols = baseline_columns(W, initial_mask_set, remaining_budget, args.baseline)
    print(f"[info] Baseline '{baseline_label}': {len(cols)} columns pre-selected")

    model = load_policy(args.rl)

    # ------------------------------------------------------------------
    # Run both episodes
    # ------------------------------------------------------------------
    print("[info] Running RL episode…")
    rl_states = run_episode_rl(env_rl, model, slice_idx)
    print(f"       {len(rl_states)} steps  |  final SSIM={rl_states[-1].ssim_gt:.4f}")

    print(f"[info] Running {baseline_label} baseline episode…")
    bl_states = run_episode_baseline(env_bl, cols, slice_idx)
    print(f"       {len(bl_states)} steps  |  final SSIM={bl_states[-1].ssim_gt:.4f}")

    # SSIM improvement summary
    delta = rl_states[-1].ssim_gt - bl_states[-1].ssim_gt
    sign  = "+" if delta >= 0 else ""
    print(f"[info] RL vs {baseline_label}:  ΔSSIM = {sign}{delta:.4f}")

    # Align lengths (both should be equal, but guard just in case)
    n_steps   = min(len(rl_states), len(bl_states))
    rl_states = rl_states[:n_steps]
    bl_states = bl_states[:n_steps]

    ssim_hist_rl = [s.ssim_gt for s in rl_states]
    ssim_hist_bl = [s.ssim_gt for s in bl_states]

    # ------------------------------------------------------------------
    # Render frames
    # ------------------------------------------------------------------
    print(f"[info] Rendering {n_steps} frames…")
    frames: List[np.ndarray] = []
    for i in range(n_steps):
        frame = render_compare_frame(
            gt_mag        = gt_mag,
            rl            = rl_states[i],
            bl            = bl_states[i],
            ssim_hist_rl  = ssim_hist_rl[:i + 1],
            ssim_hist_bl  = ssim_hist_bl[:i + 1],
            step          = i,
            total_steps   = n_steps,
            baseline_name = baseline_label,
        )
        frames.append(frame)
    print(f"       Done — {len(frames)} frames rendered")

    # ------------------------------------------------------------------
    # Save GIF
    # ------------------------------------------------------------------
    out_gif = Path(args.out_gif)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    if imageio is not None:
        imageio.mimsave(out_gif, frames, fps=max(1, args.fps))
        print(f"[ok] Saved GIF -> {out_gif}")
    else:
        print("[warn] imageio not available; skipping GIF. Install with: pip install imageio")

    # ------------------------------------------------------------------
    # Save static grid (up to 9 evenly-sampled frames, 3 per row)
    # ------------------------------------------------------------------
    K    = min(9, len(frames))
    idxs = np.linspace(0, len(frames) - 1, K).astype(int)
    grid = make_grid([frames[i] for i in idxs], cols=3)

    out_grid = Path(args.out_grid)
    out_grid.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_grid, grid)
    print(f"[ok] Saved grid -> {out_grid}")


if __name__ == "__main__":
    main()
