# MRI Reconstruction with Reinforcement Learning

Active k-space column selection for accelerated MRI using a **MaskablePPO** agent.
The RL policy learns *which* frequency columns to measure next so that a fixed U-Net
reconstructor can recover the highest-quality image within a given acquisition budget.

---

## How it works

```
  k-space (full)
       │
       ▼
  ┌─────────────┐   reset / step    ┌──────────────────────┐
  │  KspaceEnv  │ ◄──────────────── │  MaskablePPO policy  │
  │             │                   │  (action = column id) │
  │  obs:       │ ──────────────►   └──────────────────────┘
  │   zf image  │   reward = ΔSSIM
  │   mask      │
  │   budget    │
  └──────┬──────┘
         │ undersampled k-space
         ▼
  ┌─────────────┐
  │  U-Net      │  ──►  reconstructed image
  │  (frozen)   │
  └─────────────┘
```

**Pipeline overview:**

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `train_recon_dicom.py` | Train U-Net reconstructor with SSIM + L1 loss |
| 2 | `train_rl.py` | Train MaskablePPO agent with frozen reconstructor |
| 3 | `visualize_rl.py` | Animate one RL episode on a validation slice |
| 4 | `visualize_compare.py` | Side-by-side: RL vs baseline sampling strategy |

---

## Repository structure

```
mri-rl/
├── rl/
│   ├── env_ssim.py          # KspaceEnv — Gymnasium environment
│   ├── fastmri_loader.py    # DicomSingleCoilDataset
│   ├── reconstructor.py     # UNetLarge + ReconWrapper
│   ├── metrics.py           # NMSE, PSNR
│   └── utils_fft.py         # fft2_centered / ifft2_centered
│
├── data/
│   └── splits/
│       ├── train.txt        # one DICOM slice path per line
│       ├── val.txt
│       └── test.txt
│
├── checkpoints/             # saved .pth and .zip checkpoints
├── outputs/                 # generated GIFs, PNGs, CSVs
│
├── train_recon_ssim.py      # Step 1 – train reconstructor
├── train_rl.py              # Step 2 – train RL agent
├── visualize_rl.py          # Step 3 – visualize RL episode
├── visualize_compare.py     # Step 4 – RL vs baseline comparison
├── make_scan_rl_dicom_dir.py # Inference on a full DICOM directory
├── cli_args.py              # Centralised CLI argument definitions
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
# imageio is required for GIF export (not in requirements.txt):
pip install imageio
```

> Tested with Python 3.10, PyTorch 2.1, CUDA 12.1.

---

## Data preparation

Create plain-text list files where each line is the absolute path to one DICOM file.
The loader scans each file for its slices automatically.

```bash
# For DICOM datasets – adjust paths to your data roots
python generate_splits_dicom.py \
  --train-root /data/knee_train \
  --val-root   /data/knee_val \
  --out-dir    data/splits
```

This writes `data/splits/train.txt` and `data/splits/val.txt`.

---

## Step 1 — Train the reconstructor

The U-Net is trained first on random undersampling masks, independently of the RL agent.

```bash
python train_recon_dicom.py \
  --train-list data/splits/train.txt \
  --val-list   data/splits/val.txt \
  --accel      6 \
  --acs        16 \
  --epochs     50 \
  --batch-size 16 \
  --loss       ssim_l1 \
  --alpha      0.85 \
  --save       checkpoints/recon_largeunet_ssiml1_sag_6R.pth
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--accel` | `6` | Acceleration factor (e.g. 6× → ~1/6 of k-space measured) |
| `--acs` | `16` | Auto-Calibration Signal lines always measured in the centre |
| `--loss` | `ssim_l1` | Loss: `ssim` or `ssim_l1` (SSIM + weighted L1) |
| `--alpha` | `0.85` | Weight of SSIM in the combined loss |
| `--amp` | off | Enable mixed precision training (`--amp`) |
| `--cache-items` | `4096` | Maximum slices cached in RAM per worker |

---

## Step 2 — Train the RL agent

The RL agent receives the current zero-filled reconstruction, the acquisition mask
and the remaining budget as observations.
The reward at each step is **ΔSSIM** — the improvement in SSIM caused by measuring
the chosen k-space column.

```bash
python train_rl.py \
  --train-list data/splits/train.txt \
  --val-list   data/splits/val.txt \
  --recon      checkpoints/recon_largeunet_ssiml1_sag_8R.pth \
  --budget     40 \
  --acs        16 \
  --timesteps  500000
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--recon` | — | Path to the pre-trained reconstructor (frozen during RL) |
| `--budget` | `40` | Total k-space columns the agent may acquire per episode |
| `--acs` | `16` | ACS lines pre-measured before the agent starts |
| `--timesteps` | `100 000` | Total environment steps for training |
| `--start-with-acs` | `1` | Pre-fill ACS region at episode start (1 = yes) |

The trained policy is saved as `logs/ppo_maskable/ppo_maskable.zip` by default
(Stable-Baselines3 default path).

---

## Step 3 — Visualize an RL episode

Renders a GIF showing the agent progressively acquiring k-space columns and how
reconstruction quality improves at every step.

```bash
python visualize_rl.py \
  --val-list data/splits/val.txt \
  --index    202 \
  --rl       checkpoints/ppo_maskable_sag_6R.zip \
  --recon    checkpoints/recon_largeunet_ssiml1_sag_6R.pth \
  --budget   53 \
  --out-gif  outputs/rl_episode.gif \
  --out-grid outputs/rl_episode_grid.png \
  --fps      3
```

Each frame contains four panels:

```
┌──────────┬────────────┬─────────────┬─────────────────────┐
│  Ground  │ Zero-filled│    Recon    │  k-space mask       │
│  Truth   │  image     │  (U-Net+DC) │  step N | budget M  │
└──────────┴────────────┴─────────────┴─────────────────────┘
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--index` | `202` | Slice index in the validation list |
| `--use-pseudo-ref` | `1` | Use pseudo-reference SSIM reward (no GT needed) |
| `--pseudo-ref-acs` | `16` | ACS size for building the pseudo-reference |
| `--fps` | `2` | Frames per second in the output GIF |

---

## Step 4 — Side-by-side comparison: RL vs baseline

Runs the RL agent and a simple baseline strategy on the **same slice** in parallel,
renders synchronized frames, and saves a GIF + static grid.

```bash
python visualize_compare.py \
  --val-list data/splits/val.txt \
  --index    202 \
  --rl       checkpoints/ppo_maskable_sag_6R.zip \
  --recon    checkpoints/recon_largeunet_ssiml1_sag_6R.pth \
  --budget   53 \
  --baseline uniform \
  --out-gif  outputs/compare_rl_vs_uniform.gif \
  --out-grid outputs/compare_rl_vs_uniform_grid.png \
  --fps      3
```

Each frame contains:

```
┌──────────┬─────────────┬──────────┬─────────────┬──────────┐
│  Ground  │  RL Recon   │ RL mask  │ Baseline    │ Baseline │
│  Truth   │  SSIM=0.93  │ (blue)   │ Recon       │ mask     │
│          │             │          │ SSIM=0.88   │ (orange) │
└──────────┴─────────────┴──────────┴─────────────┴──────────┘
│              SSIM progress curve (RL — vs — baseline)      │
└────────────────────────────────────────────────────────────┘
```

**`--baseline` options:**

| Value | Description |
|-------|-------------|
| `uniform` | Equidistant columns across the full k-space |
| `random` | Randomly shuffled columns (deterministic, seed 42) |
| `lowfreq` | Centre-out: lowest spatial frequencies first |

---

## Inference on a full DICOM scan

Reconstruct every slice in a DICOM directory and save PNG images:

```bash
python make_scan_rl_dicom_dir.py \
  --dicom-dir  /path/to/DICOM/series \
  --rl         checkpoints/ppo_maskable_sag_6R.zip \
  --recon      checkpoints/recon_largeunet_ssiml1_sag_6R.pth \
  --budget     53 \
  --use-pseudo-ref 1 \
  --out-dir    outputs/scan_rl
```

When `--use-pseudo-ref 1` is set, the SSIM reward is computed against a
**pseudo-reference** (reconstruction from the wide ACS region only) instead of
ground truth — enabling inference on real clinical data without a reference scan.


---

## Key concepts

**ACS (Auto-Calibration Signal)** — central k-space lines that are always acquired
at the start of each episode. They provide a minimal low-resolution image and are
used to build coil sensitivity maps or pseudo-references.

**Data consistency (DC)** — after U-Net reconstruction, measured k-space lines are
restored from the original signal: `k_dc = where(mask, k_measured, k_predicted)`.
This enforces physical consistency without additional training.

**Pseudo-reference** — a reconstruction obtained from a wider ACS region
(default 48 lines) computed once per episode. Used as a GT substitute during
inference when real ground truth is unavailable.

**Action masking** — `MaskablePPO` receives a binary mask of already-acquired
columns at every step, ensuring the agent never selects a column twice.

**ΔSSIM reward** — `r_t = SSIM_t − SSIM_{t−1}`.  A small penalty (−0.01) is
applied if the agent attempts to measure an already-acquired column.

---

## Dependencies

| Package | Version | Role |
|---------|---------|------|
| PyTorch | ≥ 2.1 | Deep learning backend |
| Gymnasium | 0.29 | RL environment API |
| stable-baselines3 | 2.3.0 | PPO training + callbacks |
| sb3-contrib | 2.3.0 | MaskablePPO policy |
| h5py | 3.10.x | HDF5 data loading |
| matplotlib | — | Visualisation |
| Pillow | — | Image I/O |
| scipy | — | Signal processing |
| imageio | — | GIF export (`pip install imageio`) |

