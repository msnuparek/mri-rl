"""
Create a full MR reconstruction using a trained reconstructor and a trained
MaskablePPO policy that adaptively selects k‑space columns. Saves the final
reconstruction and the ground‑truth magnitude images to outputs.

Usage (example):
  python make_scan_rl.py \
    --val-list data/splits/val.txt \
    --index 20 \
    --rl checkpoints/ppo_maskable.zip \
    --recon checkpoints/recon_largeunet_ssiml1.pth \
    --budget 144 \
    --acs 16 \
    --out-dir outputs/scan_rl
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl.fastmri_loader import SingleCoilKneeDataset
from rl.reconstructor import build_reconstructor, ReconWrapper
from rl.env_ssim import KspaceEnv
from rl.utils_fft import ifft2_centered as ifft2, fft2_centered as fft2
from rl.metrics import nmse, psnr


def magnitude(x: torch.Tensor) -> torch.Tensor:
    """x: [B,2,H,W] -> [B,1,H,W] magnitude."""
    return torch.sqrt(x[:, :1] ** 2 + x[:, 1:2] ** 2 + 1e-12)


def load_reconstructor(path: str, device: torch.device) -> ReconWrapper:
    """Load reconstructor core into a no‑grad wrapper, handling both wrapper/core checkpoints."""
    core = build_reconstructor("unet_large", base=64, use_se=True, p_drop=0.0).to(device)
    model = ReconWrapper(core).to(device)
    if path:
        ckpt = torch.load(path, map_location=device)
        state = ckpt.get("state_dict", ckpt)
        keys = list(state.keys())
        if keys and all(k.startswith("model.") for k in keys):
            model.load_state_dict(state, strict=False)
        else:
            core.load_state_dict(state, strict=False)
        print(f"[info] loaded recon checkpoint: {path}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_policy(path: str):
    from sb3_contrib.ppo_mask import MaskablePPO
    model = MaskablePPO.load(path, device="auto")
    print(f"[info] loaded RL policy: {path}")
    return model


def save_mag_png(img: torch.Tensor, out_path: Path, pctl: float = 99.5) -> None:
    """Save a single‑channel magnitude tensor [1,1,H,W] as PNG with percentile clipping."""
    arr = img.squeeze().cpu().numpy()
    vmax = np.percentile(arr, pctl) if np.isfinite(arr).all() else 1.0
    plt.imsave(out_path, np.clip(arr / (vmax + 1e-8), 0, 1), cmap="gray")
    print(f"[ok] saved {out_path}")


def save_mag_dicom(
    img: torch.Tensor,
    out_path: Path,
    pctl: float = 99.5,
    patient_id: str = "RLRECON",
    series_desc: str = "RL Recon",
    instance_number: int = 1,
    study_uid: str | None = None,
    series_uid: str | None = None,
) -> None:
    """Save magnitude [1,1,H,W] tensor as a DICOM (Secondary Capture) file.

    Notes:
    - Scales image to uint16 using percentile clipping for stable brightness.
    - Uses SecondaryCaptureImageStorage to avoid requiring full MR acquisition tags.
    """
    try:
        import pydicom  # type: ignore
        from pydicom.dataset import Dataset, FileDataset  # type: ignore
        from pydicom.uid import generate_uid, ExplicitVRLittleEndian, SecondaryCaptureImageStorage  # type: ignore
    except Exception as e:  # pragma: no cover
        print(f"[warn] pydicom not available ({e}); skipping DICOM save for {out_path}")
        return

    arr = img.squeeze().cpu().numpy().astype(np.float32)
    vmax = np.percentile(arr, pctl) if np.isfinite(arr).all() else float(arr.max() if arr.size else 1.0)
    vmax = max(vmax, 1e-6)
    arr_n = np.clip(arr / vmax, 0.0, 1.0)
    px = np.round(arr_n * 65535.0).astype(np.uint16)

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(out_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    # Minimal patient/study/series
    ds.PatientName = patient_id
    ds.PatientID = patient_id
    ds.StudyInstanceUID = study_uid or generate_uid()
    ds.SeriesInstanceUID = series_uid or generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.Modality = "OT"  # secondary capture
    ds.SeriesDescription = series_desc
    ds.InstanceNumber = int(instance_number)

    # Image attributes
    ds.Rows, ds.Columns = int(px.shape[0]), int(px.shape[1])
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0  # unsigned
    # Optional display/window
    ds.WindowCenter = 32768
    ds.WindowWidth = 65535
    # Optional spacing (unknown -> 1.0 mm placeholders)
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 1.0

    ds.PixelData = px.tobytes()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(out_path, write_like_original=False)
    print(f"[ok] saved {out_path}")


def parse_args() -> argparse.Namespace:
    from cli_args import parse_make_scan_rl_args
    return parse_make_scan_rl_args()


def main():
    # Mild safety for Windows OpenMP mixing
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    args = parse_args()
    device = torch.device(args.device)

    # Data + components
    ds = SingleCoilKneeDataset(args.val_list)
    recon = load_reconstructor(args.recon, device)
    env = KspaceEnv(
        ds,
        recon,
        budget=int(args.budget),
        device=args.device,
        acs=int(args.acs),
        start_with_acs=bool(args.start_with_acs),
    )

    # Identify the source file for the selected dataset index
    ds_index = int(args.index) % len(ds)
    src_fp, _ = ds.index[ds_index]
    # Collect all dataset indices belonging to this file, in slice order
    file_indices = [j for j, (fp, _si) in enumerate(ds.index) if fp == src_fp]
    # Derive patient/series naming from file name
    fname = Path(src_fp).stem
    patient_id = fname[:64]  # DICOM PN length guard

    # Load RL policy once
    model = load_policy(args.rl)

    # Prepare output folders and DICOM UIDs (shared across slices)
    out_dir = Path(args.out_dir)
    series_dir_rl = out_dir / f"{fname}_RL_series"
    series_dir_gt = out_dir / f"{fname}_GT_series"
    series_dir_rl.mkdir(parents=True, exist_ok=True)
    series_dir_gt.mkdir(parents=True, exist_ok=True)

    try:
        # Optional: stable UIDs for a single study with two series (GT, RL)
        from pydicom.uid import generate_uid  # type: ignore
        study_uid = generate_uid()
        series_uid_gt = generate_uid()
        series_uid_rl = generate_uid()
    except Exception:
        # pydicom not installed -> save_png fallback still possible if enabled
        study_uid = None
        series_uid_gt = None
        series_uid_rl = None

    # Process all slices from the same source file
    for rank, j in enumerate(file_indices, start=1):
        # Reset environment to this particular dataset index
        env._i = j % len(ds)
        obs, _ = env.reset()

        # Roll out one full episode (greedy deterministic)
        done = False
        steps = 0
        while not done:
            action_masks = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = bool(terminated or truncated)
            steps += 1

        # Final reconstruction with DC (consistent with training)
        with torch.no_grad():
            zf = torch.from_numpy(obs["zf"]).unsqueeze(0).to(device)          # [1,2,H,W]
            pred = recon(zf)                                                   # [1,2,H,W]
            W = zf.shape[-1]
            mask4 = torch.from_numpy(obs["mask"].astype(np.float32)).to(device).view(1, 1, 1, W)
            k_full4 = env.kspace_full.permute(2, 0, 1).unsqueeze(0)            # [1,2,H,W]
            k_pred = fft2(pred.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)        # [1,2,H,W]
            k_dc   = torch.where(mask4.bool(), k_full4, k_pred)
            pred   = ifft2(k_dc.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)       # [1,2,H,W]

            gt  = env.target_img.permute(2, 0, 1).unsqueeze(0)                 # [1,2,H,W]
            gt_mag  = magnitude(gt)
            pr_mag  = magnitude(pred)

        # Metrics per-slice (optional info)
        nmse_val = nmse(pr_mag, gt_mag)
        psnr_val = psnr(pr_mag, gt_mag)
        print(f"[slice {rank:03d}] steps={steps}  NMSE={nmse_val:.6f}  PSNR={psnr_val:.2f} dB")

        # Save DICOMs forming two separate series (GT and RL)
        dicom_name = f"IM_{rank:04d}.dcm"
        save_mag_dicom(
            gt_mag,
            series_dir_gt / dicom_name,
            pctl=args.pctl,
            patient_id=patient_id,
            series_desc=f"GT magnitude {fname}",
            instance_number=rank,
            study_uid=study_uid,
            series_uid=series_uid_gt,
        )
        save_mag_dicom(
            pr_mag,
            series_dir_rl / dicom_name,
            pctl=args.pctl,
            patient_id=patient_id,
            series_desc=f"RL reconstruction {fname}",
            instance_number=rank,
            study_uid=study_uid,
            series_uid=series_uid_rl,
        )

        # Also store the final mask for reference (per slice)
        np.save(series_dir_rl / f"mask_{rank:04d}.npy", obs["mask"].astype(np.uint8))
    
    print(f"[ok] exported DICOM series -> {series_dir_rl}  and  {series_dir_gt}")


if __name__ == "__main__":
    main()
