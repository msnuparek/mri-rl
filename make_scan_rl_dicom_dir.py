"""Reconstruct a full DICOM series using MaskablePPO policy and reconstructor.

This variant is tailored for `DicomSingleCoilDataset`, grouping slices by
SeriesInstanceUID (or, if missing, by parent directory). It exports both the
RL reconstruction and the ground-truth magnitude as separate DICOM series.

Unlike ``make_scan_rl_dicom.py`` it reads slices directly from a user supplied
``--dicom-dir`` rather than relying on the validation list.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl.fastmri_loader import DicomSingleCoilDataset
from rl.reconstructor import build_reconstructor, ReconWrapper
from rl.env_ssim import KspaceEnv
from rl.utils_fft import ifft2_centered as ifft2, fft2_centered as fft2
from rl.metrics import nmse, psnr


_SAFE_SLUG = re.compile(r"[^A-Za-z0-9._-]+")


def _slugify(text: str, fallback: str) -> str:
    text = _SAFE_SLUG.sub("_", text).strip("._-")
    return text or fallback


def magnitude(x: torch.Tensor) -> torch.Tensor:
    """x: [B,2,H,W] -> [B,1,H,W] magnitude."""
    return torch.sqrt(x[:, :1] ** 2 + x[:, 1:2] ** 2 + 1e-12)


def load_reconstructor(path: str, device: torch.device) -> ReconWrapper:
    """Load reconstructor core into a no-grad wrapper (handles wrapper/core checkpoints)."""
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
    """Save a single-channel magnitude tensor [1,1,H,W] as PNG with percentile clipping."""
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
    """Save magnitude [1,1,H,W] tensor as a DICOM (Secondary Capture) file."""
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

    ds = FileDataset(str(out_path), {}, file_meta=file_meta, preamble=bytes(128))
    ds.PatientName = patient_id
    ds.PatientID = patient_id
    ds.StudyInstanceUID = study_uid or generate_uid()
    ds.SeriesInstanceUID = series_uid or generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.Modality = "OT"
    ds.SeriesDescription = series_desc
    ds.InstanceNumber = int(instance_number)

    ds.Rows, ds.Columns = int(px.shape[0]), int(px.shape[1])
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = px.tobytes()

    ds.save_as(out_path, write_like_original=False)
    print(f"[ok] saved {out_path}")


def _select_series(ds: DicomSingleCoilDataset, index: int):
    series_uid, instance, path = ds.samples[index]
    path_obj = Path(path)
    if series_uid:
        indices = [i for i, (uid, _inst, _path) in enumerate(ds.samples) if uid == series_uid]
    else:
        parent = path_obj.parent
        indices = [i for i, (_uid, _inst, p) in enumerate(ds.samples) if Path(p).parent == parent]
    slice_infos = [(i, ds.samples[i][1], ds.samples[i][2]) for i in indices]
    slice_infos.sort(key=lambda x: (x[1], x[2]))
    ordered_indices = [i for i, _inst, _path in slice_infos]
    return ordered_indices, path_obj, series_uid, instance


def parse_args():
    from cli_args import parse_make_scan_rl_dir_args
    return parse_make_scan_rl_dir_args()


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    args = parse_args()
    device = torch.device(args.device)

    ds = DicomSingleCoilDataset(args.dicom_dir)
    recon = load_reconstructor(args.recon, device)
    use_pseudo_ref = bool(getattr(args, 'use_pseudo_ref', 0))
    pseudo_ref_acs = int(getattr(args, 'pseudo_ref_acs', 48))
    env = KspaceEnv(
        ds,
        recon,
        budget=int(args.budget),
        device=args.device,
        acs=int(args.acs),
        start_with_acs=bool(args.start_with_acs),
        use_pseudo_ref=use_pseudo_ref,
        pseudo_ref_acs=pseudo_ref_acs,
    )
    if use_pseudo_ref:
        print(f"[info] Using pseudo-reference with ACS={pseudo_ref_acs} for reward computation")

    ds_index = int(args.index) % len(ds)
    series_indices, ref_path, series_uid, first_instance = _select_series(ds, ds_index)

    try:
        import pydicom  # type: ignore
        ref_ds = pydicom.dcmread(str(ref_path), stop_before_pixels=True, force=True)
        patient_id = str(getattr(ref_ds, "PatientID", "RLRECON")) or "RLRECON"
        study_uid_src = str(getattr(ref_ds, "StudyInstanceUID", "")) or None
        series_desc_src = str(getattr(ref_ds, "SeriesDescription", "")) or ref_path.parent.name
    except Exception:
        ref_ds = None
        patient_id = _slugify(ref_path.parent.name, "RLRECON")[:64]
        study_uid_src = None
        series_desc_src = ref_path.parent.name

    fname_base = _slugify(series_desc_src or ref_path.parent.name, "dicom_series")
    if series_uid:
        if len(series_uid) > 8:
            suffix = series_uid[-8:]
        else:
            suffix = series_uid
        fname = _slugify(f"{fname_base}_{suffix}", fname_base)
    else:
        fname = fname_base

    model = load_policy(args.rl)

    out_dir = Path(args.out_dir)
    series_dir_rl = out_dir / f"{fname}_RL_series"
    series_dir_gt = out_dir / f"{fname}_GT_series"
    series_dir_rl.mkdir(parents=True, exist_ok=True)
    series_dir_gt.mkdir(parents=True, exist_ok=True)

    try:
        from pydicom.uid import generate_uid  # type: ignore
        study_uid = study_uid_src or generate_uid()
        series_uid_gt = generate_uid()
        series_uid_rl = generate_uid()
    except Exception:
        study_uid = None
        series_uid_gt = None
        series_uid_rl = None

    for rank, slice_idx in enumerate(series_indices, start=1):
        env._i = slice_idx % len(ds)
        obs, _ = env.reset()

        done = False
        steps = 0
        while not done:
            action_masks = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = bool(terminated or truncated)
            steps += 1

        with torch.no_grad():
            zf = torch.from_numpy(obs["zf"]).unsqueeze(0).to(device)
            pred = recon(zf)
            W = zf.shape[-1]
            mask4 = torch.from_numpy(obs["mask"].astype(np.float32)).to(device).view(1, 1, 1, W)
            k_full4 = env.kspace_full.permute(2, 0, 1).unsqueeze(0)
            k_pred = fft2(pred.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            k_dc = torch.where(mask4.bool(), k_full4, k_pred)
            pred = ifft2(k_dc.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            gt = env.target_img.permute(2, 0, 1).unsqueeze(0)
            gt_mag = magnitude(gt)
            pr_mag = magnitude(pred)

        nmse_val = nmse(pr_mag, gt_mag)
        psnr_val = psnr(pr_mag, gt_mag)
        # Zobraz SSIM hodnoty pokud jsou k dispozici
        ssim_str = ""
        if "ssim_gt" in info:
            ssim_str = f"  SSIM_GT={info['ssim_gt']:.4f}  SSIM_pseudo={info['ssim_pseudo']:.4f}"
        elif "ssim" in info:
            ssim_str = f"  SSIM={info['ssim']:.4f}"
        print(f"[slice {rank:03d}] steps={steps}  NMSE={nmse_val:.6f}  PSNR={psnr_val:.2f} dB{ssim_str}")

        dicom_name = f"IM_{rank:04d}.dcm"
        instance_number = ds.samples[slice_idx][1] or first_instance or rank

        save_mag_dicom(
            gt_mag,
            series_dir_gt / dicom_name,
            pctl=args.pctl,
            patient_id=patient_id[:64],
            series_desc=f"GT magnitude {fname}",
            instance_number=instance_number,
            study_uid=study_uid,
            series_uid=series_uid_gt,
        )
        save_mag_dicom(
            pr_mag,
            series_dir_rl / dicom_name,
            pctl=args.pctl,
            patient_id=patient_id[:64],
            series_desc=f"RL reconstruction {fname}",
            instance_number=instance_number,
            study_uid=study_uid,
            series_uid=series_uid_rl,
        )

        np.save(series_dir_rl / f"mask_{rank:04d}.npy", obs["mask"].astype(np.uint8))

    print(f"[ok] exported DICOM series -> {series_dir_rl}  and  {series_dir_gt}")


if __name__ == "__main__":
    main()
