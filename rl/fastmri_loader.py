from __future__ import annotations

import os, h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import pydicom
from pydicom.errors import InvalidDicomError
try:
    from pydicom.pixel_data_handlers.util import apply_modality_lut
except ImportError:
    apply_modality_lut = None
from collections import OrderedDict
from .utils_fft import ifft2_centered as ifft2, fft2_centered as fft2

class SingleCoilKneeDataset(Dataset):
    """Iterates over slices from .h5 files listed in a txt file (absolute paths)."""
    def __init__(self, list_path: str, center_crop: int | None = 320):
        super().__init__()
        with open(list_path, "r", encoding="utf-8") as f:
            self.files = [ln.strip() for ln in f if ln.strip()]
        self.index = []  # list of (file_path, slice_idx)
        for fp in self.files:
            with h5py.File(fp, "r") as hf:
                n = hf["kspace"].shape[0]
            self.index.extend([(fp, i) for i in range(n)])
        self.center_crop = center_crop

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _to_ri(xc: np.ndarray) -> np.ndarray:
        return np.stack([xc.real, xc.imag], axis=-1).astype(np.float32)

    @staticmethod
    def _complex_from_ri(xri: np.ndarray) -> np.ndarray:
        return xri[..., 0] + 1j * xri[..., 1]

    @staticmethod
    def _center_crop2d(arr: np.ndarray, size: int) -> np.ndarray:
        H, W = arr.shape[:2]
        h0 = (H - size) // 2 if H > size else 0
        w0 = (W - size) // 2 if W > size else 0
        return arr[h0:h0+min(size,H), w0:w0+min(size,W), ...]

    def __getitem__(self, idx):
        fp, si = self.index[idx]
        with h5py.File(fp, "r") as hf:
            k = hf["kspace"][si]  # complex64 [H,W]
        k = np.asarray(k)
        if self.center_crop is not None:
            k = self._center_crop2d(k, self.center_crop)
        k_ri = self._to_ri(k)
        # GT image via IFFT2 of full k-space
        k_t = torch.from_numpy(k_ri).unsqueeze(0)  # [1,H,W,2]
        img_ri = ifft2(k_t).squeeze(0).numpy()    # [H,W,2]
        # normalize to ~[0,1]
        mag = np.sqrt(img_ri[...,0]**2 + img_ri[...,1]**2)
        s = np.percentile(mag, 99) + 1e-6
        img_ri = img_ri / s
        k_ri = k_ri / s  # keep consistent scaling
        return {
            "kspace": torch.from_numpy(k_ri).permute(2,0,1).float(),  # [2,H,W]
            "target": torch.from_numpy(img_ri).permute(2,0,1).float(), # [2,H,W]
        }

class DicomSingleCoilDataset(Dataset):
    """Iterates DICOM slices from directories or list files, matching SingleCoilKneeDataset outputs."""

    def __init__(self, source: str, center_crop: int | None = 320, recursive: bool = True,
                 return_kspace: bool = True, max_cache_items: int = 0):
        super().__init__()
        self.center_crop = center_crop
        self.recursive = recursive
        self.return_kspace = return_kspace
        self.cache_limit = max_cache_items if max_cache_items and max_cache_items > 0 else None
        self.cache = OrderedDict() if self.cache_limit else None

        src_path = os.path.abspath(source)
        roots = []
        if os.path.isdir(src_path):
            roots = [src_path]
        elif os.path.isfile(src_path):
            with open(src_path, "r", encoding="utf-8") as f:
                for raw in f:
                    entry = raw.strip()
                    if not entry or entry.startswith('#'):
                        continue
                    abs_entry = os.path.abspath(entry)
                    if not os.path.isdir(abs_entry):
                        raise ValueError(f"Listed path is not a directory: {abs_entry}")
                    roots.append(abs_entry)
        else:
            raise ValueError(f"Path is neither directory nor file: {src_path}")

        if not roots:
            raise ValueError(f"No directories found from input: {source}")

        self.root_dirs = roots
        self.samples = []
        skipped_missing_pixel = 0
        skipped_non_image = 0


        for root in self.root_dirs:
            if self.recursive:
                walker = os.walk(root)
            else:
                try:
                    entries = [entry.name for entry in os.scandir(root) if entry.is_file()]
                except FileNotFoundError:
                    continue
                walker = [(root, [], entries)]

            for current_root, _, files in walker:
                for name in files:
                    path_str = os.path.join(current_root, name)
                    try:
                        ds = pydicom.dcmread(path_str, force=True, defer_size="1 KB")
                    except (InvalidDicomError, FileNotFoundError, PermissionError, IsADirectoryError):
                        continue
                    if not self._is_image_dataset(ds):
                        if self._has_pixel_data(ds):
                            skipped_non_image += 1
                        else:
                            skipped_missing_pixel += 1
                        continue
                    series_uid = ds.get("SeriesInstanceUID") or ""
                    instance = ds.get("InstanceNumber")
                    index = int(instance) if instance is not None else 0
                    self.samples.append((series_uid, index, path_str))

        self.samples.sort(key=lambda x: (x[0], x[1], x[2]))

        self.skipped_no_pixel = skipped_missing_pixel
        self.skipped_non_image = skipped_non_image
        if skipped_missing_pixel:
            print(f"[DicomSingleCoilDataset] skipped {skipped_missing_pixel} files without pixel data")
        if skipped_non_image:
            print(f"[DicomSingleCoilDataset] skipped {skipped_non_image} non-image DICOM files")

        if not self.samples:
            raise ValueError(f"No DICOM slices found under: {', '.join(self.root_dirs)}")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _apply_modality_lut(arr: np.ndarray, ds: pydicom.dataset.Dataset) -> np.ndarray:
        if apply_modality_lut is not None:
            arr = apply_modality_lut(arr, ds)
        else:
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            arr = arr * slope + intercept
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _prepare_image(ds: pydicom.dataset.Dataset) -> np.ndarray:
        try:
            arr = ds.pixel_array.astype(np.float32, copy=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to decode pixel data for {ds.filename}") from exc
        arr = DicomSingleCoilDataset._apply_modality_lut(arr, ds)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {arr.shape}")
        if ds.get("PhotometricInterpretation", "").upper() == "MONOCHROME1":
            arr = arr.max() - arr
        return np.ascontiguousarray(arr, dtype=np.float32)

    @staticmethod
    def _has_pixel_data(ds: pydicom.dataset.Dataset) -> bool:
        return any(tag in ds for tag in (
            "PixelData",
            "FloatPixelData",
            "DoubleFloatPixelData",
        ))

    @staticmethod
    def _is_image_dataset(ds: pydicom.dataset.Dataset) -> bool:
        if not DicomSingleCoilDataset._has_pixel_data(ds):
            return False
        rows = ds.get("Rows")
        cols = ds.get("Columns")
        if rows in (None, 0) or cols in (None, 0):
            return False
        sop_uid = ds.get("SOPClassUID") or getattr(ds.file_meta, "MediaStorageSOPClassUID", None)
        if sop_uid is None:
            return False
        sop_uid = str(sop_uid)
        allowed_prefixes = (
            "1.2.840.10008.5.1.4.1.1.4",
            "1.2.840.10008.5.1.4.1.1.4.1",
            "1.2.840.10008.5.1.4.1.1.4.2",
            "1.2.840.10008.5.1.4.1.1.4.3",
            "1.2.840.10008.5.1.4.1.1.481",
        )
        if not any(sop_uid.startswith(pref) for pref in allowed_prefixes):
            return False
        for key in ("PixelData", "FloatPixelData", "DoubleFloatPixelData"):
            elem = ds.get(key)
            if elem is not None:
                vl = getattr(elem, "VL", None)
                if vl is not None and vl == 0:
                    return False
                return True
        return False

    @staticmethod
    def _normalize_size(arr: np.ndarray, size: int) -> np.ndarray:
        if size is None:
            return arr
        H, W = arr.shape[:2]
        pad_h = max(0, size - H)
        pad_w = max(0, size - W)
        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            if arr.ndim == 2:
                pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
            else:
                pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
            arr = np.pad(arr, pad_width, mode="constant")
            H, W = arr.shape[:2]
        if H > size or W > size:
            arr = SingleCoilKneeDataset._center_crop2d(arr, size)
        return arr

    def _cache_get(self, key: str):
        if self.cache is None:
            return None
        val = self.cache.get(key)
        if val is not None:
            self.cache.move_to_end(key)
        return val

    def _cache_set(self, key: str, value: dict):
        if self.cache is None:
            return
        self.cache[key] = value
        self.cache.move_to_end(key)
        if self.cache_limit is not None and len(self.cache) > self.cache_limit:
            self.cache.popitem(last=False)

    def __getitem__(self, idx: int):
        _, _, path = self.samples[idx]
        cached = self._cache_get(path)
        if cached is not None:
            return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in cached.items()}

        ds = pydicom.dcmread(path, force=True)
        img = self._prepare_image(ds)
        if self.center_crop is not None:
            img = self._normalize_size(img, self.center_crop)

        img_ri = np.stack([img, np.zeros_like(img)], axis=-1)
        k_ri = fft2(torch.from_numpy(img_ri)).detach().numpy()
        img_ri = ifft2(torch.from_numpy(k_ri)).detach().numpy()

        mag = np.sqrt(img_ri[..., 0] ** 2 + img_ri[..., 1] ** 2)
        scale = np.percentile(mag, 99) + 1e-6
        img_ri = img_ri / scale
        k_ri = k_ri / scale

        target = torch.tensor(np.moveaxis(img_ri, -1, 0), dtype=torch.float32).contiguous()
        result = {"target": target}

        if self.return_kspace:
            kspace = torch.tensor(np.moveaxis(k_ri, -1, 0), dtype=torch.float32).contiguous()
            result["kspace"] = kspace

        cache_entry = {k: v for k, v in result.items()}
        self._cache_set(path, cache_entry)

        return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in cache_entry.items()}




