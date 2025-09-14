import os, h5py, glob
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils_fft import ifft2_centered as ifft2

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