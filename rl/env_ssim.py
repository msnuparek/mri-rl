import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional

from .utils_fft import ifft2_centered as ifft2, fft2_centered as fft2


# ---------------------------
# Helpers: magnitude + SSIM
# ---------------------------
def _to_mag(x: torch.Tensor) -> torch.Tensor:
    # x: [H,W,2] -> [1,1,H,W] magnitude
    m = torch.sqrt(x[..., 0:1] ** 2 + x[..., 1:2] ** 2 + 1e-12)  # [H,W,1]
    return m.permute(2, 0, 1).unsqueeze(0)  # [1,1,H,W]

def _normalize01_by_ref(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    # x, ref: [1,1,H,W]  -> normalize by max(ref)
    maxv = ref.amax(dim=(-2, -1), keepdim=True)
    return torch.clamp(x / (maxv + 1e-8), 0.0, 1.0)

def _gaussian_kernel(ks: int, sigma: float, device, dtype):
    ax = torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx**2 + yy**2) / (2 * sigma * sigma))
    k = k / k.sum()
    return k[None, None, :, :]  # [1,1,ks,ks]

@torch.no_grad()
def ssim_torch(x01: torch.Tensor, y01: torch.Tensor, ks: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """
    SSIM na tenzorech [1,1,H,W] v rozsahu [0,1]. Vrací skalar (tensor) 0..1.
    """
    device, dtype = x01.device, x01.dtype
    w = _gaussian_kernel(ks, sigma, device, dtype)

    mu_x = F.conv2d(x01, w, padding=ks // 2)
    mu_y = F.conv2d(y01, w, padding=ks // 2)
    mu_x2, mu_y2 = mu_x**2, mu_y**2
    mu_xy = mu_x * mu_y

    sig_x2 = F.conv2d(x01 * x01, w, padding=ks // 2) - mu_x2
    sig_y2 = F.conv2d(y01 * y01, w, padding=ks // 2) - mu_y2
    sig_xy = F.conv2d(x01 * y01, w, padding=ks // 2) - mu_xy

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu_xy + C1) * (2 * sig_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2) + 1e-12
    )
    return ssim_map.mean()  # tensor()


class KspaceEnv(gym.Env):
    """Adaptive line selection along the phase-encode (W) axis.
    Obs: dict{ zf:[2,H,W], mask:[W], budget:int }  Action: Discrete(W)
    Reward: ΔSSIM (zlepšení SSIM vůči předchozímu stavu).
    """
    metadata = {"render_modes": []}

    def __init__(self, dataset, recon, budget: int = 16, device: str = "cpu",
                 acs: int = 16, start_with_acs: bool = True):
        super().__init__()
        self.ds = dataset
        self.device = torch.device(device)
        self.recon = recon.to(self.device)
        self.budget0 = int(budget)
        self.acs = int(acs)
        self.start_with_acs = bool(start_with_acs)

        # probe one sample to set spaces
        sample = self.ds[0]
        _, H, W = sample["kspace"].shape
        self.H, self.W = H, W

        self.action_space = spaces.Discrete(W)
        self.observation_space = spaces.Dict({
            "zf":     spaces.Box(-np.inf, np.inf, shape=(2, self.H, self.W), dtype=np.float32),
            "mask":   spaces.Box(0.0, 1.0, shape=(self.W,), dtype=np.float32),
            "budget": spaces.Box(0.0, float(self.budget0), shape=(1,), dtype=np.float32),
        })

        self._i = 0
        self._reset_slice()

    def _reset_slice(self):
        sample = self.ds[self._i % len(self.ds)]
        self._i += 1

        self.kspace_full = sample["kspace"].permute(1, 2, 0).contiguous().to(self.device)  # [H,W,2]
        self.target_img  = sample["target"].permute(1, 2, 0).contiguous().to(self.device)  # [H,W,2]

        self.mask = torch.zeros(self.W, dtype=torch.bool, device=self.device)
        self.budget = int(self.budget0)

        if self.start_with_acs and self.acs > 0:
            c = self.W // 2
            a = max(0, c - self.acs // 2)
            b = min(self.W, c + self.acs // 2)
            self.mask[a:b] = True
            self.budget = max(0, self.budget - int(self.mask.sum().item()))

        # počáteční zf + počáteční score (SSIM)
        self._update_zf()
        self.prev_score = self._current_ssim()  # SSIM stavu po resetu

    def _update_zf(self):
        # sestav k-space s aktuálním maskem
        k = torch.zeros_like(self.kspace_full)
        sel = self.mask.nonzero(as_tuple=False).view(-1)
        if sel.numel() > 0:
            k[:, sel, :] = self.kspace_full[:, sel, :]
        self.zf = ifft2(k.unsqueeze(0)).squeeze(0)  # [H,W,2]

    @torch.no_grad()
    def _current_ssim(self) -> float:
        """SSIM(recon(zf), GT) na magnitudách v [0,1]."""
        # rekonstruuj z aktuálního ZF a aplikuj data-consistency (stejně jako v tréninku)
        inp = self.zf.permute(2, 0, 1).unsqueeze(0).to(self.device)  # [1,2,H,W]
        recon = self.recon(inp)  # [1,2,H,W]
        # DC: nahradit měřené k-space vzorky predikovanými pouze tam, kde maska=0
        mask4 = self.mask.float().view(1, 1, 1, self.W).to(self.device)  # [1,1,1,W]
        k_full4 = self.kspace_full.permute(2, 0, 1).unsqueeze(0)         # [1,2,H,W]
        k_pred = fft2(recon.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)     # [1,2,H,W]
        k_dc   = torch.where(mask4.bool(), k_full4, k_pred)              # [1,2,H,W]
        recon  = ifft2(k_dc.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)     # [1,2,H,W]
        recon = recon.squeeze(0).permute(1, 2, 0)                        # [H,W,2]
        gt_mag   = _to_mag(self.target_img)       # [1,1,H,W]
        rec_mag  = _to_mag(recon)                 # [1,1,H,W]
        gt_n     = _normalize01_by_ref(gt_mag, gt_mag)
        rec_n    = _normalize01_by_ref(rec_mag, gt_mag)

        ssim_val = ssim_torch(rec_n, gt_n)        # tensor
        return float(ssim_val.item())

    def observation(self):
        return {
            "zf": self.zf.permute(2, 0, 1).float().detach().cpu().numpy(),       # [2,H,W]
            "mask": self.mask.float().detach().cpu().numpy().astype(np.float32), # [W]
            "budget": np.array([float(self.budget)], dtype=np.float32),          # [1]
        }

    # sb3-contrib Maskable PPO wrapper volá tohle
    def action_masks(self):
        # povol akce jen na dosud neměřených sloupcích
        return (~self.mask.detach().cpu().numpy())

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_slice()
        return self.observation(), {}

    def step(self, action: int):
        """
        Jeden akviziční krok:
        - Pokud je akce (sloupec) už změřena, malá penalizace.
        - Jinak se sloupec přidá, přepočte ZF, spočítá se recon a
          odměna je ΔSSIM = SSIM_tentokrát - SSIM_minule.
        """
        col = int(action)
        info: Dict[str, Any] = {}

        if self.mask[col]:
            # neplatná akce -> drobná penalizace
            reward = -0.01
        else:
            self.mask[col] = True
            self._update_zf()

            # nový score = SSIM(recon(zf), GT)
            score = self._current_ssim()
            reward = score - (self.prev_score if self.prev_score is not None else 0.0)
            self.prev_score = score
            info.update({"ssim": score})

        # účetnictví
        self.budget -= 1
        self._last_action = col

        terminated = (self.budget <= 0) or bool(self.mask.all().item())
        truncated = False

        info.update({
            "last_action": col,
            "budget_left": self.budget,
            "measured": int(self.mask.sum().item()),
        })

        return self.observation(), reward, terminated, truncated, info
