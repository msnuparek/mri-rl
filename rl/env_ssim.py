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

    Parametry:
        use_pseudo_ref: bool - Pokud True, používá pseudo-referenci místo GT
                               (pro inferenci bez ground truth).
        pseudo_ref_acs: int - Velikost ACS regionu pro pseudo-referenci.
                              Větší hodnota = kvalitnější pseudo-reference.
    """
    metadata = {"render_modes": []}

    def __init__(self, dataset, recon, budget: int = 16, device: str = "cpu",
                 acs: int = 16, start_with_acs: bool = True,
                 use_pseudo_ref: bool = False, pseudo_ref_acs: int = 48):
        super().__init__()
        self.ds = dataset
        self.device = torch.device(device)
        self.recon = recon.to(self.device)
        self.budget0 = int(budget)
        self.acs = int(acs)
        self.start_with_acs = bool(start_with_acs)
        self.use_pseudo_ref = bool(use_pseudo_ref)
        self.pseudo_ref_acs = int(pseudo_ref_acs)

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
        self._pseudo_ref_mag: Optional[torch.Tensor] = None
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

        # Vytvoř pseudo-referenci pokud je požadována
        if self.use_pseudo_ref:
            self._build_pseudo_reference()

        # počáteční zf + počáteční score (SSIM)
        self._update_zf()
        self.prev_score = self._current_ssim()  # SSIM stavu po resetu

    @torch.no_grad()
    def _build_pseudo_reference(self):
        """Vytvoří pseudo-referenci z rozšířeného ACS regionu.

        Pseudo-reference je rekonstrukce z centrálních k-space linek,
        která slouží jako náhrada GT při inferenci.
        """
        c = self.W // 2
        acs_half = self.pseudo_ref_acs // 2
        a = max(0, c - acs_half)
        b = min(self.W, c + acs_half)

        # Vytvoř k-space pouze s ACS regionem
        k_acs = torch.zeros_like(self.kspace_full)
        k_acs[:, a:b, :] = self.kspace_full[:, a:b, :]

        # Zero-filled rekonstrukce z ACS
        zf_acs = ifft2(k_acs.unsqueeze(0)).squeeze(0)  # [H,W,2]

        # Rekonstrukce pomocí sítě
        inp = zf_acs.permute(2, 0, 1).unsqueeze(0).to(self.device)  # [1,2,H,W]
        recon = self.recon(inp)  # [1,2,H,W]

        # Data-consistency krok pro pseudo-referenci
        acs_mask = torch.zeros(self.W, dtype=torch.bool, device=self.device)
        acs_mask[a:b] = True
        mask4 = acs_mask.float().view(1, 1, 1, self.W)
        k_full4 = self.kspace_full.permute(2, 0, 1).unsqueeze(0)
        k_pred = fft2(recon.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        k_dc = torch.where(mask4.bool(), k_full4, k_pred)
        recon = ifft2(k_dc.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        recon = recon.squeeze(0).permute(1, 2, 0)  # [H,W,2]

        # Ulož magnitudu jako pseudo-referenci
        self._pseudo_ref_mag = _to_mag(recon)  # [1,1,H,W]

    def _update_zf(self):
        # sestav k-space s aktuálním maskem
        k = torch.zeros_like(self.kspace_full)
        sel = self.mask.nonzero(as_tuple=False).view(-1)
        if sel.numel() > 0:
            k[:, sel, :] = self.kspace_full[:, sel, :]
        self.zf = ifft2(k.unsqueeze(0)).squeeze(0)  # [H,W,2]

    @torch.no_grad()
    def _reconstruct_current(self) -> torch.Tensor:
        """Rekonstruuje obraz z aktuálního ZF s data-consistency.

        Returns:
            rec_mag: [1,1,H,W] magnitudový obraz rekonstrukce
        """
        inp = self.zf.permute(2, 0, 1).unsqueeze(0).to(self.device)  # [1,2,H,W]
        recon = self.recon(inp)  # [1,2,H,W]
        # DC: nahradit měřené k-space vzorky predikovanými pouze tam, kde maska=0
        mask4 = self.mask.float().view(1, 1, 1, self.W).to(self.device)  # [1,1,1,W]
        k_full4 = self.kspace_full.permute(2, 0, 1).unsqueeze(0)         # [1,2,H,W]
        k_pred = fft2(recon.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)     # [1,2,H,W]
        k_dc   = torch.where(mask4.bool(), k_full4, k_pred)              # [1,2,H,W]
        recon  = ifft2(k_dc.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)     # [1,2,H,W]
        recon = recon.squeeze(0).permute(1, 2, 0)                        # [H,W,2]
        return _to_mag(recon)  # [1,1,H,W]

    @torch.no_grad()
    def _current_ssim(self) -> float:
        """SSIM na magnitudách v [0,1].

        Pokud use_pseudo_ref=True, počítá SSIM vůči pseudo-referenci.
        Jinak počítá SSIM vůči GT (ground truth).
        """
        rec_mag = self._reconstruct_current()  # [1,1,H,W]

        if self.use_pseudo_ref and self._pseudo_ref_mag is not None:
            # Pseudo-reference režim (pro inferenci bez GT)
            ref_mag = self._pseudo_ref_mag
        else:
            # GT režim (pro trénink)
            ref_mag = _to_mag(self.target_img)  # [1,1,H,W]

        ref_n = _normalize01_by_ref(ref_mag, ref_mag)
        rec_n = _normalize01_by_ref(rec_mag, ref_mag)

        ssim_val = ssim_torch(rec_n, ref_n)
        return float(ssim_val.item())

    @torch.no_grad()
    def _current_ssim_gt(self) -> float:
        """SSIM vůči GT - vždy použije ground truth (pro evaluaci)."""
        rec_mag = self._reconstruct_current()
        gt_mag = _to_mag(self.target_img)
        gt_n = _normalize01_by_ref(gt_mag, gt_mag)
        rec_n = _normalize01_by_ref(rec_mag, gt_mag)
        ssim_val = ssim_torch(rec_n, gt_n)
        return float(ssim_val.item())

    @torch.no_grad()
    def _current_ssim_pseudo(self) -> float:
        """SSIM vůči pseudo-referenci - vždy použije pseudo-ref."""
        if self._pseudo_ref_mag is None:
            self._build_pseudo_reference()
        rec_mag = self._reconstruct_current()
        ref_mag = self._pseudo_ref_mag
        ref_n = _normalize01_by_ref(ref_mag, ref_mag)
        rec_n = _normalize01_by_ref(rec_mag, ref_mag)
        ssim_val = ssim_torch(rec_n, ref_n)
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

        Info dict obsahuje:
        - ssim: SSIM použité pro reward (GT nebo pseudo-ref podle nastavení)
        - ssim_gt: SSIM vůči GT (pouze pokud use_pseudo_ref=True, pro evaluaci)
        - ssim_pseudo: SSIM vůči pseudo-referenci (pouze pokud use_pseudo_ref=True)
        """
        col = int(action)
        info: Dict[str, Any] = {}

        if self.mask[col]:
            # neplatná akce -> drobná penalizace
            reward = -0.01
        else:
            self.mask[col] = True
            self._update_zf()

            # nový score = SSIM(recon(zf), reference)
            score = self._current_ssim()
            reward = score - (self.prev_score if self.prev_score is not None else 0.0)
            self.prev_score = score
            info["ssim"] = score

            # Pro evaluaci: vrať obě SSIM hodnoty při použití pseudo-reference
            if self.use_pseudo_ref:
                info["ssim_pseudo"] = score
                info["ssim_gt"] = self._current_ssim_gt()

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
