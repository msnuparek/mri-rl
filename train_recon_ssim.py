# train_recon_ssim.py
import argparse, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rl.fastmri_loader import SingleCoilKneeDataset
from rl.utils_fft import ifft2_centered as ifft2, fft2_centered as fft2
from rl.reconstructor import build_reconstructor

# ----------------------------
# Utils: complex -> magnitude
# ----------------------------
def to_mag(x: torch.Tensor) -> torch.Tensor:
    # x: [B,2,H,W]  -> [B,1,H,W]
    return torch.sqrt(x[:, :1]**2 + x[:, 1:2]**2 + 1e-12)

def normalize01_by_ref(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    # normalizace podle GT: [0, max(GT)], robustní a stabilní
    maxv = ref.amax(dim=(-2, -1), keepdim=True)
    return torch.clamp(x / (maxv + 1e-8), 0.0, 1.0)

# ----------------------------
# Differentiable SSIM (single-channel, [0,1])
# ----------------------------
def _gaussian_kernel(ks: int = 11, sigma: float = 1.5, device="cpu", dtype=torch.float32):
    ax = torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel[None, None, :, :]  # [1,1,ks,ks]

class SSIMLoss(nn.Module):
    """Returns 1 - SSIM (minimize). Inputs must be in [0,1], shape [B,1,H,W]."""
    def __init__(self, ks: int = 11, sigma: float = 1.5):
        super().__init__()
        self.ks = ks
        self.sigma = sigma
        self.register_buffer("_dummy", torch.zeros(1))  # pro zařízení/dtype

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4 and x.size(1) == 1, "SSIMLoss expects [B,1,H,W] in [0,1]"
        assert y.shape == x.shape
        device, dtype = x.device, x.dtype
        w = _gaussian_kernel(self.ks, self.sigma, device=device, dtype=dtype)

        # lokální momenty
        mu_x = F.conv2d(x, w, padding=self.ks // 2)
        mu_y = F.conv2d(y, w, padding=self.ks // 2)
        mu_x2, mu_y2 = mu_x**2, mu_y**2
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(x * x, w, padding=self.ks // 2) - mu_x2
        sigma_y2 = F.conv2d(y * y, w, padding=self.ks // 2) - mu_y2
        sigma_xy = F.conv2d(x * y, w, padding=self.ks // 2) - mu_xy

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
            (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-12
        )
        ssim_val = ssim_map.mean()
        return 1.0 - ssim_val  # loss

@torch.no_grad()
def ssim_metric(x: torch.Tensor, y: torch.Tensor) -> float:
    """SSIM metriku (ne gradient) na [B,1,H,W] v [0,1]."""
    loss = SSIMLoss()(x, y)
    return float(1.0 - loss.item())

# ----------------------------
# Data consistency
# ----------------------------
def apply_dc(img_pred: torch.Tensor, k_full: torch.Tensor, mask4: torch.Tensor) -> torch.Tensor:
    # img_pred: [B,2,H,W] -> FFT -> nahrazení měřených vzorků -> IFFT -> [B,2,H,W]
    k_pred = fft2(img_pred.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    k_dc   = torch.where(mask4.bool(), k_full, k_pred)
    img_dc = ifft2(k_dc.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    return img_dc

# ----------------------------
# Masky (1D sloupce s ACS)
# ----------------------------
def sample_mask(W: int, accel: int = 2, acs: int = 32, rng: np.random.Generator | None = None) -> torch.Tensor:
    rng = np.random.default_rng() if rng is None else rng
    m = np.zeros(W, dtype=bool)
    c = W // 2
    a, b = max(0, c - acs // 2), min(W, c + acs // 2)
    m[a:b] = True
    target = max(acs, W // max(1, accel))
    need = max(0, target - m.sum())
    if need > 0:
        pool = np.setdiff1d(np.arange(W), np.where(m)[0])
        pick = rng.choice(pool, size=need, replace=False)
        m[pick] = True
    return torch.from_numpy(m.astype(np.float32))

def sample_mask_batch(B: int, W: int, accel: int, acs: int) -> torch.Tensor:
    m = torch.stack([sample_mask(W, accel, acs) for _ in range(B)], dim=0)  # [B, W]
    return m.view(B, 1, 1, W)

@torch.no_grad()
def zf_from_masked_k(kspace: torch.Tensor, mask4: torch.Tensor) -> torch.Tensor:
    k_masked = kspace * mask4
    zf = ifft2(k_masked.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    return zf

# ----------------------------
# Trénink / evaluace (SSIM)
# ----------------------------
def train_epoch(model, loader, opt, device, accel: int, acs: int, loss_mode: str, alpha: float):
    model.train()
    ssim_loss = SSIMLoss().to(device)

    total_loss = total_zf_mse = total_pr_mse = 0.0
    total_ssim = 0.0
    n = 0

    for batch in loader:
        k  = batch["kspace"].to(device)   # [B,2,H,W]
        gt = batch["target"].to(device)   # [B,2,H,W]
        B, _, H, W = k.shape

        mask4 = sample_mask_batch(B, W, accel, acs).to(device)

        with torch.no_grad():
            zf = zf_from_masked_k(k, mask4)

        pred = model(zf)
        pred = apply_dc(pred, k, mask4)

        # --- loss na magnitudě v [0,1] ---
        gt_mag   = to_mag(gt)
        pred_mag = to_mag(pred)
        gt_n   = normalize01_by_ref(gt_mag, gt_mag)
        pred_n = normalize01_by_ref(pred_mag, gt_mag)

        if loss_mode == "ssim":
            loss = ssim_loss(pred_n, gt_n)
        elif loss_mode == "ssim_l1":
            loss = alpha * ssim_loss(pred_n, gt_n) + (1 - alpha) * F.l1_loss(pred_n, gt_n)
        else:
            raise ValueError("--loss must be 'ssim' or 'ssim_l1'")

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            mse_zf = torch.mean((to_mag(zf) - gt_mag) ** 2).item()
            mse_pr = torch.mean((pred_mag - gt_mag) ** 2).item()
            ssim_val = 1.0 - ssim_loss(pred_n, gt_n).item()

        bs = k.size(0)
        total_loss     += float(loss.item()) * bs
        total_zf_mse   += mse_zf * bs
        total_pr_mse   += mse_pr * bs
        total_ssim     += ssim_val * bs
        n += bs

    return (total_loss / n, total_zf_mse / n, total_pr_mse / n, total_ssim / n)

@torch.no_grad()
def eval_epoch(model, loader, device, accel: int, acs: int, loss_mode: str, alpha: float):
    model.eval()
    ssim_loss = SSIMLoss().to(device)

    total_loss = total_zf_mse = total_pr_mse = 0.0
    total_ssim = 0.0
    n = 0

    for batch in loader:
        k  = batch["kspace"].to(device)
        gt = batch["target"].to(device)
        B, _, H, W = k.shape

        mask4 = sample_mask_batch(B, W, accel, acs).to(device)
        zf   = zf_from_masked_k(k, mask4)
        pred = model(zf)
        pred = apply_dc(pred, k, mask4)

        gt_mag   = to_mag(gt)
        pred_mag = to_mag(pred)
        zf_mag   = to_mag(zf)

        gt_n   = normalize01_by_ref(gt_mag, gt_mag)
        pred_n = normalize01_by_ref(pred_mag, gt_mag)

        if loss_mode == "ssim":
            loss = ssim_loss(pred_n, gt_n)
        else:
            loss = alpha * ssim_loss(pred_n, gt_n) + (1 - alpha) * F.l1_loss(pred_n, gt_n)

        mse_zf = torch.mean((zf_mag - gt_mag) ** 2).item()
        mse_pr = torch.mean((pred_mag - gt_mag) ** 2).item()
        ssim_val = 1.0 - loss.item() if loss_mode == "ssim" else 1.0 - (alpha * (1.0 - ssim_metric(pred_n, gt_n)) + (1 - alpha) * F.l1_loss(pred_n, gt_n).item())

        bs = k.size(0)
        total_loss   += float(loss.item()) * bs
        total_zf_mse += mse_zf * bs
        total_pr_mse += mse_pr * bs
        total_ssim   += ssim_val * bs
        n += bs

    return (total_loss / n, total_zf_mse / n, total_pr_mse / n, total_ssim / n)

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    from cli_args import parse_train_recon_ssim_args
    args = parse_train_recon_ssim_args()

    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    train_ds = SingleCoilKneeDataset(args.train_list)
    val_ds   = SingleCoilKneeDataset(args.val_list)

    # Poznámka: na Windows je bezpečnější num_workers=0 nebo 2; zde nechám 0 pro robustnost
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)

    device = torch.device(args.device)
    model = build_reconstructor("unet_large", base=64, use_se=True, p_drop=0.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr) 

    best_ssim = -1.0
    for ep in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        tr_loss, tr_mse_zf, tr_mse_pr, tr_ssim = train_epoch(
            model, train_ld, opt, device, args.accel, args.acs, args.loss, args.alpha
        )
        va_loss, va_mse_zf, va_mse_pr, va_ssim = eval_epoch(
            model, val_ld, device, args.accel, args.acs, args.loss, args.alpha
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        print(
            f"epoch {ep:02d}: "
            f"train loss {tr_loss:.4f} | val loss {va_loss:.4f} | "
            f"val MSE ZF {va_mse_zf:.4f} -> Recon {va_mse_pr:.4f} | Δ {(va_mse_zf - va_mse_pr):.4f} | "
            f"val SSIM {va_ssim:.4f} | t/epoch {dt:.2f}s"
        )

        if va_ssim > best_ssim:
            best_ssim = va_ssim
            torch.save({"state_dict": model.state_dict()}, args.save)
            print(f"  saved best (SSIM={best_ssim:.4f}) -> {args.save}")
