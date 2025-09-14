import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

__all__ = [
    "SmallUNet",
    "UNetLarge",
    "CascadedRecon",
    "ReconWrapper",
    "build_reconstructor",
]


# ----------------------
# Lightweight baseline
# ----------------------
class SmallUNet(nn.Module):
    """Tiny baseline kept for backward compatibility.
    Input/Output: [B, 2, H, W] real/imag.
    """
    def __init__(self, c: int = 32):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(2, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(c, 2*c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(2*c, 2*c, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.up  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec = nn.Sequential(
            nn.Conv2d(3*c, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c, 2, 3, padding=1)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        u  = self.up(e2)
        cat = torch.cat([u, e1], dim=1)
        return self.dec(cat)


# ----------------------
# Robust UNet backbone
# ----------------------
class SEBlock(nn.Module):
    def __init__(self, c: int, r: int = 8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, max(1, c // r), 1), nn.ReLU(inplace=True),
            nn.Conv2d(max(1, c // r), c, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w

class ResDoubleConv(nn.Module):
    """Conv->GN->ReLU->Conv->GN with residual, optional SE."""
    def __init__(self, cin: int, cout: int, groups: int = 8, use_se: bool = True, p_drop: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.gn1   = nn.GroupNorm(num_groups=min(groups, cout), num_channels=cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.gn2   = nn.GroupNorm(num_groups=min(groups, cout), num_channels=cout)
        self.act   = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity()
        self.proj  = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.se    = SEBlock(cout) if use_se else nn.Identity()
    def forward(self, x):
        idn = self.proj(x)
        x = self.act(self.gn1(self.conv1(x)))
        x = self.drop(self.act(self.gn2(self.conv2(x))))
        x = self.se(x)
        return self.act(x + idn)

class Down(nn.Module):
    def __init__(self, cin, cout, **kw):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ResDoubleConv(cin, cout, **kw)
    def forward(self, x):
        return self.block(self.pool(x))

class Up(nn.Module):
    def __init__(self, cin, cout, **kw):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block = ResDoubleConv(cin, cout, **kw)
    def forward(self, x, skip):
        x = self.up(x)
        # pad if odd sizes
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        if dh or dw:
            x = F.pad(x, (0, dw, 0, dh))
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

class UNetLarge(nn.Module):
    """Deeper UNet with residual blocks, GroupNorm and Squeeze-and-Excitation.

    Recommended when training purely on 2D slices. It is more stable than 3D U‑Net
    for this codebase because our loader feeds single slices; 3D would require
    stacking neighboring slices and redesigning the dataset/augmentation pipeline.
    """
    def __init__(self, base: int = 48, use_se: bool = True, p_drop: float = 0.0):
        super().__init__()
        gk = dict(use_se=use_se, p_drop=p_drop)
        self.inp = ResDoubleConv(2, base, **gk)
        self.d1  = Down(base,     base*2, **gk)
        self.d2  = Down(base*2,   base*4, **gk)
        self.d3  = Down(base*4,   base*8, **gk)
        self.bot = ResDoubleConv(base*8, base*16, **gk)
        self.u3  = Up(base*16 + base*8, base*8, **gk)
        self.u2  = Up(base*8  + base*4, base*4, **gk)
        self.u1  = Up(base*4  + base*2, base*2, **gk)
        self.u0  = Up(base*2  + base,   base,   **gk)
        self.out = nn.Conv2d(base, 2, 3, padding=1)

    def forward(self, x):  # x: [B,2,H,W]
        s0 = self.inp(x)
        s1 = self.d1(s0)
        s2 = self.d2(s1)
        s3 = self.d3(s2)
        b  = self.bot(s3)
        x  = self.u3(b, s3)
        x  = self.u2(x, s2)
        x  = self.u1(x, s1)
        x  = self.u0(x, s0)
        return self.out(x)


# ----------------------
# Cascaded refinement (K>1 UNets)
# ----------------------
class CascadedRecon(nn.Module):
    """Apply K UNetLarge stages to iteratively refine the image (predict residuals).

    Args:
        stages: number of cascades
        base:   base channels per stage (passed to UNetLarge)
        tie_weights: if True, reuse the same UNet across stages (memory‑light)
    """
    def __init__(self, stages: int = 3, base: int = 48, tie_weights: bool = False,
                 use_se: bool = True, p_drop: float = 0.0):
        super().__init__()
        self.stages = int(stages)
        if tie_weights:
            core = UNetLarge(base=base, use_se=use_se, p_drop=p_drop)
            self.blocks = nn.ModuleList([core for _ in range(self.stages)])
        else:
            self.blocks = nn.ModuleList([
                UNetLarge(base=base, use_se=use_se, p_drop=p_drop) for _ in range(self.stages)
            ])

    def forward(self, x):
        y = x
        for blk in self.blocks:
            y = y + blk(y)  # residual refinement
        return y


# ----------------------
# Wrapper & factory
# ----------------------
class ReconWrapper(nn.Module):
    """No‑grad inference wrapper used inside the RL env."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, zf: torch.Tensor) -> torch.Tensor:  # [B,2,H,W]
        self.model.eval()
        return self.model(zf)


def build_reconstructor(name: str = "unet_large", **kw) -> nn.Module:
    name = (name or "").lower()
    if name in ("small", "small_unet", "tiny"):
        return SmallUNet(**kw)
    if name in ("unet", "unet_large", "large"):
        return UNetLarge(**kw)
    if name in ("cascade", "cascaded", "cascaded_unet"):
        return CascadedRecon(**kw)
    raise ValueError(f"Unknown reconstructor '{name}'")


