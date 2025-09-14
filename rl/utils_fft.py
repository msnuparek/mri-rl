import torch

def _to_complex(x):
    """x: [B,2,H,W] or [...,2] -> (complex tensor [...,H,W], layout)"""
    if x.dim() == 4 and x.size(1) == 2:
        return torch.complex(x[:, 0, ...], x[:, 1, ...]), "first"
    if x.size(-1) == 2:
        return torch.complex(x[..., 0], x[..., 1]), "last"
    raise ValueError(f"Expected 2-channel real/imag, got {tuple(x.shape)}")

def _from_complex(z, layout):
    return (torch.stack([z.real, z.imag], dim=1) if layout == "first"
            else torch.stack([z.real, z.imag], dim=-1))

def ifft2(x, norm: str = "backward"):
    z, layout = _to_complex(x.float())
    zi = torch.fft.ifftn(z, dim=(-2, -1), norm=norm)
    return _from_complex(zi, layout)

def fft2(x, norm: str = "backward"):
    z, layout = _to_complex(x.float())
    zf = torch.fft.fftn(z, dim=(-2, -1), norm=norm)
    return _from_complex(zf, layout)

def ifft2_centered(x, norm: str = "ortho"):
    z, layout = _to_complex(x.float())
    z = torch.fft.ifftshift(z, dim=(-2, -1))
    zi = torch.fft.ifftn(z, dim=(-2, -1), norm=norm)
    zi = torch.fft.fftshift(zi, dim=(-2, -1))
    return _from_complex(zi, layout)

def fft2_centered(x, norm: str = "ortho"):
    """Centered 2D FFT with orthonormal scaling to pair with ifft2_centered.

    Accepts [B,2,H,W] or [...,2] real/imag layout and returns the same layout.
    """
    z, layout = _to_complex(x.float())
    z = torch.fft.ifftshift(z, dim=(-2, -1))
    zf = torch.fft.fftn(z, dim=(-2, -1), norm=norm)
    zf = torch.fft.fftshift(zf, dim=(-2, -1))
    return _from_complex(zf, layout)
