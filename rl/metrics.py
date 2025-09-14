import torch
import torch.nn.functional as F

def nmse(pred, target):
    num = torch.sum((pred - target) ** 2)
    den = torch.sum(target ** 2) + 1e-12
    return (num / den).item()

def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2) + 1e-12
    return (20.0 * torch.log10(1.0 / torch.sqrt(mse))).item()  # expects inputs scaled to ~[0,1]