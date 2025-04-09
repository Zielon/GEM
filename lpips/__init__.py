import torch

from .modules.lpips import LPIPS

criterion = LPIPS('vgg', '0.1').cuda().eval()

def lpips(x: torch.Tensor, y: torch.Tensor):
    return criterion(x, y)
