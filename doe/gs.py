"""Многоплоскостной Герчберг-Сэкстон (для сравнения)."""
import math

import torch

from .optics import wrap


@torch.no_grad()
def gerchberg_saxton(fresnel, target, iters=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    rnd = (torch.rand(target.shape, generator=g) * 2 - 1) * math.pi
    phi = torch.angle(fresnel.backward(torch.polar(target, rnd.to(target.device))).sum(1, keepdim=True))
    for _ in range(iters):
        k = fresnel(phi)
        k = torch.polar(target, torch.angle(k))
        phi = torch.angle(fresnel.backward(k).sum(1, keepdim=True))
    return wrap(phi)
