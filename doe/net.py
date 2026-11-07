"""Сеть синтеза фазового ДОЭ для 4 плоскостей.

1. U-Net по 4 изображениям предсказывает фазу и поправку амплитуды поля в каждой плоскости,
   поля обратно распространяются на ДОЭ и складываются (схема 3D-CGH-Net, Rymov et al. 2025).
2. Вторая U-Net уточняет фазу в плоскости ДОЭ.
3. 16 итераций многоплоскостного Герчберга-Сэкстона с обученными весами плоскостей.
4. Шаги Adam по градиенту ошибки восстановления 2.5*(1 - КК) + (1 - ИСС).
   Число шагов задаёт баланс между скоростью и качеством.
"""
import math

import torch
import torch.nn as nn

from .metrics import doe_loss
from .optics import Fresnel, wrap


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU())


class UNet(nn.Module):
    def __init__(self, cin, cout, base=32, depth=4, zero_head=False):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth)]
        self.downs = nn.ModuleList()
        prev = cin
        for c in chs:
            self.downs.append(block(prev, c))
            prev = c
        self.pool = nn.MaxPool2d(2)
        self.mid = block(prev, prev * 2)
        prev *= 2
        self.upconvs, self.ups = nn.ModuleList(), nn.ModuleList()
        for c in reversed(chs):
            self.upconvs.append(nn.ConvTranspose2d(prev, c, 2, stride=2))
            self.ups.append(block(2 * c, c))
            prev = c
        self.head = nn.Conv2d(prev, cout, 1)
        if zero_head:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.mid(x)
        for up, blk, s in zip(self.upconvs, self.ups, reversed(skips)):
            x = blk(torch.cat([up(x), s], 1))
        return self.head(x)


class DOENet(nn.Module):
    def __init__(self, planes=4, gs_iters=16):
        super().__init__()
        self.fresnel = Fresnel()
        self.phase_unet = UNet(planes, planes, 32)
        self.amp_unet = UNet(planes, planes, 16, zero_head=True)
        self.refine_unet = UNet(3, 1, 16, zero_head=True)
        self.gs_beta = nn.Parameter(torch.zeros(gs_iters, planes))
        self.gs_logw = nn.Parameter(torch.zeros(gs_iters, planes))
        # параметры уточнения при синтезе
        self.steps, self.lr0, self.lr1 = 64, 0.2, 0.004
        self.w_pearson, self.w_window = 2.5, 0.0

    def initial_phase(self, target):
        x = target * 2 - 1
        theta = math.pi * self.phase_unet(x)
        amp = target * torch.exp(self.amp_unet(x).clamp(-3, 3))
        h = self.fresnel.backward(torch.polar(amp, theta)).sum(1, keepdim=True)
        phi = torch.angle(h)
        mag = h.abs() / (h.abs().mean(dim=(-2, -1), keepdim=True) + 1e-8)
        return phi + math.pi * torch.tanh(self.refine_unet(torch.cat([phi.cos(), phi.sin(), mag], 1)))

    def gerchberg_saxton(self, phi, target):
        planes = target.shape[1]
        for t in range(len(self.gs_beta)):
            k = self.fresnel(phi)
            r = k.abs()
            scale = (r * target).sum(dim=(-2, -1), keepdim=True) / ((r * r).sum(dim=(-2, -1), keepdim=True) + 1e-8)
            ratio = ((target + 0.05) / (r * scale + 0.05)).clamp(0.2, 5.0)
            amp = target * ratio ** self.gs_beta[t][None, :, None, None]
            w = torch.softmax(self.gs_logw[t], 0)[None, :, None, None] * planes
            phi = torch.angle((self.fresnel.backward(amp * k / (r + 1e-8)) * w).sum(1, keepdim=True))
        return phi

    def refine(self, phi, target, steps, lr, w_pearson, w_window):
        """Adam по фазе; градиент ошибки считается как внешний сигнал (без производных второго порядка)."""
        m = v = None
        for t in range(steps):
            with torch.inference_mode(False), torch.enable_grad():
                p = phi.detach().clone().requires_grad_(True)
                loss = doe_loss(self.fresnel.amplitude(p), target.clone(), w_pearson, w_window=w_window)
                g, = torch.autograd.grad(loss, p)
            m = 0.1 * g if m is None else 0.9 * m + 0.1 * g
            v = 0.001 * g * g if v is None else 0.999 * v + 0.001 * g * g
            mhat, vhat = m / (1 - 0.9 ** (t + 1)), v / (1 - 0.999 ** (t + 1))
            phi = phi - lr[t] * mhat / (vhat.sqrt() + 1e-8)
        return phi

    def forward(self, target, steps=None, lr=None, w_pearson=None, w_window=None):
        """target: (B,4,N,N), каждое изображение нормировано на максимум -> фаза ДОЭ (B,1,N,N)."""
        steps = self.steps if steps is None else steps
        if lr is None:  # косинусное убывание шага
            t = torch.arange(steps, dtype=torch.float32, device=target.device)
            lr = self.lr1 + 0.5 * (self.lr0 - self.lr1) * (1 + torch.cos(math.pi * t / max(steps, 1)))
        elif not torch.is_tensor(lr):
            lr = torch.full((steps,), float(lr), device=target.device)
        phi = self.initial_phase(target)
        phi = self.gerchberg_saxton(phi, target)
        phi = self.refine(phi, target, steps, lr, self.w_pearson if w_pearson is None else w_pearson,
                          self.w_window if w_window is None else w_window)
        return wrap(phi)


def load_net(path, device="cpu"):
    net = DOENet()
    missing, unexpected = net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=False)
    if unexpected or set(missing) - {"fresnel.chirp"}:
        raise RuntimeError(f"веса не подходят к модели: {missing} {unexpected}")
    return net.to(device).eval()
