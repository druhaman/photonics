"""Обучение DOENet без готовых голограмм: ошибка считается по изображениям, восстановленным Френелем.

    python train.py --iters 6000 [--init weights/doe_net.pt] [--out weights/doe_net.pt]

При обучении сеть делает 16-32 шага уточнения с постоянным шагом 0.04 (случайная длина развёртки),
в последних 30% итераций фаза квантуется до 8 бит (straight-through).
"""
import argparse
import math
import time
from pathlib import Path

import torch

from doe.data import build_pool, sample
from doe.metrics import doe_loss, pearson_t, ssim_t
from doe.net import DOENet
from doe.optics import wrap


def quantize(phi):
    w = wrap(phi)
    q = torch.round((w + math.pi) / (2 * math.pi) * 255) / 255 * 2 * math.pi - math.pi
    return w + (q - w).detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--pool", type=int, default=40000)
    ap.add_argument("--init", default="")
    ap.add_argument("--out", default="weights/doe_net.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                    "mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = args.device
    pool = build_pool(args.pool)
    gen = torch.Generator().manual_seed(1)
    val = sample(build_pool(256, seed=12345), 32, device=dev, generator=gen)

    net = DOENet().to(dev)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=True), strict=False)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1, (i + 1) / 200) * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * i / args.iters))))
    quant_from = int(args.iters * 0.7)
    best, t0 = -1, time.time()
    for it in range(args.iters):
        target = sample(pool, args.batch, device=dev, generator=gen)
        steps = int(torch.randint(16, 33, (1,)))
        phi = net(target, steps=steps, lr=0.04, w_pearson=1.0, w_window=0.0)
        if it >= quant_from:
            phi = quantize(phi)
        loss = doe_loss(net.fresnel.amplitude(phi), target, balance=0.25)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 50 == 0:
            print(f"{it:5d}  loss {loss.item():.4f}  {time.time() - t0:.0f} с", flush=True)
        if (it + 1) % 500 == 0 or it == args.iters - 1:
            net.eval()
            with torch.no_grad():
                rec = net.fresnel.amplitude(quantize(net(val, steps=32, lr=0.04, w_pearson=1.0, w_window=0.0)))
                cc, ss = pearson_t(rec, val).mean().item(), ssim_t(rec, val).mean().item()
            net.train()
            print(f"  проверка: КК {cc:.3f}  ИСС {ss:.3f}", flush=True)
            if cc + ss > best:
                best = cc + ss
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                torch.save({k: v for k, v in net.state_dict().items() if not k.startswith("fresnel.")}, args.out)


if __name__ == "__main__":
    main()
