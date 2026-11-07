"""Оценка: синтез ДОЭ сетью и алгоритмом Герчберга-Сэкстона, восстановление из сохранённых PNG,
метрики со слайда (КК и ИСС, усреднение по 4 изображениям комплекта) и время.

    python evaluate.py testsets [--steps 64] [--gs-iters 200] [--out results]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from doe.gs import gerchberg_saxton
from doe.metrics import pearson, ssim, ssim_windowed
from doe.net import load_net
from doe.optics import Fresnel, load_image, phase_to_png, reconstruct, save_image
from synth import ROOT, find_sets, pick_device, sync


def score(images, doe):
    recon = reconstruct(doe)
    return (np.mean([pearson(a, b) for a, b in zip(images, recon)]),
            np.mean([ssim(a, b) for a, b in zip(images, recon)]),
            np.mean([ssim_windowed(a, b) for a, b in zip(images, recon)]), recon)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--weights", default=str(ROOT / "weights" / "doe_net.pt"))
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--gs-iters", type=int, default=200)
    ap.add_argument("--pearson", type=float, default=2.5, help="вес КК в ошибке шагов уточнения")
    ap.add_argument("--window", type=float, default=0.0, help="вес оконного SSIM в ошибке (медленнее на CPU)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    device = pick_device(args.device)
    net = load_net(args.weights, device)
    net.w_pearson, net.w_window = args.pearson, args.window
    fresnel = Fresnel().to(device)
    methods = {
        "net": lambda t: net(t, steps=args.steps),
        "gs": lambda t: gerchberg_saxton(fresnel, t, args.gs_iters),
    }
    warm = torch.rand(1, 4, 128, 128, device=device)
    with torch.no_grad():
        for f in methods.values():
            f(warm)

    out = Path(args.out)
    lines = ["| комплект | метод | время, с | КК | ИСС | SSIM (окно 11) |", "|---|---|---|---|---|---|"]
    summary = {m: [] for m in methods}
    sets = find_sets(args.input)
    if not sets:
        raise SystemExit(f"в {args.input} не найдено комплектов из 4 PNG")
    for name, files in sets.items():
        images = [load_image(f) for f in files]
        target = torch.from_numpy(np.stack(images).astype(np.float32))[None].to(device)
        target = target / target.amax(dim=(-2, -1), keepdim=True).clamp_min(1)
        for m, f in methods.items():
            sync(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                phi = f(target)
            sync(device)
            dt = time.perf_counter() - t0
            doe = phase_to_png(phi[0, 0].cpu().numpy())
            cc, ss, sw, recon = score(images, doe)
            summary[m].append((dt, cc, ss, sw))
            d = out / name / m
            d.mkdir(parents=True, exist_ok=True)
            save_image(d / "doe.png", doe)
            for k, img in enumerate(recon, 1):
                save_image(d / f"recon_{k}.png", img)
            lines.append(f"| {name} | {m} | {dt:.3f} | {cc:.3f} | {ss:.3f} | {sw:.3f} |")
    for m, v in summary.items():
        v = np.mean(v, 0)
        lines.append(f"| **среднее** | **{m}** | {v[0]:.3f} | **{v[1]:.3f}** | **{v[2]:.3f}** | {v[3]:.3f} |")
    report = "\n".join(lines)
    (out / "report.md").write_text(f"Устройство: {device}, шагов сети: {args.steps}, итераций GS: {args.gs_iters}\n\n{report}\n")
    print(report)


if __name__ == "__main__":
    main()
