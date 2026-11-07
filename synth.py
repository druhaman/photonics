"""Синтез фазового ДОЭ по комплектам из 4 изображений.

    python synth.py INPUT OUTPUT [--steps 64] [--recon]

INPUT  — папка с 4 PNG (один комплект) или папка с подпапками-комплектами.
         Изображения сортируются по имени: 1-е -> 24.1 мм, ..., 4-е -> 27.7 мм.
OUTPUT — для каждого комплекта OUTPUT/<комплект>/doe.png (0 = -pi, 255 = +pi);
         с --recon ещё 4 восстановленных изображения; OUTPUT/times.csv — время синтеза.
"""
import argparse
import csv
import re
import time
from pathlib import Path

import numpy as np
import torch

from doe.net import load_net
from doe.optics import load_image, phase_to_png, reconstruct, save_image

ROOT = Path(__file__).parent


def natural_key(p):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def pngs(folder):
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png"), key=natural_key)


def find_sets(folder):
    """Подпапки с 4 PNG — комплекты; если подпапок нет, комплектом считается сама папка."""
    folder = Path(folder)
    sets = {}
    for sub in sorted((d for d in folder.iterdir() if d.is_dir()), key=natural_key):
        files = pngs(sub)
        if len(files) == 4:
            sets[sub.name] = files
        elif files:
            print(f"пропуск {sub}: {len(files)} PNG вместо 4")
    if not sets and len(pngs(folder)) == 4:
        sets[folder.name] = pngs(folder)
    return sets


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    torch.set_num_threads(1)  # для 128x128 один поток быстрее многопоточного режима
    return torch.device("cpu")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser(description="Синтез фазового ДОЭ")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--weights", default=str(ROOT / "weights" / "doe_net.pt"))
    ap.add_argument("--steps", type=int, default=64, help="шаги уточнения: больше — точнее и медленнее")
    ap.add_argument("--pearson", type=float, default=2.5, help="вес КК в ошибке шагов уточнения")
    ap.add_argument("--window", type=float, default=0.0, help="вес оконного SSIM в ошибке (медленнее на CPU)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--recon", action="store_true", help="сохранить восстановленные изображения")
    args = ap.parse_args()

    device = pick_device(args.device)
    net = load_net(args.weights, device)
    net.w_pearson, net.w_window = args.pearson, args.window
    sets = find_sets(args.input)
    if not sets:
        raise SystemExit(f"в {args.input} не найдено комплектов из 4 PNG")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    warm = torch.rand(1, 4, 128, 128, device=device)
    with torch.no_grad():
        net(warm, steps=2)
    rows = []
    for name, files in sets.items():
        images = np.stack([load_image(f) for f in files]).astype(np.float32)
        target = torch.from_numpy(images)[None].to(device)
        target = target / target.amax(dim=(-2, -1), keepdim=True).clamp_min(1)
        sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            phi = net(target, steps=args.steps)
        sync(device)
        dt = time.perf_counter() - t0

        doe = phase_to_png(phi[0, 0].cpu().numpy())
        (out / name).mkdir(exist_ok=True)
        save_image(out / name / "doe.png", doe)
        if args.recon:
            for k, img in enumerate(reconstruct(doe), 1):
                save_image(out / name / f"recon_{k}.png", img)
        rows.append({"set": name, "time_s": round(dt, 4)})
        print(f"{name}: {dt:.3f} с")

    with open(out / "times.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["set", "time_s"])
        w.writeheader()
        w.writerows(rows)
    print(f"комплектов: {len(rows)}, среднее время {np.mean([r['time_s'] for r in rows]):.3f} с ({device})")


if __name__ == "__main__":
    main()
