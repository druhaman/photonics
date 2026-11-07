"""Обучающие изображения 128x128: MNIST, Fashion-MNIST, фото (COCO, BSDS500), буквы и фигуры.

Датасеты кладутся в data/ (ссылки в README). Отсутствующие источники пропускаются.
"""
import gzip
import io
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

N = 128
DATA = Path(__file__).resolve().parent.parent / "data"
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZАБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЭЮЯ"
FONTS = [f for f in ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
                     "/System/Library/Fonts/Supplemental/Courier New.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"] if Path(f).exists()]


def _font(size):
    return ImageFont.truetype(random.choice(FONTS), size) if FONTS else ImageFont.load_default(size)


def procedural(rng):
    im = Image.new("L", (N, N), 0)
    d = ImageDraw.Draw(im)
    if rng.random() < 0.35:
        k = rng.choice([1, 1, 2, 3, 4])
        text = "".join(rng.choice(CHARS) for _ in range(k))
        f = _font(rng.randint(30, 110) if k == 1 else rng.randint(24, 60))
        l, t, r, b = d.textbbox((0, 0), text, font=f)
        x = rng.randint(min(0, N - (r - l)), max(0, N - (r - l))) - l
        y = rng.randint(min(0, N - (b - t)), max(0, N - (b - t))) - t
        d.text((x, y), text, fill=rng.randint(150, 255), font=f)
    else:
        for _ in range(rng.randint(1, 6)):
            x0, y0 = rng.randint(0, N - 10), rng.randint(0, N - 10)
            box = [x0, y0, rng.randint(x0 + 5, N), rng.randint(y0 + 5, N)]
            v = rng.randint(60, 255)
            kind = rng.random()
            if kind < 0.35:
                d.ellipse(box, fill=v if rng.random() < 0.6 else None, outline=v, width=rng.randint(2, 6))
            elif kind < 0.7:
                d.rectangle(box, fill=v if rng.random() < 0.6 else None, outline=v, width=rng.randint(2, 6))
            else:
                d.line(box, fill=v, width=rng.randint(2, 8))
    return np.asarray(im, np.float32) / 255


def _photos():
    files = sorted((DATA / "val2017").glob("*.jpg"))
    root = DATA / "BSDS500-master"
    if root.exists():  # test-часть BSDS500 оставлена для проверки
        files += sorted(f for f in root.rglob("*.jpg") if f.parent.name in ("train", "val"))
    out = []
    for f in files:
        im = Image.open(f).convert("L")
        im.thumbnail((200, 200))
        out.append(np.asarray(im, np.float32) / 255)
    return out


def _small_images():
    out = []
    f = DATA / "mnist" / "train-images-idx3-ubyte.gz"
    if f.exists():
        with gzip.open(f) as g:
            out.append(np.frombuffer(g.read(), np.uint8, offset=16).reshape(-1, 28, 28))
    f = DATA / "fashion_train.parquet"
    if f.exists():
        import pyarrow.parquet as pq
        col = pq.read_table(f).column("image").to_pylist()
        out.append(np.stack([np.asarray(Image.open(io.BytesIO(c["bytes"])).convert("L")) for c in col]))
    return out


def build_pool(size=40000, seed=0):
    """Пул uint8 (size, N, N): 15% MNIST/Fashion-MNIST, 55% кропы фото, 30% буквы и фигуры."""
    cache = DATA / f"pool_{size}_{seed}.pt"
    if cache.exists():
        return torch.load(cache)
    rng = random.Random(seed)
    photos, small = _photos(), _small_images()
    imgs = []
    for _ in range(size):
        r = rng.random()
        if r < 0.15 and small:
            arr = rng.choice(small)
            s = rng.randint(64, 128)
            digit = Image.fromarray(arr[rng.randrange(len(arr))]).resize((s, s), Image.BILINEAR)
            a = np.zeros((N, N), np.float32)
            y, x = rng.randint(0, N - s), rng.randint(0, N - s)
            a[y:y + s, x:x + s] = np.asarray(digit, np.float32) / 255
        elif r < 0.7 and photos:
            p = rng.choice(photos)
            s = rng.randint(min(p.shape) // 4, min(p.shape))
            y, x = rng.randint(0, p.shape[0] - s), rng.randint(0, p.shape[1] - s)
            crop = Image.fromarray(np.uint8(p[y:y + s, x:x + s] * 255)).resize((N, N), Image.BILINEAR)
            a = np.asarray(crop, np.float32) / 255
        else:
            a = procedural(rng)
        if rng.random() < 0.5:
            a = a[:, ::-1]
        a = np.rot90(a, rng.randint(0, 3))
        if rng.random() < 0.3:
            a = a ** rng.uniform(0.6, 1.8)
        imgs.append(np.uint8(np.clip(np.round(a * 255), 0, 255)))
    pool = torch.from_numpy(np.stack(imgs))
    DATA.mkdir(exist_ok=True)
    torch.save(pool, cache)
    return pool


def sample(pool, batch, planes=4, device="cpu", generator=None):
    """Случайные комплекты (batch, planes, N, N), каждое изображение нормировано на максимум."""
    idx = torch.randint(0, len(pool), (batch * planes,), generator=generator)
    x = pool[idx].float().view(batch, planes, N, N)
    return (x / x.amax(dim=(-2, -1), keepdim=True).clamp_min(1)).to(device)
