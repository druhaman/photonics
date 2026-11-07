"""Тестовые комплекты (4 изображения 128x128, 8 бит) в папке testsets/.

    python make_testsets.py [--mnist PATH_t10k-images-idx3-ubyte.gz] [--bsds PATH_BSDS500/data/images/test]

Комплекты MNIST и BSDS500 берутся из тестовых частей этих датасетов (в обучении не участвуют).
"""
import argparse
import gzip
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

N = 128
OUT = Path(__file__).parent / "testsets"
FONTS = ["/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial Unicode.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


def font(size):
    for f in FONTS:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default(size)


def glyph(ch, size=100, box=(0, 0, N, N)):
    """Символ по центру прямоугольника box на чёрном фоне."""
    im = Image.new("L", (N, N), 0)
    d = ImageDraw.Draw(im)
    f = font(size)
    l, t, r, b = d.textbbox((0, 0), ch, font=f)
    x0, y0, x1, y1 = box
    d.text(((x0 + x1 - (r - l)) / 2 - l, (y0 + y1 - (b - t)) / 2 - t), ch, fill=255, font=f)
    return np.asarray(im)


CORNERS = [(0, 0, 64, 64), (64, 0, 128, 64), (0, 64, 64, 128), (64, 64, 128, 128)]


def corner_set(chars):
    """Изображение k: символ chars[k] в углу k (левый верхний, правый верхний, левый нижний, правый нижний)."""
    return [glyph(c, 50, CORNERS[k]) for k, c in enumerate(chars)]


def corners_all(rows):
    """В каждом изображении 4 символа во всех углах."""
    out = []
    for chars in rows:
        im = np.zeros((N, N), np.uint8)
        for k, c in enumerate(chars):
            im = np.maximum(im, glyph(c, 50, CORNERS[k]))
        out.append(im)
    return out


def shapes(kind):
    im = Image.new("L", (N, N), 0)
    d = ImageDraw.Draw(im)
    if kind == 0:
        d.ellipse([24, 24, 104, 104], outline=255, width=8)
    elif kind == 1:
        d.rectangle([28, 28, 100, 100], fill=200)
    elif kind == 2:
        d.polygon([(64, 16), (112, 108), (16, 108)], outline=255, width=8)
    else:
        for i in range(4):
            d.line([(16, 24 + i * 26), (112, 24 + i * 26)], fill=255, width=6)
    return np.asarray(im)


def photo(a):
    a = np.asarray(a, np.float64)
    if a.ndim == 3:
        a = a[..., :3] @ [0.2125, 0.7154, 0.0721]
    a = a / (255 if a.max() > 1.5 else 1)
    return np.asarray(Image.fromarray(np.uint8(a * 255)).resize((N, N), Image.BICUBIC))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnist", default="")
    ap.add_argument("--bsds", default="")
    args = ap.parse_args()

    sets = {
        "letters": [glyph(c) for c in "МИФИ"],
        "digits": [glyph(c) for c in "1234"],
        "shapes": [shapes(k) for k in range(4)],
        "corners_digits": corner_set("2580"),
        "corners_letters": corner_set("ABCD"),
        "corners_symbols": corner_set("#♥Σ@"),
        "corners_all": corners_all(["1A#2", "B3♥C", "4Σ@D", "%5E♦"]),
    }
    try:
        from skimage import data
        sets["natural"] = [photo(data.camera()), photo(data.coins()), photo(data.moon()), photo(data.astronaut())]
    except ImportError:
        print("scikit-image не установлен — комплект natural пропущен")
    if args.mnist:
        with gzip.open(args.mnist) as f:
            raw = f.read()
        mn = np.frombuffer(raw, np.uint8, offset=16).reshape(-1, 28, 28)
        sets["mnist"] = [np.asarray(Image.fromarray(mn[i]).resize((N, N), Image.BILINEAR)) for i in (3, 2, 1, 18)]
    if args.bsds:
        files = sorted(Path(args.bsds).glob("*.jpg"))
        sets["bsds"] = [photo(Image.open(files[i]).convert("L")) for i in (0, 50, 100, 150)]

    for name, imgs in sets.items():
        (OUT / name).mkdir(parents=True, exist_ok=True)
        for k, im in enumerate(imgs, 1):
            Image.fromarray(np.uint8(im), mode="L").save(OUT / name / f"{k}.png")
    print("комплекты:", ", ".join(sets))


if __name__ == "__main__":
    main()
