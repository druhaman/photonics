"""Проверки: Френель совпадает с формулой со слайда и с прямым интегралом, кодирование фазы, метрики.

    python -m tests.test_optics
"""
import math

import numpy as np
import torch

from doe.metrics import pearson, ssim
from doe.optics import DISTANCES, PIXEL, SIZE, WAVELENGTH, Fresnel, phase_to_png, png_to_phase


def fresnel_formula(phi, z):
    """Формула со слайда как есть, включая множители перед FFT."""
    c = np.arange(SIZE) - SIZE / 2
    u, v = c[None, :] * PIXEL, c[:, None] * PIXEL
    inner = np.exp(1j * phi) * np.exp(1j * np.pi / (WAVELENGTH * z) * (u ** 2 + v ** 2))
    f = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(inner))) * PIXEL ** 2
    dx = WAVELENGTH * z / (SIZE * PIXEL)
    x, y = c[None, :] * dx, c[:, None] * dx
    k = 2 * np.pi / WAVELENGTH
    return np.exp(1j * k * z) / (1j * WAVELENGTH * z) * np.exp(1j * np.pi / (WAVELENGTH * z) * (x ** 2 + y ** 2)) * f


def test_fresnel():
    phi = np.random.default_rng(0).uniform(-np.pi, np.pi, (SIZE, SIZE))
    amp = Fresnel().amplitude(torch.from_numpy(phi).float()[None, None])[0].numpy()
    for k, z in enumerate(DISTANCES):
        ref = np.abs(fresnel_formula(phi, z))
        assert np.abs(amp[k] - ref / ref.max()).max() < 1e-4
    # формула с одним FFT == прямой интеграл Френеля
    z = DISTANCES[0]
    ref = fresnel_formula(phi, z)
    c = (np.arange(SIZE) - SIZE / 2) * PIXEL
    u, v = np.meshgrid(c, c)
    dx = WAVELENGTH * z / (SIZE * PIXEL)
    for iy, ix in [(64, 64), (5, 120), (100, 30)]:
        x, y = (ix - SIZE / 2) * dx, (iy - SIZE / 2) * dx
        direct = np.exp(2j * np.pi / WAVELENGTH * z) / (1j * WAVELENGTH * z) * np.sum(
            np.exp(1j * phi) * np.exp(1j * np.pi / (WAVELENGTH * z) * ((x - u) ** 2 + (y - v) ** 2))) * PIXEL ** 2
        assert abs(direct - ref[iy, ix]) / abs(direct) < 1e-6


def test_orientation():
    """Линза с наклоном фокусирует свет в ожидаемый пиксель (изображение не зеркалится)."""
    z = DISTANCES[2]
    dx = WAVELENGTH * z / (SIZE * PIXEL)
    c = (np.arange(SIZE) - SIZE / 2) * PIXEL
    u, v = np.meshgrid(c, c)
    lens = -np.pi / (WAVELENGTH * z) * (u ** 2 + v ** 2) + 2 * np.pi / (WAVELENGTH * z) * (u * 20 * dx + v * 35 * dx)
    amp = Fresnel().amplitude(torch.from_numpy(lens).float()[None, None])[0, 2].numpy()
    assert np.unravel_index(amp.argmax(), amp.shape) == (64 + 35, 64 + 20)


def test_phase_png():
    phi = np.random.default_rng(1).uniform(-np.pi, np.pi, (SIZE, SIZE))
    err = np.abs(np.angle(np.exp(1j * (png_to_phase(phase_to_png(phi)) - phi)))).max()
    assert err <= math.pi / 255 + 1e-9
    assert png_to_phase(np.array(0)) == -math.pi and png_to_phase(np.array(255)) == math.pi


def test_metrics():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 256, (SIZE, SIZE))
    b = np.clip(a + rng.integers(-40, 40, (SIZE, SIZE)), 0, 255)
    assert abs(pearson(a, b) - np.corrcoef(a.ravel(), b.ravel())[0, 1]) < 1e-12
    assert abs(pearson(a, a) - 1) < 1e-12 and abs(ssim(a, a) - 1) < 1e-12


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("OK", name)
