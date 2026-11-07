"""Дифракция Френеля с одним БПФ и запись фазы в PNG."""
import math

import numpy as np
import torch
from PIL import Image

WAVELENGTH = 532e-9
PIXEL = 10e-6
SIZE = 128
DISTANCES = (24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3)


def wrap(phi):
    return torch.remainder(phi + math.pi, 2 * math.pi) - math.pi


def fft2c(x):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def ifft2c(x):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


class Fresnel(torch.nn.Module):
    """K = C(z) * exp(jpi(x^2+y^2)/(lz)) * FFT[ exp(j*phi) * exp(jpi(u^2+v^2)/(lz)) ].

    Множители перед FFT меняют только фазу и общий масштаб поля, поэтому для |K| они опущены.
    """

    def __init__(self, distances=DISTANCES, wavelength=WAVELENGTH, pixel=PIXEL, size=SIZE):
        super().__init__()
        c = torch.arange(size, dtype=torch.float64) - size / 2
        r2 = (c[None, :] ** 2 + c[:, None] ** 2) * pixel ** 2
        z = torch.tensor(distances, dtype=torch.float64)[:, None, None]
        chirp = torch.remainder(math.pi * r2 / (wavelength * z), 2 * math.pi)
        self.register_buffer("chirp", chirp.float())

    def forward(self, phi):
        """phi (B,1,N,N) -> комплексное поле (B,Z,N,N) в плоскостях восстановления."""
        return fft2c(torch.polar(torch.ones_like(self.chirp), phi + self.chirp))

    def backward(self, field):
        """Обратное распространение из плоскостей (B,Z,N,N) в плоскость ДОЭ."""
        return ifft2c(field) * torch.polar(torch.ones_like(self.chirp), -self.chirp)

    def amplitude(self, phi):
        """Восстановленные изображения: |K|, нормированные на максимум."""
        k = self(phi)
        a = torch.sqrt(k.real ** 2 + k.imag ** 2 + 1e-12)
        return a / a.amax(dim=(-2, -1), keepdim=True)


def load_image(path, size=SIZE):
    im = Image.open(path)
    if im.mode.startswith("I"):
        a = np.asarray(im, np.float64)
        im = Image.fromarray(np.uint8(np.round(a / max(a.max(), 1) * 255)))
    im = im.convert("L")
    if im.size != (size, size):
        im = im.resize((size, size), Image.BICUBIC)
    return np.asarray(im, np.uint8)


def save_image(path, a):
    Image.fromarray(np.asarray(a, np.uint8), mode="L").save(path)


def phase_to_png(phi):
    """[-pi, pi) -> 0..255 (0 = -pi, 255 = +pi)."""
    phi = np.remainder(np.asarray(phi, np.float64) + math.pi, 2 * math.pi) - math.pi
    return np.uint8(np.clip(np.round((phi + math.pi) / (2 * math.pi) * 255), 0, 255))


def png_to_phase(a):
    return np.asarray(a, np.float64) / 255 * 2 * math.pi - math.pi


def to_uint8(img):
    return np.uint8(np.clip(np.round(np.asarray(img) * 255), 0, 255))


def reconstruct(doe_png):
    """Восстановление из 8-битного ДОЭ: 4 изображения uint8 (так считает проверка)."""
    phi = torch.from_numpy(png_to_phase(doe_png)).float()[None, None]
    with torch.no_grad():
        amp = Fresnel().amplitude(phi)[0].numpy()
    return [to_uint8(a) for a in amp]
