"""Метрики со слайда 3 (КК Пирсона, ИСС) и функция потерь для обучения."""
import numpy as np
import torch


def _norm(a):
    a = np.asarray(a, np.float64)
    return a / a.max() if a.max() > 0 else a


def pearson(a, b):
    a = _norm(a).ravel()
    b = _norm(b).ravel()
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def ssim(a, b, c1=0.01 ** 2, c2=0.03 ** 2):
    """ИСС по формуле слайда: средние, дисперсии и ковариация по всему изображению.
    Оба изображения перед сравнением нормируются на максимум (= 1)."""
    a, b = _norm(a), _norm(b)
    ma, mb = a.mean(), b.mean()
    cov = ((a - ma) * (b - mb)).mean()
    return float((2 * ma * mb + c1) * (2 * cov + c2) / ((ma ** 2 + mb ** 2 + c1) * (a.var() + b.var() + c2)))


def ssim_windowed(a, b, c1=0.01 ** 2, c2=0.03 ** 2):
    """Справочно: SSIM Wang 2004 с гауссовым окном 11x11, sigma 1.5."""
    from numpy.lib.stride_tricks import sliding_window_view
    a, b = _norm(a), _norm(b)
    g = np.exp(-(np.arange(11) - 5) ** 2 / 4.5)
    g /= g.sum()
    w = np.outer(g, g)
    f = lambda x: (sliding_window_view(x, (11, 11)) * w).sum(axis=(-2, -1))
    ma, mb = f(a), f(b)
    va, vb, cov = f(a * a) - ma ** 2, f(b * b) - mb ** 2, f(a * b) - ma * mb
    return float(((2 * ma * mb + c1) * (2 * cov + c2) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))).mean())


# torch-версии для функции ошибки, x и y: (B, Z, H, W) в [0, 1]

def pearson_t(x, y):
    x = x - x.mean(dim=(-2, -1), keepdim=True)
    y = y - y.mean(dim=(-2, -1), keepdim=True)
    return (x * y).sum(dim=(-2, -1)) / torch.sqrt((x * x).sum(dim=(-2, -1)) * (y * y).sum(dim=(-2, -1)) + 1e-8)


def ssim_t(x, y, c1=0.01 ** 2, c2=0.03 ** 2):
    mx, my = x.mean(dim=(-2, -1)), y.mean(dim=(-2, -1))
    vx, vy = x.var(dim=(-2, -1), unbiased=False), y.var(dim=(-2, -1), unbiased=False)
    cov = ((x - mx[..., None, None]) * (y - my[..., None, None])).mean(dim=(-2, -1))
    return (2 * mx * my + c1) * (2 * cov + c2) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))


def ssim_windowed_t(x, y, c1=0.01 ** 2, c2=0.03 ** 2):
    """Оконный SSIM для функции ошибки: гауссово окно (sigma ~1.4) приближено тремя проходами
    усреднения 3x3, так быстрее свёртки 11x11."""
    b, z, h, w = x.shape
    maps = torch.stack([x, y, x * x, y * y, x * y], 2).reshape(b * z * 5, 1, h, w)
    for _ in range(3):
        maps = torch.nn.functional.avg_pool2d(maps, 3, stride=1)
    maps = maps.reshape(b, z, 5, h - 6, w - 6)
    mx, my = maps[:, :, 0], maps[:, :, 1]
    vx, vy, cov = maps[:, :, 2] - mx ** 2, maps[:, :, 3] - my ** 2, maps[:, :, 4] - mx * my
    s = (2 * mx * my + c1) * (2 * cov + c2) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))
    return s.mean(dim=(-2, -1))


def doe_loss(rec, target, w_pearson=1.0, w_ssim=1.0, w_window=0.0, balance=0.0):
    """Ошибка по плоскостям: w_pearson*(1 - КК) + w_ssim*(1 - ИСС).
    balance > 0 добавляет штраф худшей плоскости (используется при обучении)."""
    loss = w_pearson * (1 - pearson_t(rec, target)) + w_ssim * (1 - ssim_t(rec, target))
    if w_window:
        loss = loss + w_window * (1 - ssim_windowed_t(rec, target))
    if balance:
        weights = torch.softmax(loss.detach() / 0.05, dim=-1)
        loss = loss + balance * loss.shape[-1] * weights * loss
    return loss.mean()
