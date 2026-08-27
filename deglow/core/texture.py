"""M-A · 纹理实测：一切阈值的度量单位来源（规格 §3 M-A）。

- σ_tex(x)：逐像素 robust 高通噪声水平（1.4826 · 局部 MAD）
- ℓ_tex   ：高通自相关第一次降到 1/e 的滞后（clamp [2,12]）
- σ̄_tex   ：σ_tex 图均值（平坦区下限 clamp 1.0）
"""
from __future__ import annotations

import cv2
import numpy as np

from deglow.core.types import TexStats

_LUM = np.array([0.299, 0.587, 0.114], np.float32)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """HxWx3 float32 → HxW float32 亮度。"""
    return rgb @ _LUM


def _med(arr: np.ndarray, k: int) -> np.ndarray:
    """中值滤波（float32 数组；cv2.medianBlur 兜底整数量化回退）。"""
    k = int(k) | 1
    a = arr.astype(np.float32)
    try:
        return cv2.medianBlur(a, k)
    except cv2.error:
        lo = float(a.min()); hi = float(a.max())
        span = max(hi - lo, 1e-6)
        b = np.clip((a - lo) / span * 255.0, 0, 255).astype(np.uint8)
        out = cv2.medianBlur(b, k).astype(np.float32)
        return lo + (out / 255.0) * span


def _autocorr_lag(hp: np.ndarray, target: float = 1.0 / np.e,
                  max_lag: int = 12) -> int:
    """高通图归一化空间自相关首次降到 target 的滞后（行列平均）。

    在 ≤512² 的降采样网格上估计，O(max_lag·N) 可接受。
    """
    ih = max(1, int(min(hp.shape[0], 512)))
    iw = max(1, int(min(hp.shape[1], 512)))
    if (hp.shape[0], hp.shape[1]) != (ih, iw):
        x = cv2.resize(hp, (iw, ih), interpolation=cv2.INTER_AREA)
    else:
        x = hp
    var = float(np.var(x))
    if var < 1e-6:
        return max_lag  # 平坦区：无纹理相关长度 → 取上限
    for k in range(1, max_lag + 1):
        a, b = x[:, :-k], x[:, k:]
        c = float(np.mean((a - a.mean()) * (b - b.mean()))) / var
        a2, b2 = x[:-k, :], x[k:, :]
        c = 0.5 * (c + float(np.mean((a2 - a2.mean()) * (b2 - b2.mean()))) / var)
        if c <= target:
            return k
    return max_lag


def estimate_texture(P: np.ndarray) -> TexStats:
    """规格 M-A：输入 HxWx3 float32 观测图，输出纹理统计。

    HP = P − medianBlur(P,5)（按亮度场）；σ_tex = 1.4826·MAD32(HP)，
    平坦区下限 clamp 1.0，再用 32×32 中值平滑一次防零方差。
    """
    gray = luminance(P)
    gray_u = np.clip(gray, 0, 255).astype(np.uint8)
    med = _med(gray_u.astype(np.float32), 5)
    hp = gray - med

    mad = _med(np.abs(hp), 32)
    sigma_tex = np.clip(1.4826 * mad, 1.0, None)
    sigma_tex = _med(sigma_tex, 32)
    sigma_tex = np.maximum(sigma_tex, 1.0)

    bar = float(np.maximum(sigma_tex.mean(), 1.0))
    l_tex = _autocorr_lag(hp, max_lag=12)
    return TexStats(sigma_tex=sigma_tex.astype(np.float32), l_tex=l_tex, bar=bar)


def gauss_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """按 σ 模糊 float32 数组（ksize 自适应）。"""
    s = max(float(sigma), 0.01)
    ksize = max(int(round(6 * s)) | 1, 1)
    return cv2.GaussianBlur(a.astype(np.float32), (ksize, ksize),
                            sigmaX=s, borderType=cv2.BORDER_REPLICATE)