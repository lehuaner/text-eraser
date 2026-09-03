"""M-C · 双偏差场、频段分离与三通道种子（规格 §3 M-C）。

偏差场相对 **全局背景场** B̂_global：
  D_high = Dg − gauss(Dg, σ̄_tex)   → 高频（文本化纹理/噪声）
  D_low  = gauss(Dg, σ̄_tex)         → 低频（大面积光晕/光照缓变）
载体掩码 = 检测器掩码 ∪ (|D_high|>k·σ_tex 闭运算后、面积≥9ℓ²的连通块)。
种子     = 载体外沿带 dilate(carrier,3)&~carrier ∪ 低频显著孤立光斑
           （|D_low|>k_lf·σ_tex、面积≥max(9ℓ²,50)、solidity(矩形度)≥0.7）。
"""
from __future__ import annotations

import cv2
import numpy as np

from deglow.core.background import ellipse
from deglow.core.types import TexStats

_K3 = np.ones((3, 3), np.uint8)


def build_fields(P: np.ndarray, Bg: np.ndarray, sig: TexStats
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (Dg, D_high, D_low)。"""
    Dg = (P - Bg).astype(np.float32)
    ksize = max(int(round(6 * sig.bar)) | 1, 1)
    D_low = cv2.GaussianBlur(Dg, (ksize, ksize), sigmaX=sig.bar,
                             borderType=cv2.BORDER_REPLICATE)
    D_high = Dg - D_low
    return Dg, D_high, D_low


def _big_blobs(bin_mask: np.ndarray, min_area: int) -> np.ndarray:
    """取面积 ≥ min_area 的连通块（标量 uint8 0/1）。"""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        bin_mask.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(lab, bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep |= lab == i
    return np.asarray(keep, bool)


def carrier_union(tmask: np.ndarray, D_high: np.ndarray, sig: TexStats,
                  k: float = 3.0) -> np.ndarray:
    """载体掩码：检测器蒙版 ∪ 高频显著连通块（文字笔画 + 微纹理载体）。"""
    hi = np.max(np.abs(D_high), axis=-1) > k * sig.sigma_tex
    hi = cv2.morphologyEx(hi.astype(np.uint8), cv2.MORPH_CLOSE, _K3) > 0
    min_a = max(9 * sig.l_tex * sig.l_tex, 1)
    carrier = (tmask > 0) | _big_blobs(hi, min_a) if tmask is not None and tmask.any() \
        else _big_blobs(hi, min_a)
    return carrier


def collect_seeds(carrier: np.ndarray, D_low: np.ndarray, sig: TexStats,
                  k_lf: float = 4.0) -> np.ndarray:
    """种子 = 载体外沿带 ∪ 低频显著孤立光斑。"""
    apron = cv2.dilate(carrier.astype(np.uint8), ellipse(3)) > 0
    apron &= ~carrier
    low = np.max(np.abs(D_low), axis=-1) > k_lf * sig.sigma_tex
    low = cv2.morphologyEx(low.astype(np.uint8), cv2.MORPH_CLOSE, _K3)
    min_a = max(9 * sig.l_tex * sig.l_tex, 50)
    blobs = _big_blobs(low, min_a)
    # 矩形度(solidity 近似)：面积/bbox 面积 ≥ 0.7 → 孤立紧凑光斑
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        blobs.astype(np.uint8), connectivity=8)
    solid = np.zeros_like(lab, bool)
    for i in range(1, n):
        ar = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]; h = stats[i, cv2.CC_STAT_HEIGHT]
        if w > 0 and h > 0 and ar / (w * h) >= 0.7:
            solid |= lab == i
    return (apron | solid)