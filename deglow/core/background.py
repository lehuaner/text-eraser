"""M-B · 双背景场（规格 §3 M-B 的基准分工律落点）。

- B̂_global：numpy 引导滤波（He et al. 2013），r=clamp(round(2.5·ℓ_tex),3,31)，
  eps=max(σ̄_tex,2)²。仅用于方向/种子/初判，禁止进入幅值计算。
- B̂_ring  ：域内环带背景场。本实现用「guided 全局场底座 + 常数截距」对齐
  环带实测 robust 均值（规格的 Laplace 稀疏求解列为后续优化项，见落地文档
  偏差 D1；若 scipy 可用其接口已预留）。域内有效，间距平滑由反演校验兜底。
- 曲率信任：kv_local = k_v / (1 + E_curv/σ̄_tex²)，供 M-G 使用。
"""
from __future__ import annotations

import cv2
import numpy as np

from deglow.core.texture import luminance
from deglow.core.types import Domain


def ellipse(p: int) -> np.ndarray:
    """(2p+1)² 椭圆结构元。"""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p * 2 + 1, p * 2 + 1))


def _box(a: np.ndarray, r: int) -> np.ndarray:
    k = r * 2 + 1
    return cv2.boxFilter(np.asarray(a, np.float32), -1, (k, k),
                         normalize=True, borderType=cv2.BORDER_REPLICATE)


def guided_filter(P: np.ndarray, r: int = 8, eps: float = 1e-3) -> np.ndarray:
    """灰度引导的全色引导滤波（纯 numpy，不依赖 ximgproc）。

    P: HxWx3 float32；返回同形 HxWx3 平滑背景场。
    """
    g = luminance(P)
    mean_g = _box(g, r)
    var_g = np.maximum(_box(g * g, r) - mean_g * mean_g, 0.0)
    out = np.empty_like(P)
    for c in range(3):
        p = P[..., c]
        mean_p = _box(p, r)
        cov = _box(p * g, r) - mean_p * mean_g
        a = cov / (var_g + eps)
        b = mean_p - a * mean_g
        ma = _box(a, r)
        mb = _box(b, r)
        out[..., c] = ma * g + mb
    return out


def build_global(P: np.ndarray, l_tex: int, bar: float,
                 carrier: np.ndarray | None = None) -> np.ndarray:
    """§2.3 全局背景场引导滤波参数。carrier 保留用作调用约定（暂不影响滤波）。"""
    r = int(np.clip(round(2.5 * l_tex), 3, 31))
    eps = max(bar, 2.0) ** 2
    return guided_filter(P, r=r, eps=eps)


def _robust_band_mean(P: np.ndarray, band: np.ndarray) -> np.ndarray:
    """环带背景色：取亮度 ≤40% 分位像素的逐通道均值（暗侧截尾）。

    与通道法「暗背景分位数估计」同策略：发光尾迹比真实背景亮，
    只取暗侧均值可避免被亮绿/亮白尾迹拉偏（这是已在真实样例上验证
    零残留的关键数据选择，见项目记忆「背景用暗背景分位数估计」）。
    """
    idx = np.where(band)
    if len(idx[0]) < 8:
        return P.reshape(-1, 3).mean(0)
    g = luminance(P)[idx]
    thr = np.percentile(g, 40)
    keep = g <= thr
    if not keep.any():
        keep = g <= np.percentile(g, 50)
    return P[idx][keep].mean(0)


def estimate_global_bg(P: np.ndarray) -> np.ndarray:
    """全局中性暗背景色（供色晕生长参考 & 溯源）。

    取「低色度(chroma<15) ∧ 亮度≤其40%分位」像素的逐通道均值；
    无足够中性暗像素时回退到低色度像素均值。与通道法「暗背景分位数」
    同思路，且不依赖任何发光域估计。
    """
    chroma = np.max(P, axis=-1) - np.min(P, axis=-1)
    lum = luminance(P)
    neutral = chroma < 15.0
    if neutral.sum() >= 64:
        thr = np.percentile(lum[neutral], 40)
        dark = neutral & (lum <= thr)
        if dark.sum() >= 32:
            return P[dark].mean(0).astype(np.float32)
        return P[neutral].mean(0).astype(np.float32)
    return np.array([74.0, 66.0, 59.0], np.float32)


def build_ring(dom: Domain, P: np.ndarray, carrier: np.ndarray,
               sigma_g: float | None = None,
               exclude: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """环带背景场 + 曲率信任系数。

    exclude: 可选其他发光域的并集（小块域环带若与相邻域重叠，会把
    发光像素算进「背景」→ 环带污染 → 必须排除，规格 M-B 环带定义：
    dilate(mask,w) & ~mask & ~carrier 中还应排除他域）。

    返回 (B_ring 场 HxWx3 float32，kv_local)。
    """
    Bg = dom.B_global
    if Bg is None:
        Bg = P.astype(np.float32)
    w = max(int(np.ceil(1.5 * (sigma_g or dom.sigma_g))) + 2, 3)
    band = cv2.dilate(dom.mask.astype(np.uint8), ellipse(w)) > 0
    band &= ~dom.mask & ~carrier
    if exclude is not None:
        band &= ~exclude
    # 环带过稀（亮斑贴载体）时放宽一圈
    if band.sum() < 128:
        band = cv2.dilate(dom.mask.astype(np.uint8), ellipse(w + 3)) > 0
        band &= ~dom.mask & ~carrier
        if exclude is not None:
            band &= ~exclude
    if band.any():
        mb = _robust_band_mean(P, band)
        # 小块域（<256px）环带若仍高色度（环带本身落在发光包络内）→
        # 退回全局中性暗背景，避免把发光当背景
        mb_chroma = float(np.max(mb) - np.min(mb))
        if dom.mask.sum() < 256 and mb_chroma >= 12.0:
            mb = estimate_global_bg(P)
        # 锚定到「域内」：offset 使 B̂_ring 在域内的均值 = 环带实测均值，
        # （大而平滑的光晕中心需要真实背景 ≈ 环带背景；域内字段保持空间变化）
        off = mb - np.mean(Bg[dom.mask], axis=0) if dom.mask.any() else 0.0
        B_ring = Bg + off
        # FILLED 重建用常数环带背景色（对标通道法「整区拉平到一个背景色」，
        # 避免引导场在光晕中心残留的绿/亮色漂移到重建区）
        dom.ring_fill = np.broadcast_to(
            mb.astype(np.float32)[None, None, :], dom.mask.shape + (3,)).copy()
    else:
        B_ring = Bg
    # 曲率信任：环带亮度分散度 → E_curv / σ̄² 用域内 bar 归一
    if band.any():
        g = luminance(P)[band]
        e_curv = float(np.var(g)) / max(dom.report.get("sig_bar", 1.0) ** 2, 1e-6)
    else:
        e_curv = 0.0
    kv_local = 1.0 / (1.0 + e_curv)
    dom.B_ring = B_ring
    dom.kv_local = kv_local
    return B_ring, kv_local