"""M-F · α 场、距离衰减与染色判别（规格 §3 M-F）。

- blend 域：α_meas = clip((P−B̂_ring)·û/(G−B̂_ring) 投影点除, 0, 1.5)，中值3→高斯1.5σ̂_g 平滑。
- additive/screen/unknown 域：无标定标度，α 仅为路由代理（α̃ 用 p95）；
  仍做距离衰减拟合以估计 σ̂_g（驱动环带宽度/曲率信任）。
- σ̂_g：在 d>2px 样本上拟合 ln α ≈ ln A − d²/(2σ̂_g²)（单高斯；双高斯 BIC 择优留待后续里程碑）。
- 染色判别：笔画边缘带 m=median(D·û)，m>0.3·α(0⁻)·|u| → through，m<0.1·α(0⁻)·|u| → behind，中间冲突回退 behind。
"""
from __future__ import annotations

import cv2
import numpy as np

from deglow.core.grow import core_direction, flow_dir
from deglow.core.texture import gauss_blur
from deglow.core.types import Domain, Dye, Mode

_K3 = np.ones((3, 3), np.uint8)


def _alpha_proxy(dom: Domain, P: np.ndarray, B_ring: np.ndarray) -> np.ndarray:
    """域内 α 场（blend 实测；其余 = 投影强度代理，仅作路由/拟合用）。"""
    D_ring = (P - B_ring).astype(np.float32)
    if dom.mode == Mode.BLEND and dom.G is not None and dom.calibrated:
        u_vec = dom.G - B_ring
        den = np.sum(u_vec * u_vec, axis=-1) + 1e-9
        num = np.sum(D_ring * u_vec, axis=-1)
        a = np.clip(num / den, 0.0, 1.5)
    else:
        u = dom.u_hat
        if u is None:                      # 新拆域方向未置的防御
            u = core_direction(D_ring, dom.mask)
            dom.u_hat = u
        a = np.clip(np.sum(D_ring * u, axis=-1), 0.0, 3.0)
        # 无标度归一：以域内 p95 为 1 量级（路由代理语义）
        p95 = np.quantile(a[dom.mask], 0.95) if dom.mask.any() else 1.0
        if p95 > 1e-3:
            a = a / p95
        a = np.clip(a, 0.0, 1.5)
    return a.astype(np.float32)


def _signed_distance(carrier: np.ndarray) -> np.ndarray:
    """发光像素到载体边界的距离（外向为正；载体内为 0）。"""
    f = (~carrier).astype(np.uint8)
    return cv2.distanceTransform(f, cv2.DIST_L2, 5).astype(np.float32)


def fit_sigma_g(dom: Domain, alpha: np.ndarray, distance: np.ndarray,
                ) -> float:
    """单高斯衰减拟合：样本点 ln α ≈ lnA − d²/2σ² → σ_g（Huber 化去离群）。"""
    use = dom.mask & (distance > 2.0)
    if dom.saturated is not None:
        use &= ~dom.saturated
    a = alpha[use]; d = distance[use]
    if a.size < 32 or float(np.median(a)) < 0.02:
        return 3.0
    ok = (a > 0.01)
    a, d = a[ok], d[ok]
    if a.size < 32:
        return 3.0
    y = np.log(np.clip(a, 1e-3, None))
    x = d * d
    # 去上下 5% 离群后最小二乘
    lo, hi = np.percentile(y, [5, 95])
    keep = (y >= lo) & (y <= hi)
    if keep.sum() < 32:
        return 3.0
    A = np.vstack([np.ones(keep.sum()), x[keep]]).T
    coef, *_ = np.linalg.lstsq(A, y[keep], rcond=None)
    slope = float(coef[1])
    if slope >= -1e-4:            # 单调不减 → 拟合失败
        return 3.0
    sg = float(np.sqrt(np.clip(-0.5 / slope, 0.5, 40.0)))
    return sg


def build_alpha(dom: Domain, P: np.ndarray, Dg: np.ndarray,
                carrier: np.ndarray) -> Domain:
    """原地回填 dom.alpha / sigma_g / dye / u_field(grow 已置 mode)。"""
    B_ring = dom.B_ring
    if B_ring is None:
        dom.alpha = None
        return dom
    a = _alpha_proxy(dom, P, B_ring)
    dist = _signed_distance(carrier)
    sg = fit_sigma_g(dom, a, dist)
    dom.sigma_g = float(sg)
    a_s = a.copy()
    a_s[dom.mask] = gauss_blur(a_s, max(sg * 1.5, 0.5))[dom.mask]
    a_s[dom.mask] = np.clip(a_s[dom.mask], 0.0, 1.5)
    dom.alpha = a_s.astype(np.float32)

    if dom.mode == Mode.SCREEN:
        dom.u_field = flow_dir(Dg).astype(np.float32)
    dom.dye = _judge_dye(dom, P, B_ring, a)
    return dom


def _judge_dye(dom: Domain, P: np.ndarray, B_ring: np.ndarray,
               alpha: np.ndarray) -> str:
    """笔画边缘带（距载体 ≤3px）的染色判别。"""
    edge = dom.mask & (cv2.dilate(dom.carrier_mask.astype(np.uint8), _K3) > 0)
    edge &= ~dom.carrier_mask
    if dom.saturated is not None:
        edge &= ~dom.saturated
    if edge.sum() < 16 or dom.u_hat is None:
        return Dye.BEHIND
    D_ring = (P - B_ring).astype(np.float32)
    m = float(np.median(np.sum(D_ring[edge] * dom.u_hat, axis=-1)))
    a0 = float(np.median(alpha[edge])) if edge.sum() else 0.0
    if m > 0.3 * (a0 + 1e-3):
        return Dye.THROUGH
    if m < 0.1 * (a0 + 1e-3):
        return Dye.BEHIND
    # 冲突：始终优先 behind（字画在发光上 → 不反解笔画，只清发光带）
    return Dye.BEHIND