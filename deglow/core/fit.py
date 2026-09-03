"""M-E · 模式拟合：性质 2/3 的实现（规格 §3 M-E）。

按 B̂_ring 亮度分箱（tol=4 级、窗内下限 64px、核取窗内 |D| top10% 且 ≥8px），
三假设评估：
  additive  r_add = 各窗核均值与窗 B̂ 均值的差 std（等偏移期望→小）
  screen    r_scr = 逐通道 (C̄w−B̄w)/(1−B̄w/255) 残差 std（screen 期望→小）
  blend     rho=<ΔB̂,ΔC>/<ΔB̂,ΔB̂>；r_bld = rms(ΔC − rho·ΔB̂)（线性期望→小）
取残差最小者；次优/最优 < 1.5 → unknown（→ 档2 减除仍正确，性质2 方向可辨识）。
blend 需 0<α(=1−rho)<1 才算标定成功；窗数 <3 或 B̂ 无色差 → 无条件 unknown。
"""
from __future__ import annotations

import numpy as np

from deglow.core.background import build_ring
from deglow.core.grow import core_direction
from deglow.core.types import Domain, Mode

_TOL = 4.0       # 分箱容差（亮度级）
_MIN_WIN = 64    # 最小窗像素
_MARGIN = 1.5    # 次优/最优门
_MIN_BINS = 3
_BIN_SPREAD = 6.0
_ALPHA_LO, _ALPHA_HI = 0.05, 0.95


def _lum_B(B_ring: np.ndarray) -> np.ndarray:
    return B_ring @ np.array([0.299, 0.587, 0.114], np.float32)


def _split_windows(dom: Domain, B_ring: np.ndarray
                   ) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """按 B̂_ring 亮度量化分箱（跳过饱和与载体像素）。

    返回 [(win_mask, level), ...]；窗不足或色差不足 → None。
    """
    use = dom.mask.copy()
    if dom.saturated is not None:
        use &= ~dom.saturated
    use &= ~dom.carrier_mask
    if not use.any():
        return None
    lum = _lum_B(B_ring)
    levels = np.quantile(lum[use], np.linspace(0, 1, 9))[1:-1]
    wins: list[tuple[np.ndarray, float]] = []
    for lv in levels:
        px = use & (np.abs(lum - lv) <= _TOL)
        if px.sum() >= _MIN_WIN:
            wins.append((px, float(lv)))
    if len(wins) < _MIN_BINS:
        return None
    lv_min = min(w for _, w in wins); lv_max = max(w for _, w in wins)
    if lv_max - lv_min < _BIN_SPREAD:
        return None
    return wins


def _window_core(px: np.ndarray, D_ring: np.ndarray) -> np.ndarray:
    """窗内 |D| top10% 核均值（下限 8px）。"""
    vals = D_ring[px]
    mag = np.sqrt(np.sum(vals * vals, axis=-1))
    if mag.size < 8:
        order = np.argsort(mag)[-8:] if mag.size else np.array([], int)
        sel = np.zeros(vals.shape[0], bool)
        sel[order] = True
    else:
        t = np.quantile(mag, 0.9)
        sel = mag >= t
    return vals[sel].mean(0)


def fit_mode(dom: Domain, P: np.ndarray, Dg: np.ndarray,
             build_ring_fn=build_ring) -> Domain:
    """对域做模式拟合，原地回填 mode/u_hat/G/alpha_max/calibrated。"""
    B_ring = dom.B_ring
    if B_ring is None:
        B_ring, _ = build_ring_fn(dom, P, dom.carrier_mask)
    Dg_dom = (P - B_ring).astype(np.float32)

    wins = _split_windows(dom, B_ring)
    if wins is None:
        dom.mode = Mode.UNKNOWN
        dom.u_hat = core_direction(Dg_dom, dom.mask)
        return dom

    Cs: list[np.ndarray] = []
    Bs: list[np.ndarray] = []
    for px, _ in wins:
        Cs.append(_window_core(px, Dg_dom))
        Bs.append(B_ring[px].mean(0))
    Cs = np.asarray(Cs, np.float32)
    Bs = np.asarray(Bs, np.float32)

    # additive：等偏移残差
    r_add = float(np.std(np.linalg.norm(Cs - Bs, axis=-1)))
    # screen：逐通道归一残差
    den = np.maximum(1.0 - Bs / 255.0, 1e-3)
    r_scr = float(np.std(np.linalg.norm((Cs - Bs) / den, axis=-1)))
    # blend：线性回归残差
    dC = Cs - Cs.mean(0)
    dB = Bs - Bs.mean(0)
    den2 = float(np.sum(dB * dB)) + 1e-9
    rho = float(np.sum(dB * dC) / den2)
    r_bld = float(np.sqrt(np.mean(np.sum((dC - rho * dB) ** 2, axis=-1))))

    res = {"additive": r_add, "screen": r_scr, "blend": r_bld}
    order = sorted(res.items(), key=lambda kv: kv[1])
    best, second = order[0], order[1]
    mode = best[0]
    if second[1] / (best[1] + 1e-9) < _MARGIN:
        mode = Mode.UNKNOWN

    C_hat = np.mean(Cs, axis=0)
    B_hat = np.mean(Bs, axis=0)
    nrm = np.linalg.norm(C_hat)
    u_hat = (C_hat / (nrm + 1e-9)).astype(np.float32)

    if mode == Mode.BLEND:
        alpha = 1.0 - rho
        if _ALPHA_LO < alpha < _ALPHA_HI:
            dom.mode = Mode.BLEND
            dom.u_hat = u_hat
            dom.alpha_max = float(alpha)
            G = np.clip((C_hat - (1.0 - alpha) * B_hat) / alpha, 0, 255)
            dom.G = G.astype(np.float32)
            dom.calibrated = True
        else:
            dom.mode = Mode.UNKNOWN
            dom.u_hat = u_hat
    elif mode == Mode.ADDITIVE:
        dom.mode = Mode.ADDITIVE
        dom.u_hat = u_hat
        dom.G = None
    else:  # screen
        dom.mode = Mode.SCREEN
        dom.u_hat = None            # 方向场在 alpha/invert 阶段由 Dg 提供
        dom.G = None
    return dom