"""M-G · 反演、校验与三态路由（规格 §3 M-G，管线核心）。

逐像素档位（性质1：投影幅值免疫）：
  blend(标定)   a<0.5 → 档1  B=(P−a·G)/(1−a)；a<0.9 → 档2  B=P−L；否则档3
  additive/screen            → 档2  B=P−L（additive）+ B=(P−L)/(1−L/255)（screen）
  unknown/未标定            → ᾶ≥0.9 → 档3；否则档2  B=P−L
饱和像素无条件档3。
校验三门（档1/档2 候选）：
  ① 值域 −2≤B≤257；② |B−B̂_ring| ≤ kv_local·σ_tex(x)；
  ③ 域内边界2px带 |∇B−∇B̂_ring| ≤ kv_local·σ_grad。
通过 → INVERTED/SUBTRACTED（conf=余量比）；不通过 → 档3 FILLED（由重建填充）。
"""
from __future__ import annotations

import cv2
import numpy as np

from deglow.core.background import ellipse
from deglow.core.grow import core_direction
from deglow.core.types import Domain, Mode, Prov, TexStats

_SUB = np.array([0.299, 0.587, 0.114], np.float32)


def _gray(a: np.ndarray) -> np.ndarray:
    return a @ _SUB


def invert_and_route(P: np.ndarray, doms: list[Domain], sig: TexStats,
                     kv: float = 2.0, strength: float = 1.0,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (out 去发光图 float32, prov uint8, conf float32)。"""
    H, W = P.shape[:2]
    out = P.copy().astype(np.float32)
    prov = np.zeros((H, W), np.uint8)          # ORIGINAL
    conf = np.ones((H, W), np.float32)
    sig_bar = max(sig.bar, 2.0)
    sigma_grad = sig_bar * 1.5

    for dom in doms:
        m = dom.mask
        if not m.any():
            continue
        B_ring = dom.B_ring
        if B_ring is None:
            continue
        D_ring = (P - B_ring).astype(np.float32)

        # ---- 方向场 ----
        if dom.mode == Mode.SCREEN and dom.u_field is not None:
            u = dom.u_field
        else:
            u_hat = dom.u_hat
            if u_hat is None:
                u_hat = core_direction(D_ring, m)
                dom.u_hat = u_hat
            u = np.broadcast_to(u_hat, (H, W, 3)).astype(np.float32)

        # ---- 逐像素档位 ----
        t = np.zeros((H, W), np.uint8)
        a = dom.alpha
        if dom.mode == Mode.BLEND and dom.calibrated and dom.G is not None \
                and a is not None:
            t[m] = np.where(a[m] < 0.5, 1, np.where(a[m] < 0.9, 2, 3))
        elif dom.mode in (Mode.ADDITIVE, Mode.SCREEN):
            t[m] = 2
        else:                                   # unknown / 未标定
            if a is None:
                a = _alpha_fallback(dom, P, D_ring)
            t[m] = np.where(a[m] >= 0.9, 3, 2)
        sat = dom.saturated
        if sat is not None:
            t[sat] = 3

        L = np.sum(D_ring * u, axis=-1)         # 投影标量（性质1 幅值免疫）
        Lv = L[..., None] * u                   # 投影矢量

        # ---- 目标背景 B ----
        B = D_ring * 0.0 + B_ring
        t12 = (t <= 2)
        if dom.mode == Mode.BLEND and dom.calibrated and dom.G is not None \
                and a is not None:
            t1 = m & (t == 1)
            t2 = m & (t == 2)
            B[t1] = (P[t1] - a[t1, None] * dom.G) / (1.0 - a[t1, None] + 1e-6)
        # 档2 统一投影减除（additive 精确 / screen 归一化处理见下）
        t2any = m & (t == 2)
        if dom.mode == Mode.SCREEN:
            den = np.clip(1.0 - Lv / 255.0, 1e-3, None)
            B[t2any] = (P[t2any] - Lv[t2any]) / den[t2any]
        else:
            B[t2any] = P[t2any] - Lv[t2any]

        # ---- 校验三门 ----
        tol_x = kv * dom.kv_local * np.maximum(sig.sigma_tex, 4.0)   # 纹理下限防平滑区误杀
        val_ok = np.all((B >= -2.0) & (B <= 257.0), axis=-1)
        dev = np.max(np.abs(B - B_ring), axis=-1)
        bgdiff = dev <= tol_x
        edge = m & (cv2.dilate(m.astype(np.uint8), ellipse(2)) > 0) \
                 & ~(cv2.erode(m.astype(np.uint8), ellipse(2)) > 0)
        grad_ok = np.ones((H, W), bool)
        if edge.any():
            gB = np.abs(np.gradient(_gray(np.clip(B, 0, 255))))
            gR = np.abs(np.gradient(_gray(np.clip(B_ring, 0, 255))))
            gd = (gB[0] + gB[1]) / 2.0 - (gR[0] + gR[1]) / 2.0
            gd = np.abs(gd)
            grad_ok[edge] = gd[edge] <= kv * dom.kv_local * sigma_grad
        pass12 = t12 & val_ok & bgdiff & grad_ok

        # ---- 落表 ----
        dom.tier = t
        prov[m] = np.where(pass12[m],
                           np.where(t[m] == 1, Prov.INVERTED, Prov.SUBTRACTED),
                           Prov.FILLED)
        # conf：校验余量比
        margin = np.maximum(1.0 - dev / (tol_x + 1e-9), 0.0)
        conf[m] = np.clip(np.where(pass12[m], margin[m], 0.0), 0.0, 1.0)
        # 应用反演（strength 拉伸力度：0=不去色，1=完全反解）
        s = (float(np.clip(strength, 0.0, 1.0)) * pass12)[..., None]
        out[m] = P[m] + s[m] * (B[m] - P[m])
    return out, prov, conf


def _alpha_fallback(dom: Domain, P: np.ndarray, D_ring: np.ndarray) -> np.ndarray:
    """α 代理兜底：投影强度按域内 p95 归一（路由代理语义）。"""
    a = np.clip(np.sum(D_ring * (dom.u_hat if dom.u_hat is not None
                                 else np.array([1.0, 0.0, 0.0], np.float32)),
                       axis=-1), 0.0, 3.0)
    p95 = np.quantile(a[dom.mask], 0.95) if dom.mask.any() else 1.0
    if p95 > 1e-3:
        a = a / p95
    return np.clip(a, 0.0, 1.5).astype(np.float32)