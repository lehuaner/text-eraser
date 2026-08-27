"""M-D · 两轮生长 + 饱和纳入 + 域后处理（规格 §3 M-D）。

生长条件（第 1 轮宽松 τ=0.5 / 第 2 轮收紧 τ=0.7）：
  cos(Dg[x], u_dir(x)) > τ  且  |Dg[x]| > k·σ_tex(x)
第 2 轮 screen 模式用逐像素方向场 u_dir(x)=normalize(Dg(x))（G=None 退化为方向场），
blend/additive 用域常向量 u_hat。
饱和：任通道 ≥253（8-bit 距上限 ≤2 LSB）且八邻接域内 → 强制纳入并标 saturated；
该类像素不参与方向统计/拟合/α 估计，路由固定重建。
"""
from __future__ import annotations

import cv2
import numpy as np

from deglow.core.background import ellipse
from deglow.core.types import Domain, Mode, TexStats

_K3 = np.ones((3, 3), np.uint8)


def flow_dir(Dg: np.ndarray) -> np.ndarray:
    """逐像素单位方向场 u(x)=normalize(Dg(x))（screen 用；平坦处 0）。"""
    mag = np.sqrt(np.sum(Dg * Dg, axis=-1))[..., None]
    safe = np.maximum(mag, 1e-6)
    return Dg / safe


def core_direction(Dg: np.ndarray, region: np.ndarray,
                   top_frac: float = 0.25) -> np.ndarray:
    """种子区 |Dg| 前 top_frac 强的加权均值方向（规格 M-D u0）。"""
    mag = np.sqrt(np.sum(Dg * Dg, axis=-1))
    vals_mag = mag[region]
    if not vals_mag.size:
        return np.array([1.0, 0.0, 0.0], np.float32)
    t = np.quantile(vals_mag, 1.0 - top_frac)
    top = region & (mag >= t)
    w = np.maximum(mag[top], 1e-6)
    u = np.sum(Dg[top] * w[..., None], axis=0)
    n = np.linalg.norm(u)
    if n < 1e-6:
        # 归一化失败兜底：取区域均值方向（仍可辨识，性质2）
        u = Dg[top].mean(0)
        n = np.linalg.norm(u)
    return (u / (n + 1e-9)).astype(np.float32)


def _aligned(Dg: np.ndarray, udir: np.ndarray, sig: TexStats,
             tau: float, k: float) -> np.ndarray:
    dot = np.sum(Dg * udir, axis=-1)
    mag = np.sqrt(np.maximum(np.sum(Dg * Dg, axis=-1), 0.0))
    cos = dot / np.maximum(mag * (np.linalg.norm(udir) + 1e-9), 1e-9)
    return (cos > tau) & (mag > k * sig.sigma_tex)


def grow_round(Dg: np.ndarray, seeds: np.ndarray, udir: np.ndarray,
               sig: TexStats, tau: float = 0.5, k: float = 3.0,
               max_iter: int = 400) -> np.ndarray:
    """BFS 生长（迭代膨胀 + 条件纳入），返回生长后掩码。"""
    cond = _aligned(Dg, udir, sig, tau, k)
    zone = (seeds > 0).copy()
    cur = zone
    for _ in range(max_iter):
        dil = cv2.dilate(cur.astype(np.uint8), _K3) > 0
        add = dil & cond & ~zone
        if not add.any():
            break
        zone |= add
        cur = zone
    return zone


def grow_round2(Dg: np.ndarray, doms: list[Domain], sig: TexStats,
                tau: float = 0.7, k: float = 3.0,
                max_iter: int = 300) -> list[np.ndarray]:
    """模式自适应重生长：blend/additive 用 u_hat，screen 用方向场。"""
    outs = []
    for d in doms:
        if d.mode == Mode.SCREEN:
            udir = flow_dir(Dg)
            outs.append(grow_round(Dg, d.mask, udir, sig, tau, k, max_iter))
        else:
            u = d.u_hat if d.u_hat is not None else core_direction(Dg, d.mask)
            outs.append(grow_round(Dg, d.mask, u, sig, tau, k, max_iter))
    return outs


def grow_veil(P: np.ndarray, mask: np.ndarray, carrier: np.ndarray,
              u: np.ndarray, global_bg: np.ndarray, tau: float = 0.6,
              chroma_min: float = 20.0, max_iter: int = 400,
              max_area: float = 0.25) -> np.ndarray:
    """色晕连通生长（通用版通道法 zone 生长，任意色不限于绿）。

    条件：① 色度 ≥ chroma_min；② (P − 全局暗背景)·û / |·| > tau；
      ③ 亮度不低于暗背景太多。纯亮度/灰阶纹理被 ① 豁免（规格 R8），
      不依赖对平滑大光晕的偏差场（引导滤波会把大光晕当背景 → D 偏小，
      这正是 v4 首轮生长欠覆盖的直接原因，见落地文档偏差 D3）。
    从已检出的发光域连通生长，排除载体（文字笔画）；面积上限防吞图。
    """
    H, W = P.shape[:2]
    bg = global_bg.astype(np.float32)
    u = np.asarray(u, np.float32)
    un = np.linalg.norm(u)
    if un < 1e-6:
        return mask
    off = P.astype(np.float32) - bg[None, None, :]
    chroma = np.max(P, axis=-1) - np.min(P, axis=-1)
    dot = off @ u
    mag = np.sqrt(np.clip(np.sum(off * off, axis=-1), 0, None))
    cos = dot / (mag * un + 1e-9)
    lum_bg = float(bg @ np.array([0.299, 0.587, 0.114], np.float32))
    lum = P.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    cond = (chroma >= chroma_min) & (cos > tau) & (lum >= lum_bg)
    zone = (mask > 0).copy()
    cur = zone
    budget = int(H * W * max_area)
    for _ in range(max_iter):
        dil = cv2.dilate(cur.astype(np.uint8), _K3) > 0
        add = dil & cond & ~zone
        if not add.any():
            break
        zone |= add
        if int(zone.sum()) > budget:
            zone &= ~(zone & ~mask & add)          # 超限回退最后一步
            break
        cur = zone
    zone &= ~carrier                               # 永远不碰文字笔画
    return zone


def saturation_mask(P: np.ndarray) -> np.ndarray:
    """任通道 ≥ 253（距 8-bit 上限 ≤ 2 LSB）。"""
    return np.max(P, axis=-1) >= 253.0


def split_domains(mask: np.ndarray, carrier: np.ndarray, sig: TexStats,
                  ) -> list[np.ndarray]:
    """生长区 → 连通块 → 丢弃小域（面积<max(9ℓ²,50)）。"""
    min_a = max(int(9 * sig.l_tex * sig.l_tex), 50)
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    outs = []
    for i in range(1, n):
        comp = lab == i
        if comp.sum() < min_a:
            continue
        outs.append(comp)
    return outs


def merge_like(doms: list[Domain], cos_thr: float = 0.9) -> list[Domain]:
    """两域膨胀相交且方向 cos > 0.9 → 合并重拟合；交叠像素归面积大者。"""
    n = len(doms)
    if n <= 1:
        return doms
    # 方向 cos 表
    cos = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a = doms[i].u_hat; b = doms[j].u_hat
            if a is not None and b is not None:
                cos[i, j] = cos[j, i] = float(np.clip(a @ b, -1, 1))
    # union-find 组
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for i in range(n):
        for j in range(i + 1, n):
            if cos[i, j] > cos_thr and (cv2.dilate(doms[i].mask.astype(np.uint8), _K3)
                                        & doms[j].mask).any():
                union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    merged: list[Domain] = []
    for _, idxs in groups.items():
        if len(idxs) == 1:
            merged.append(doms[idxs[0]])
            continue
        base = doms[idxs[0]]
        mask = base.mask.copy()
        for j in idxs[1:]:
            mask |= doms[j].mask
        # 合并到新域（元数据以面积最大者为准，方向/拟合重算由 fit 兜底）
        new_dom = Domain(id=base.id, mask=mask, carrier_mask=base.carrier_mask,
                         B_global=base.B_global, mode=Mode.UNKNOWN,
                         sigma_g=float(np.mean([doms[j].sigma_g for j in idxs])))
        merged.append(new_dom)
    return merged