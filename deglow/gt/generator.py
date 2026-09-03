"""M0 · 合成 GT 生成器（规格 §5）。

compose() 按解析式合成：
  d        = signed_distance(M_text)        # 笔画内为负、笔画外为正
  α        = A · Φ(−d / σ_g)                # 距笔画越远光晕越弱
  through  : 先成字 U → 再罩发光（blend/additive/screen）
  behind   : 先罩发光 L → 再写字

轴覆盖（规格 §5）：
  发光色 HSL 全环(步进15°) × 三模式 × through/behind × 六类背景
  × σ_g ∈ [4,40] × 文字色(含同色向极端例) × 载体(文字/几何/孤立域)
  × 饱和诱导(深色 G × additive × A≥0.9) × 源稀缺(发光≥半幅) × 灰度退化轴
  + 重点交叉(纹理×无色差×blend、screen×大色差) 各 200 张。

生成器自检：合成结果与**独立解析式**（Python float 逐像素手算）抽样比对，
误差 < 1e-3（M0 验收门）。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.stats import norm as _norm

# ---------------------------------------------------------------------------
# 基础合成原语
# ---------------------------------------------------------------------------

def signed_distance(mask01: np.ndarray) -> np.ndarray:
    """二值载体 → 有符号距离场：笔画内为负、笔画外为正、边界≈0。"""
    fg = (mask01 > 0).astype(np.uint8)
    bg = (1 - fg)
    d_in = cv2.distanceTransform(fg, cv2.DIST_L2, 5).astype(np.float32)
    d_out = cv2.distanceTransform(bg, cv2.DIST_L2, 5).astype(np.float32)
    return (d_out - d_in).astype(np.float32)


def aa_coverage(d: np.ndarray, ramp: float = 2.0) -> np.ndarray:
    """笔画 AA 覆盖率：d∈[−2,2] 过渡带（0=纯背景 1=纯笔画）。"""
    return np.clip(0.5 - d / ramp, 0, 1).astype(np.float32)


def compose(B: np.ndarray, M_text: np.ndarray, T, G, sigma_g: float,
            A: float = 0.6, mode: str = "blend", dye: str = "through",
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                       np.ndarray]:
    """规格 §5 compose()。

    参数：
      B      HxWx3 float32 真值背景
      M_text HxW bool 载体（文字/几何）笔画
      T, G   文字色 / 发光色 (3,)
      sigma_g, A  光晕半径与峰值强度
      mode   'blend'|'additive'|'screen'
      dye    'through'|'behind'
    返回：
      P        观测图 HxWx3 float32
      U        无发光参考（字层）HxWx3 float32
      M_aa     笔画 AA 覆盖率 HxW float32
      alpha    α 场 HxW float32
      glow_truth 发光域真值 bool（|P−U|>1，光晕真实改变了像素）
    """
    d = signed_distance(M_text)
    alpha = (A * _norm.cdf(-d / max(float(sigma_g), 1e-3))).astype(np.float32)
    M_aa = aa_coverage(d)
    Gv = np.asarray(G, np.float32)[None, None, :]
    Tv = np.asarray(T, np.float32)[None, None, :]
    Bf = np.asarray(B, np.float32)
    if Bf.shape != d.shape + (3,):
        Bf = np.broadcast_to(Bf, d.shape + (3,)).copy()
    Bv = Bf

    # 字层（含 AA）：through/behind 都以此为最终目标
    U = Bv * (1.0 - M_aa[..., None]) + Tv * M_aa[..., None]
    a3 = alpha[..., None]
    if dye == "through":
        if mode == "blend":
            P = (1.0 - a3) * U + a3 * Gv
        elif mode == "additive":
            P = np.clip(U + a3 * Gv, 0, 255)
        else:                                   # screen
            P = U + a3 * Gv * (1.0 - U / 255.0)
    else:                                       # behind：字画在发光上
        if mode == "blend":
            L = (1.0 - a3) * Bv + a3 * Gv
        elif mode == "additive":
            L = np.clip(Bv + a3 * Gv, 0, 255)
        else:
            L = Bv + a3 * Gv * (1.0 - Bv / 255.0)
        P = L * (1.0 - M_aa[..., None]) + Tv * M_aa[..., None]
    glow_truth = (np.max(np.abs(P - U), axis=-1) > 1.0)
    return (np.clip(P, 0, 255).astype(np.float32), U.astype(np.float32),
            M_aa, alpha, glow_truth)


# ---------------------------------------------------------------------------
# 轴：背景 / 载体 / 颜色
# ---------------------------------------------------------------------------

BG_NAMES = ("gray", "warm", "green", "bright", "dark", "texture")


def make_background(kind: str, H: int, W: int, rng: np.random.Generator,
                    sigma_tex: float | None = None) -> np.ndarray:
    """六类背景（灰阶/暖/绿/亮/暗/纹理）。纹理类 σ_tex∈[10,40]。"""
    if kind == "gray":
        base = np.full((H, W, 3), 128.0)
    elif kind == "warm":
        base = np.full((H, W, 3), [236.0, 198.0, 158.0])
    elif kind == "green":
        base = np.full((H, W, 3), [58.0, 138.0, 92.0])
    elif kind == "bright":
        base = np.full((H, W, 3), [232.0, 233.0, 240.0])
    elif kind == "dark":
        base = np.full((H, W, 3), [42.0, 46.0, 52.0])
    elif kind == "texture":
        st = sigma_tex if sigma_tex is not None else float(rng.uniform(10, 40))
        noise = rng.normal(0, st, (H, W, 3))
        tex = cv2.GaussianBlur(noise, (0, 0), sigmaX=2.0)   # 表面纹理（低频）
        base = np.full((H, W, 3), 128.0) + tex
    else:
        raise ValueError(kind)
    # 轻微横向渐变，避免完全平板（更贴近真实，也给引导滤波留台阶）
    ramp = np.linspace(-8.0, 8.0, W, dtype=np.float32)[None, :, None]
    base = base + ramp
    return np.clip(base, 0, 255).astype(np.float32)


def make_carrier(kind: str, H: int, W: int) -> np.ndarray:
    """载体（文字/几何/孤立点）→ bool 笔画掩码。"""
    m = np.zeros((H, W), np.uint8)
    if kind == "text":
        cv2.putText(m, "GLOW", (int(W * 0.20), int(H * 0.52)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.7, 255, 3, cv2.LINE_AA)
        cv2.putText(m, "TEST", (int(W * 0.24), int(H * 0.70)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 3, cv2.LINE_AA)
    elif kind == "geometry":
        cv2.rectangle(m, (int(W * 0.24), int(H * 0.28)),
                      (int(W * 0.72), int(H * 0.66)), 255, -1)
        cv2.circle(m, (int(W * 0.50), int(H * 0.47)), int(H * 0.09), 255, -1)
        cv2.line(m, (int(W * 0.08), int(H * 0.86)),
                 (int(W * 0.92), int(H * 0.14)), 255, 3)
    elif kind == "isolated":
        cv2.circle(m, (int(W * 0.50), int(H * 0.50)), int(H * 0.045), 255, -1)
    else:
        raise ValueError(kind)
    return (m > 0)


def hsv_rgb(h: float, s: float = 0.7, v: float = 0.9) -> np.ndarray:
    """HSV（0-360 度制）→ RGB float32 (3,)。"""
    px = np.array([[[h / 2.0, s * 255.0, v * 255.0]]], np.uint8)
    return cv2.cvtColor(px, cv2.COLOR_HSV2RGB)[0, 0].astype(np.float32)


# ---------------------------------------------------------------------------
# 用例
# ---------------------------------------------------------------------------

@dataclass
class GenCase:
    """一个合成 GT 用例（规格 §5 的产物 + 真值）。"""
    index: int
    size: int
    image: np.ndarray            # uint8 HxWx3 观测图 P
    truth: np.ndarray            # uint8 HxWx3 无发光参考 U
    B: np.ndarray                # float HxWx3 真值背景
    M_text: np.ndarray           # bool 载体笔画
    M_aa: np.ndarray             # float32 笔画 AA 覆盖率
    alpha: np.ndarray            # float32 α 场
    glow_truth: np.ndarray       # bool 发光域真值（光晕实际改变像素的区域）
    G: np.ndarray                # (3,) 发光色
    T: np.ndarray                # (3,) 文字色
    sigma_g: float
    A: float
    mode: str
    dye: str
    axis: dict = field(default_factory=dict)   # 轴标签（参与 gt_hash）


def _build_case(idx: int, *, hue: float, mode: str, dye: str, bg: str,
                sigma_g: float, A: float, carrier: str,
                text_color: str, size: int, glow_v: float = 0.9,
                bg_kind_override: str | None = None) -> GenCase:
    rng = np.random.default_rng(1000 + idx)
    H = W = size
    kind = bg_kind_override or bg
    B = make_background(kind, H, W, rng)
    M_text = make_carrier(carrier, H, W)
    G = hsv_rgb(hue, s=0.7, v=glow_v)
    if text_color == "extreme":            # 同色向极端例：同 hue 的深饱和文字
        T = hsv_rgb(hue, s=0.9, v=0.32)
    else:
        T = np.array([28.0, 28.0, 32.0], np.float32)
    P, U, M_aa, alpha, glow_truth = compose(B, M_text, T, G, sigma_g, A,
                                            mode, dye)
    axis = {"hue": round(float(hue), 2), "mode": mode, "dye": dye,
            "bg": kind, "sigma_g": round(float(sigma_g), 2),
            "A": round(float(A), 2), "carrier": carrier,
            "text_color": text_color, "size": size}
    return GenCase(
        index=idx, size=size,
        image=np.clip(P, 0, 255).astype(np.uint8),
        truth=np.clip(U, 0, 255).astype(np.uint8),
        B=B.astype(np.float32), M_text=M_text, M_aa=M_aa,
        alpha=alpha, glow_truth=glow_truth, G=G, T=T,
        sigma_g=float(sigma_g), A=float(A), mode=mode, dye=dye, axis=axis,
    )


def _base_grid(size: int) -> list[GenCase]:
    """基础网格：24 hue × 3 mode × 2 dye × 6 bg × 2 σ_g ≈ 1728 张。"""
    cases: list[GenCase] = []
    idx = 0
    for hue in range(0, 360, 15):                # 24 色相
        for mode in ("blend", "additive", "screen"):
            for dye in ("through", "behind"):
                for bg in BG_NAMES:
                    for sg in (10.0, 30.0):
                        tc = "extreme" if (idx % 4 == 0) else "neutral"
                        carrier = "text" if (idx % 5) else "geometry"
                        cases.append(_build_case(
                            idx, hue=hue, mode=mode, dye=dye, bg=bg,
                            sigma_g=sg, A=0.6, carrier=carrier,
                            text_color=tc, size=size))
                        idx += 1
    return cases


def _extra_axes(size: int) -> list[GenCase]:
    """孤立域 / 饱和诱导 / 源稀缺 / 灰度退化 + 重点交叉各 200。"""
    cases: list[GenCase] = []
    idx = 100000

    def add(**kw):
        nonlocal idx
        cases.append(_build_case(idx, size=size, **kw))
        idx += 1

    # 孤立域：无文字，只有孤立载体 + 全环光晕
    for hue in range(0, 360, 15):
        for mode in ("blend", "additive", "screen"):
            add(hue=hue, mode=mode, dye="through", bg="dark",
                sigma_g=14.0, A=0.6, carrier="isolated", text_color="neutral")
    # 饱和诱导：深色 G × additive × A≥0.9
    for hue in range(0, 360, 15):
        add(hue=hue, mode="additive", dye="through", bg="dark",
            sigma_g=12.0, A=0.95, carrier="text", text_color="neutral",
            glow_v=0.35)
    # 源稀缺：大光晕 ≥ 半幅（σ_g 大、A 高）
    for hue in range(0, 360, 15):
        for mode in ("blend", "additive"):
            add(hue=hue, mode=mode, dye="behind", bg="gray",
                sigma_g=40.0, A=0.9, carrier="text", text_color="neutral")
    # 灰度退化轴：灰光晕（chroma≈0）
    for mode in ("blend", "additive", "screen"):
        for dye in ("through", "behind"):
            add(hue=0.0, mode=mode, dye=dye, bg="gray",
                sigma_g=16.0, A=0.7, carrier="text", text_color="neutral",
                bg_kind_override="gray")
    # 重点交叉 1：纹理 × 无色差 × blend（200）
    rng = np.random.default_rng(20260827)
    for _ in range(200):
        add(hue=0.0, mode="blend", dye="behind", bg="texture",
            sigma_g=float(rng.uniform(8, 36)), A=float(rng.uniform(0.4, 0.9)),
            carrier="text", text_color="neutral",
            bg_kind_override="texture")
    # 重点交叉 2：screen × 大色差（200）
    for _ in range(200):
        add(hue=float(rng.uniform(0, 360)), mode="screen", dye="through",
            bg="green" if rng.random() < 0.5 else "warm",
            sigma_g=float(rng.uniform(8, 36)), A=float(rng.uniform(0.4, 0.9)),
            carrier="text" if rng.random() < 0.7 else "geometry",
            text_color="neutral")
    return cases


# ---------------------------------------------------------------------------
# 用例集 / 哈希 / 子集
# ---------------------------------------------------------------------------

def full_cases(size: int = 512) -> list[GenCase]:
    """规格 §5 基础网格 + 重点交叉轴（固定种子，确定性）。"""
    return _base_grid(size) + _extra_axes(size)


def hash_cases(cases: list[GenCase]) -> str:
    """用例集哈希：轴规格的规范 JSON → sha256（前 16 位）。

    参与 frozen.json 的 gt_hash；生成器任一轴改动即漂移 → 冻结失效。
    """
    lines = [json.dumps(c.axis, sort_keys=True, ensure_ascii=False)
             for c in cases]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]


def iter_cases(size: int = 512, subset: str = "all",
               limit: int | None = None) -> list[GenCase]:
    """按子集取用例：
      all        = 全集（标定/回归都用它算 gt_hash）
      calib      = 基础网格 1/3(步长3) + 全部重点交叉轴（规格 §7 标定子集）
      acceptance = 基础网格剩余 2/3（回归验收）
    """
    full = full_cases(size)
    n_base = len(_base_grid(size))
    if subset == "all":
        cases = full
    elif subset == "calib":
        cases = full[:n_base][::3] + full[n_base:]
    elif subset == "acceptance":
        cases = full[:n_base][1::3] + full[:n_base][2::3]
    else:
        raise ValueError(subset)
    if limit is not None:
        cases = cases[:limit]
    return cases


# ---------------------------------------------------------------------------
# 自检（M0 验收门：合成 vs 独立解析式 < 1e-3）
# ---------------------------------------------------------------------------

def _analytic_pixel(B, M_aa, alpha, T, G, mode, dye):
    """单像素解析式（Python float，独立于向量化实现）。"""
    def ch(a):                      # 逐通道
        Bv, Tv, Gv = B[a], T[a], G[a]
        if dye == "through":
            U = Bv * (1 - M_aa) + Tv * M_aa
            if mode == "blend":
                return (1 - alpha) * U + alpha * Gv
            if mode == "additive":
                return max(0.0, min(255.0, U + alpha * Gv))
            return U + alpha * Gv * (1 - U / 255.0)
        else:
            if mode == "blend":
                L = (1 - alpha) * Bv + alpha * Gv
            elif mode == "additive":
                L = max(0.0, min(255.0, Bv + alpha * Gv))
            else:
                L = Bv + alpha * Gv * (1 - Bv / 255.0)
            return L * (1 - M_aa) + Tv * M_aa
    return np.array([ch(0), ch(1), ch(2)], np.float64)


def self_check(size: int = 256, n_samples: int = 3000,
               verbose: bool = True) -> float:
    """合成结果与独立解析式抽样比对，返回最大绝对误差（须 < 1e-3）。"""
    rng = np.random.default_rng(7)
    hues = list(range(0, 360, 45))
    modes = ("blend", "additive", "screen")
    dyes = ("through", "behind")
    worst = 0.0
    n_tested = 0
    for hue in hues:
        for mode in modes:
            for dye in dyes:
                B = make_background("texture", size, size,
                                    np.random.default_rng(1))
                M_text = make_carrier("text", size, size)
                G = hsv_rgb(hue, 0.7, 0.9)
                T = hsv_rgb(hue, 0.9, 0.32)
                for sg in (6.0, 20.0, 40.0):
                    P, *_ = compose(B, M_text, T, G, sg, 0.6, mode, dye)
                    d = signed_distance(M_text)
                    alpha = 0.6 * _norm.cdf(-d / sg)
                    M_aa = aa_coverage(d)
                    ys, xs = rng.integers(0, size, (2, n_samples))
                    for y, x in zip(ys, xs):
                        ref = _analytic_pixel(B[y, x], float(M_aa[y, x]),
                                              float(alpha[y, x]),
                                              T, G, mode, dye)
                        got = P[y, x].astype(np.float64)
                        e = float(np.max(np.abs(ref - got)))
                        worst = max(worst, e)
                        n_tested += 1
    if verbose:
        print(f"[GT self_check] {n_tested} 像素 × {len(hues) * len(modes) * len(dyes) * 3}"
              f" 组合，最大误差 {worst:.3e}（门 < 1e-3）→ "
              f"{'PASS' if worst < 1e-3 else 'FAIL'}")
    return worst


if __name__ == "__main__":
    self_check()
