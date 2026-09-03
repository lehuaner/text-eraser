"""M-H · 重建蒙版与填充、源充足度分级、残差回灌（规格 §3 M-H）。

集成策略（遵循项目规则「先去发光 → 原样走普通去文字路径」）：
- 档3（FILLED）/饱和像素在**去发光阶段**就地重建，保证返回的去发光图可不残留光晕：
  源 = 原图 ∧ ~全部发光域（纯背景）∪ 反演校验通过像素，按纹理一致性排序；
  S = |有效源|/|待重建| ≥ S_min → PatchMatch（复用 core.patch_fill.inpaint）；
  S < S_min → 合成回退（B̂_ring 底 + 环带 FFT 幅值谱随机相位噪声），溯源 SYNTH。
- 残差回灌：D_out·û > max(2σ_tex, 6) 的像素补入蒙版，仅重填充一次（防循环）。
"""
from __future__ import annotations

import cv2
import numpy as np

from deglow.core.background import ellipse
from deglow.core.texture import luminance
from deglow.core.types import Domain, Prov, TexStats
from core.patch_fill import inpaint as pm_inpaint

_S_MIN = 1.5


def _sources(masks: list[np.ndarray], prov: np.ndarray) -> np.ndarray:
    """填充分布：纯背景（非发光域）∪ 反演校验通过像素。"""
    union = np.zeros_like(prov, bool)
    for dm in masks:
        union |= dm
    ok_inv = (prov == Prov.INVERTED) | (prov == Prov.SUBTRACTED)
    return (~union) | ok_inv


def _synth_fill(P: np.ndarray, fill_mask: np.ndarray, dom: Domain,
                ring_band: np.ndarray, sig: TexStats) -> np.ndarray:
    """合成回退：B̂_ring 底 + 环带高通 FFT 幅值谱、随机相位、σ 匹配。"""
    B_ring = dom.B_ring
    H, W = P.shape[:2]
    out = P.copy()
    if B_ring is None or not ring_band.any() or not fill_mask.any():
        return out
    gray = luminance(P)
    hp = (gray - cv2.GaussianBlur(gray, (9, 9), sigmaX=4,
                                  sigmaY=4)).astype(np.float32)
    spec = np.fft.fft2(hp)
    amp = np.abs(spec)
    phase = np.random.default_rng(0).standard_normal((H, W)) * np.pi
    tex = np.fft.ifft2(amp * np.exp(1j * phase)).real.astype(np.float32)
    # σ 匹配（按环带实测纹理幅度，避免过强/过弱）
    if ring_band.any():
        target = float(np.std(hp[ring_band]))
        current = float(np.std(tex[ring_band])) + 1e-6
        tex = tex * (target / current)
    fill = np.clip(B_ring + tex[..., None], 0, 255).astype(np.float32)
    out[fill_mask] = fill[fill_mask]
    return out


def _ring_band(dom: Domain, carrier: np.ndarray) -> np.ndarray:
    w = max(int(np.ceil(1.5 * dom.sigma_g)) + 2, 3)
    band = cv2.dilate(dom.mask.astype(np.uint8), ellipse(w)) > 0
    band &= ~dom.mask & ~carrier
    return band


def _real_bg_color(P: np.ndarray, prov: np.ndarray, dom: Domain,
                   carrier: np.ndarray, min_px: int = 32) -> np.ndarray | None:
    """档3 落底色：取「域外 prov==ORIGINAL 邻域」的背景色。

    与 build_ring 的环带均值相比：平滑大光晕会被引导滤波当成背景，环带常
    整个落在光晕尾迹内 → 环带均值=光晕色 → 档3 落底整片泛绿（实测
    ring_fill=[72,75,58] 刷绿 17045px）。此处只从**未被处理过的真背景**
    (prov==ORIGINAL，即非发光域、非载体像素) 取样，不足再向外扩圈；
    仍不足 → 返回 None，由调用方回退全局中性暗背景。

    取样判据（实测校准，见 data/_dbg_band.py）：真实带纹理背景无「中性区」
    （该图 ring 色度全在 16~21），亮/暗两端都混有绿尾迹。**亮度中位窗
    35%~75%** 恰好落在真背景主体（band 均值 [94,91,77]、中位亮度窗均值
    [99,94,81]，均为 R>G>B 暖向，与真实背景同色相）——避开暗绿尾迹
    （暗侧截尾曾选出 [74.8,75.1,60.5] 橄榄）与亮反光；再剔除强彩色像素
    （chroma>60，光晕核心色）防污染。
    """
    w0 = max(int(np.ceil(1.5 * dom.sigma_g)) + 2, 3)
    for w in (w0, w0 + 4, w0 * 3):
        band = cv2.dilate(dom.mask.astype(np.uint8), ellipse(w)) > 0
        band &= ~dom.mask & ~carrier & (prov == Prov.ORIGINAL)
        if band.sum() < min_px:
            continue
        lv = luminance(P[band])
        lo, hi = np.percentile(lv, [35, 75])
        win = (lv >= lo) & (lv <= hi)
        if win.sum() < 8:
            continue
        px = P[band][win]
        ch = np.max(px, axis=-1) - np.min(px, axis=-1)
        strong = ch > 60.0
        if (ch <= 60.0).sum() >= 8:
            px = px[ch <= 60.0]
        return px.mean(0).astype(np.float32)
    return None


def reconstruct(out: np.ndarray, prov: np.ndarray, P: np.ndarray,
                doms: list[Domain], carrier: np.ndarray, sig: TexStats,
                strength: float = 1.0,
                global_bg: np.ndarray | None = None,
                ) -> tuple[np.ndarray, np.ndarray]:
    """就地重建档3像素；返回 (out, prov)。

    光晕域优先级「环带常数背景色落底」（对标通道法「整区拉平到一个背景色」：
    避免 PatchMatch 源把域外残余发光复制进来 → 泛绿/残留亮），但落底色不再
    无脑取 build_ring 的环带均值（平滑大光晕会把环带污染成光晕色 → 整片刷绿，
    实测 ring_fill=[72,75,58] 刷绿 17045px）——改为：
      ① _real_bg_color：域外 prov==ORIGINAL 真背景的暗侧截尾均值（首选）；
      ② 取不到 → 全局中性暗背景 estimate_global_bg；
      ③ 仍失败 → 保底沿用 ring_fill/B_ring。
    strength ∈ [0,1]：0=不去色；1=完全还原为背景色（与「去发光强度」滑条一致）。
    无背景信息的小块区域 → 干净源 PatchMatch 兜底。
    """
    fill_mask = (prov == Prov.FILLED)
    if not fill_mask.any():
        return out, prov
    s = float(np.clip(strength, 0.0, 1.0))
    fallback = None
    if global_bg is not None:
        fallback = np.asarray(global_bg, np.float32)[None, None, :]
    leftover = np.zeros_like(fill_mask)
    for dom in doms:
        dm = fill_mask & dom.mask
        if not dm.any():
            continue
        bgv = None
        if dom.B_ring is not None or dom.ring_fill is not None:
            bgv = _real_bg_color(P, prov, dom, carrier)
            if bgv is None:
                if fallback is not None:
                    bgv = np.asarray(fallback[0, 0], np.float32).copy()
                elif dom.ring_fill is not None:      # 防御：无全局背景参数
                    bgv = np.asarray(dom.ring_fill[dm].mean(0), np.float32)
        if bgv is not None:
            fill_v = np.broadcast_to(bgv[None, None, :],
                                     dm.shape + (3,))[dm]
            out[dm] = P[dm] + s * (fill_v - P[dm])
        else:
            leftover |= dm
    if leftover.any():
        src = _sources([d.mask for d in doms], prov)
        S = float(src.sum()) / max(float(leftover.sum()), 1.0)
        if S >= _S_MIN:
            out = pm_inpaint(np.clip(out, 0, 255).astype(np.uint8),
                             (leftover.astype(np.uint8) * 255),
                             sample_mask=(src.astype(np.uint8) * 255))
        else:                                   # 源不足 → 合成回退
            for dom in doms:
                dm = leftover & dom.mask
                if not dm.any():
                    continue
                band = _ring_band(dom, carrier)
                out = _synth_fill(out, dm, dom, band, sig)
                prov[dm] = Prov.SYNTH
    return out.astype(np.float32), prov


def residual_mask(out: np.ndarray, doms: list[Domain], sig: TexStats,
                  thr_min: float = 6.0) -> np.ndarray:
    """残差回灌掩码：D_out·û > max(2σ_tex, 6)。"""
    res = np.zeros_like(sig.sigma_tex, bool)
    for dom in doms:
        if dom.mask is None or dom.u_hat is None or dom.B_ring is None:
            continue
        D = (out - dom.B_ring).astype(np.float32)
        proj = np.abs(np.sum(D * dom.u_hat, axis=-1))
        thr = np.maximum(2.0 * sig.sigma_tex, thr_min)
        near = cv2.dilate(dom.mask.astype(np.uint8), ellipse(6)) > 0
        res |= near & (proj > thr) & ~dom.carrier_mask
    return res


def residual_cleanup(out: np.ndarray, prov: np.ndarray, P: np.ndarray,
                     doms: list[Domain], global_bg: np.ndarray,
                     carrier: np.ndarray,
                     green_thr: int = 6, min_num: int = 8,
                     max_touch: int = 6000) -> tuple[np.ndarray, np.ndarray]:
    """域邻域残绿清理（M-H「单次残差检测回灌」的工程兜底）。

    处理两类残迹（均在发光域邻域内，不触碰远处真实物体）：
      ① 纯绿  g−max(r,b)>green_thr 且色度可见（原判据）；
      ② 橄榄/黄绿  g≈r>b、蓝通道显著偏低、与全局背景明显不同、且偏移方向
         与光晕方向一致——去发光后残留常被拉进「亮但不绿」区间，纯绿门会漏。
    回填全局暗背景色并落 FILLED。
    """
    r = out[..., 0].astype(np.int16)
    g = out[..., 1].astype(np.int16)
    b = out[..., 2].astype(np.int16)
    chroma = np.max(out, axis=-1) - np.min(out, axis=-1)
    green = (g - np.maximum(r, b) > green_thr) & (chroma > 10)
    # 扩展判据：橄榄/黄绿残迹（g≈r>b，纯绿门 g−max(r,b)>6 会漏）。
    # 必要三条件：底色蓝通道显著偏低 + 与全局背景明显不同 + 偏移方向与
    # 光晕方向一致（防误删域邻域内真实的同向物体）。
    bg = np.asarray(global_bg, np.float32)[None, None, :]
    off = out.astype(np.float32) - bg
    offmag = np.max(np.abs(off), axis=-1)
    resid = np.zeros_like(green)
    for dom in doms:
        w = int(max(dom.sigma_g * 3, 4.0)) + 6
        near = cv2.dilate(dom.mask.astype(np.uint8), ellipse(w)) > 0
        u = dom.u_hat
        aligned = np.ones_like(green)
        if u is not None and float(np.linalg.norm(u)) > 1e-6:
            dot = np.sum(off * u, axis=-1)
            magv = np.sqrt(np.clip(np.sum(off * off, axis=-1), 0, None))
            aligned = (dot / (magv + 1e-9)) > 0.35
        olive = ((g > np.maximum(r, b) - 2) & (b < np.minimum(r, g) - 3)
                 & (chroma > 10) & (offmag > 18) & aligned)
        resid |= near & ~carrier & (green | olive)
    if int(resid.sum()) < min_num:
        return out, prov
    if int(resid.sum()) > max_touch:          # 清理范围失控 → 放弃（防误删大区域）
        return out, prov
    out = out.copy()
    bg = np.asarray(global_bg, np.float32)[None, None, :]
    out[resid] = bg
    prov = prov.copy()
    prov[resid] = Prov.FILLED
    return out, prov


def residual_fill(result: np.ndarray, doms: list[Domain], P: np.ndarray,
                  sig: TexStats, carrier: np.ndarray) -> tuple[np.ndarray, int]:
    """对擦除结果做单次残差回灌重填充；返回 (result, 回灌像素数)。"""
    res = residual_mask(result.astype(np.float32), doms, sig)
    if not res.any():
        return result, 0
    union = np.zeros_like(res)
    for d in doms:
        union |= d.mask
    src = (~union) | ((result.astype(np.float32) - P.astype(np.float32)) == 0)
    out = pm_inpaint(np.clip(result, 0, 255).astype(np.uint8),
                     (res.astype(np.uint8) * 255),
                     sample_mask=(src.astype(np.uint8) * 255))
    return out, int(res.sum())