"""v4.1 通用去发光 · 主编排（规格 §2.3）。

run(rgb) → DeGlowResult
  M-A 纹理实测 → M-B 全局背景场 → M-C 双偏差场+种子
  → 第1轮生长(τ=0.5) → 环带 → 模式拟合
  → 第2轮生长(τ=0.7, 模式自适应方向) → 合并 → α 场(σ̂_g)
  → 环带重估 → 二次拟合 → 反演+校验+三态路由 → 档3重建(S≥S_min PatchMatch /
    源不足 SYNTH) → 溯源报告。

尺寸约简：种子为空 → has_glow=False（no-op，不触图）。
"""
from __future__ import annotations

import numpy as np

from deglow.core import (alpha as alpha_mod, background, deviation, fit,
                         grow, invert, provenance, rebuild, texture)
from deglow.core.types import DeGlowResult, Domain, TexStats

# 规格 §4 参数兜底默认（自动标定冻结前使用；frozen.json 机制见落地文档）
_DEFAULTS = {
    "tau1": 0.5, "tau2": 0.7, "k": 3.0, "k_lf": 4.0,
    "kv": 2.0, "s_min": 1.5, "max_doms": 24,
    "q_off": 55.0, "max_area_ratio": 0.40, "max_box_ratio": 0.40,
    "max_side": 960,
    "fallback_chroma": 12.0, "fallback_align": 0.5,
}


def _veil_fallback(P, seeds, carrier, Dg, sig, p) -> list:
    """D3 兜底：首轮生长/守卫失败时，用色晕轮（不依赖 Dg）补检平滑光晕。

    从载体沿带按「色度 ≥ veil_chroma ∧ 相对全局暗背景色向余弦 > veil_tau ∧
    亮度 ≥ 背景」连通生长；25% 安全阀防误吞彩色大区；输出域再过两道弱门：
      ① split_domains 最小面积（丢 AA 边碎片）
      ② 域均色度 ≥ fallback_chroma（默认 12）——防「纯文字无发光」误报：
         实测真实纯文字图域均色度≈4（灰度阶调），中量光晕（A≈0.6）≈20+。
    守卫的强彩色核心门（chroma>45 占比≥35%）不适用于兜底：中量光晕核心
    chroma 只到 ~45，会再次误杀；fallback 仅在主路径判定「无强发光信号」
    后进入，安全性由 grow_veil 自身条件 + 载体排除 + 最小面积 + 域均色度
    + 25% 阀共同承担。
    """
    from deglow.core import background as bgmod
    from deglow.core import grow as growmod
    H, W = P.shape[:2]
    if not seeds.any():
        return []
    u = growmod.core_direction(Dg, seeds)
    gb = bgmod.estimate_global_bg(P)
    veil = growmod.grow_veil(P, seeds, carrier, u, gb,
                             tau=p.get("veil_tau", 0.6),
                             chroma_min=p.get("veil_chroma", 20.0))
    if veil.sum() > 0.25 * H * W:            # 安全阀：误吞彩色大区 → 放弃
        veil = seeds.copy()
    comps = growmod.split_domains(veil & ~carrier, carrier, sig)
    if not comps:
        return []
    chroma = np.max(P, axis=-1) - np.min(P, axis=-1)
    thr = p.get("fallback_chroma", 12.0)
    keep = []
    for c in comps:
        if float(chroma[c].mean()) < thr:
            continue
        u = growmod.core_direction(Dg, c)
        dot = np.sum(Dg[c] * u, axis=-1)
        mag = np.sqrt(np.clip(np.sum(Dg[c] * Dg[c], axis=-1), 0, None))
        align = float(np.mean((dot / (mag + 1e-9)) > 0.5) if mag.size else 0.0)
        if align >= p.get("fallback_align", 0.5):   # 与守卫同款方向一致性门
            keep.append(c)
    return keep


def _merge_params(params: dict | None) -> dict:
    p = dict(_DEFAULTS)
    if params:
        p.update({k: v for k, v in params.items() if v is not None})
    return p


def _apply_frozen(p: dict, calib: dict | None) -> dict:
    """并入冻结参数（规格 §4 frozen.json）。calib 为空 → 维持兜底默认。"""
    if not calib:
        return p
    mapping = {"tau_dir": ("tau1", "tau2"), "kv": ("kv",), "k": ("k",),
               "k_lf": ("k_lf",), "s_min": ("s_min",),
               "veil_chroma": ("veil_chroma",), "veil_tau": ("veil_tau",),
               "guard_sig": ("guard_sig",), "guard_chroma": ("guard_chroma",),
               "guard_align": ("guard_align",)}
    for key, dsts in mapping.items():
        v = calib.get(key)
        if v is None:
            continue
        if isinstance(v, (list, tuple)) and len(v) == 2:
            p[dsts[0]], p[dsts[1]] = float(v[0]), float(v[1])
        else:
            for d in dsts:
                p[d] = float(v)
    return p


def detect_carrier(rgb: np.ndarray, params: dict) -> np.ndarray:
    """检测器载体（文字笔画蒙版）；内部默认走 DBNet(ml)。"""
    from core.text_select import detect_text_mask
    tmask, _ = detect_text_mask(
        rgb, method="ml", q_off=params["q_off"],
        max_area_ratio=params["max_area_ratio"],
        max_box_ratio=params["max_box_ratio"],
        max_side=params["max_side"], tint_fill=False,
    )
    return tmask


def run(rgb: np.ndarray, carrier_mask: np.ndarray | None = None,
        deglow_strength: float = 1.0,
        params: dict | None = None,
        calib: dict | None = None) -> DeGlowResult:
    """calib：frozen.json 冻结参数（规格 §4）。传入后覆盖兜底默认。"""
    p = _apply_frozen(_merge_params(params), calib)
    H, W = rgb.shape[:2]
    P = np.asarray(rgb, np.float32)
    prov0 = np.zeros((H, W), np.uint8)
    conf0 = np.ones((H, W), np.float32)

    # ---- M-A / M-B / M-C ----
    sig = texture.estimate_texture(P)
    Bg = background.build_global(P, sig.l_tex, sig.bar)
    Dg, Dh, Dl = deviation.build_fields(P, Bg, sig)

    tmask = carrier_mask if (carrier_mask is not None and carrier_mask.any()) \
        else detect_carrier(rgb, p)
    carrier = deviation.carrier_union(tmask, Dh, sig, k=p["k"])
    seeds = deviation.collect_seeds(carrier, Dl, sig, k_lf=p["k_lf"])
    if not seeds.any():
        return DeGlowResult(
            image=np.clip(P, 0, 255).astype(np.uint8), prov=prov0, conf=conf0,
            report={"has_glow": False, "glow_pix": 0, "tier_pix": {},
                    "domains": [], "sigma_bar": sig.bar, "l_tex": sig.l_tex},
            domains=[], has_glow=False,
        )

    # ---- 第 1 轮生长 → 环带 → 拟合 ----
    u0 = grow.core_direction(Dg, seeds)
    grown = grow.grow_round(Dg, seeds, u0, sig, tau=p["tau1"], k=p["k"])
    comps = grow.split_domains(grown, carrier, sig)
    satP = grow.saturation_mask(P)
    comps.sort(key=lambda c: int(c.sum()), reverse=True)
    # 强发光信号守卫（对照通道法 strong_green/min_strong 语义，防普通图误报）：
    # 大而平滑的光晕相对全局背景场偏差可能很小（引导滤波把光晕当背景），
    # 但其「强彩色 + 明亮」像素（类比 strong_green：chroma>45 且 max>90）
    # 在域内占比高、方向一致 → 这才是 veil 信号的真实判据；纯亮度/灰阶纹理、
    # 衣物阶调（规格 R8 灰度轴）与小幅彩色 AA 边会被豁免。
    chroma = np.max(P, axis=-1) - np.min(P, axis=-1)
    strong_colored = (chroma > 45.0) & (np.max(P, axis=-1) > 90.0)
    kept = []
    for c in comps:
        u = grow.core_direction(Dg, c)
        dot = np.sum(Dg[c] * u, axis=-1)
        mag = np.sqrt(np.clip(np.sum(Dg[c] * Dg[c], axis=-1), 0, None))
        align = float(np.mean((dot / (mag + 1e-9)) > 0.5) if mag.size else 0.0)
        c_mean = float(np.mean(chroma[c])) if c.sum() else 0.0
        sig_frac = float(strong_colored[c].sum()) / max(float(c.sum()), 1.0)
        if (sig_frac >= p.get("guard_sig", 0.35)
                and c_mean >= p.get("guard_chroma", 15.0)
                and align >= p.get("guard_align", 0.5)):
            kept.append(c)
    comps = kept
    if not comps:
        # D3 兜底：引导滤波把平滑大光晕当背景 → 首轮 |D| 偏小 → 欠覆盖/守卫拒绝。
        # 色晕轮不依赖 Dg，从载体沿带按「色度+方向+亮度」连通生长，可兜住该轴。
        comps = _veil_fallback(P, seeds, carrier, Dg, sig, p)
        if not comps:
            return DeGlowResult(
                image=np.clip(P, 0, 255).astype(np.uint8), prov=prov0,
                conf=conf0, report={"has_glow": False, "glow_pix": 0,
                                    "tier_pix": {}, "domains": [],
                                    "sigma_bar": sig.bar, "l_tex": sig.l_tex,
                                    "guard": "no-strong-glow-signal"},
                domains=[], has_glow=False,
            )
    doms = [Domain(id=i, mask=c, carrier_mask=carrier, B_global=Bg,
                   saturated=satP & c)
            for i, c in enumerate(comps[: p["max_doms"]])]

    for d in doms:
        background.build_ring(d, P, carrier, 3.0)
    for d in doms:
        fit.fit_mode(d, P, Dg)

    # ---- 第 2 轮生长（模式自适应方向）→ 合并 → 色晕生长（补欠覆盖）→ 重拆域 ----
    outs2 = grow.grow_round2(Dg, doms, sig, tau=p["tau2"], k=p["k"])
    for d, m2 in zip(doms, outs2):
        d.mask = (m2 & ~d.saturated) | (d.mask & d.saturated)
        if d.saturated is not None:
            d.saturated &= d.mask
    merged = grow.merge_like([d for d in doms if d.mask.any()])

    # 色晕连通生长：对标通道法 zone（全域亮绿 11701px vs 首轮只长到 ~3k ——
    # 引导滤波把平滑大光晕当背景导致 D 偏小 → 欠覆盖，偏差 D3）
    global_bg = background.estimate_global_bg(P)
    veil = np.zeros((H, W), bool)
    p_veil = p.get("veil_chroma", 20.0)
    for d in merged:
        u = d.u_hat if d.u_hat is not None else grow.core_direction(Dg, d.mask)
        veil |= grow.grow_veil(P, d.mask, carrier, u, global_bg,
                               tau=p.get("veil_tau", 0.6),
                               chroma_min=p_veil)
    # 安全阀：色晕轮扩张超过全图 25% → 判定为误吞（彩色大区域而非光晕），
    # 放弃色晕轮，回退到第 2 轮生长结果，避免整图被拉平
    if veil.sum() > 0.25 * H * W:
        veil[:, :] = False
        for d in merged:
            veil |= d.mask
    comps2 = grow.split_domains(veil & ~carrier, carrier, sig)
    if comps2:
        comps2.sort(key=lambda c: int(c.sum()), reverse=True)
        doms = [Domain(id=i, mask=c, carrier_mask=carrier, B_global=Bg,
                       saturated=satP & c)
                for i, c in enumerate(comps2[: p["max_doms"]])]
    else:
        doms = merged

    union_all = np.zeros((H, W), bool)
    for d in doms:
        union_all |= d.mask
    for d in doms:
        alpha_mod.build_alpha(d, P, Dg, carrier)     # 得 σ̂_g（距离衰减拟合）
        background.build_ring(d, P, carrier, d.sigma_g, exclude=union_all)
        fit.fit_mode(d, P, Dg)                        # 两轮环带收敛后的二次拟合
        alpha_mod.build_alpha(d, P, Dg, carrier)

    # ---- M-G 反演 + M-H 重建 + 残绿清理 + M-I 溯源 ----
    out, prov, conf = invert.invert_and_route(P, doms, sig, kv=p["kv"],
                                              strength=deglow_strength)
    out, prov = rebuild.reconstruct(out, prov, P, doms, carrier, sig,
                                    strength=deglow_strength,
                                    global_bg=global_bg)
    out, prov = rebuild.residual_cleanup(out, prov, P, doms, global_bg,
                                         carrier)
    report = provenance.assemble_report(prov, conf, doms, sig)
    return DeGlowResult(
        image=np.clip(out, 0, 255).astype(np.uint8),
        prov=prov, conf=conf, report=report, domains=doms, has_glow=True,
    )