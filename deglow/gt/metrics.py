"""M0 · 分层指标（规格 §6）——按 prov 溯源态分层评估。

参考真值 = GT 无发光参考 U（字层）：
  INVERTED    RMSE<8 且 ΔE94<6
  SUBTRACTED  低频 RMSE<8；高频模式相关 ρ_HP≥0.9；高频幅值比报告 ≈(1−α)
  FILLED      ΔE94<6 且 SSIM≥0.90
  SYNTH       ΔE94<10 且 SSIM≥0.85，覆盖率单独报告
路由 ε       被路由至保留/反演态(ORIGINAL/INVERTED/SUBTRACTED)但 ΔE94>6 的
             发光域像素 / 发光域像素，目标 <0.1%（用 GT 域真值掩码核算）。

评估范围统一为「光晕域」（glow_truth ∧ ~载体），不评估文字笔画本身——
产品契约是去光晕、保文字。
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.color import rgb2lab
from skimage.metrics import structural_similarity as _ssim

from deglow.core.types import Prov


def delta_e94(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE94（kL=kC=kH=1，K1=0.045，K2=0.015）逐像素色差。"""
    dL = lab1[..., 0] - lab2[..., 0]
    C1 = np.sqrt(lab1[..., 1] ** 2 + lab1[..., 2] ** 2)
    C2 = np.sqrt(lab2[..., 1] ** 2 + lab2[..., 2] ** 2)
    dC = C1 - C2
    da = lab1[..., 1] - lab2[..., 1]
    db = lab1[..., 2] - lab2[..., 2]
    dH2 = np.maximum(da * da + db * db - dC * dC, 0.0)
    dH = np.sqrt(dH2)
    SC = 1.0 + 0.045 * C1
    SH = 1.0 + 0.015 * C1
    return np.sqrt((dL) ** 2 + (dC / SC) ** 2 + (dH / SH) ** 2)


def _lab(rgb_uint8: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 → Lab float（skimage，D65）。"""
    return rgb2lab(rgb_uint8.astype(np.float64) / 255.0)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a.astype(np.float64)
                                  - b.astype(np.float64)) ** 2)))


def _luma(a: np.ndarray) -> np.ndarray:
    return (a @ np.array([0.299, 0.587, 0.114], np.float32))


def _highpass(a: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    g = cv2.GaussianBlur(a, (0, 0), sigmaX=sigma)
    return (a - g).astype(np.float32)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if den < 1e-9:
        return 0.0
    return float(np.sum(a * b) / den)


def _ssim_bbox(a: np.ndarray, b: np.ndarray, mask: np.ndarray):
    """mask 包围盒内 SSIM（uint8 HxWx3）。区域过小 → None。"""
    ys, xs = np.where(mask)
    if ys.size < 256:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    if (y1 - y0) < 16 or (x1 - x0) < 16:
        return None
    try:
        return float(_ssim(a[y0:y1, x0:x1], b[y0:y1, x0:x1],
                           channel_axis=2, data_range=255))
    except ValueError:
        return None


def layered_metrics(out: np.ndarray, prov: np.ndarray, truth: np.ndarray,
                    glow_mask: np.ndarray, alpha: np.ndarray | None,
                    ) -> dict:
    """按 prov 溯源态分层的指标（规格 §6）。out/truth uint8，glow_mask bool。"""
    lab_o = _lab(out)
    lab_t = _lab(truth)
    dE = delta_e94(lab_o, lab_t)
    layers = {}
    for code, name in ((Prov.INVERTED, "inverted"),
                       (Prov.SUBTRACTED, "subtracted"),
                       (Prov.FILLED, "filled"),
                       (Prov.SYNTH, "synth")):
        m = (prov == code) & glow_mask
        if not m.any():
            layers[name] = {"pix": 0}
            continue
        o = out[m]; t = truth[m]
        l = {
            "pix": int(m.sum()),
            "rmse": round(rmse(o, t), 3),
            "dE94": round(float(dE[m].mean()), 3),
            "dE94_p95": round(float(np.percentile(dE[m], 95)), 3),
            "ssim": _ssim_bbox(out, truth, m),
        }
        if name == "subtracted":
            # 低频 RMSE（蒙版内低通分量）
            l["rmse_low"] = round(rmse(
                cv2.GaussianBlur(o.astype(np.float32), (0, 0), sigmaX=3),
                cv2.GaussianBlur(t.astype(np.float32), (0, 0), sigmaX=3)), 3)
            # 高频模式相关 ρ_HP
            l["rho_hp"] = round(_pearson(_highpass(_luma(o), 2.0),
                                         _highpass(_luma(t), 2.0)), 4)
            # 高频幅值比 ≈ (1−α)（报告值）
            if alpha is not None and alpha[m].size:
                exp = 1.0 - float(np.mean(alpha[m]))
            else:
                exp = None
            hp_o = np.std(_highpass(_luma(o), 2.0))
            hp_t = np.std(_highpass(_luma(t), 2.0)) + 1e-6
            l["amp_ratio"] = round(float(hp_o / hp_t), 4)
            l["amp_ratio_expected"] = round(exp, 4) if exp is not None else None
        layers[name] = l
    # 保留态（ORIGINAL）在发光域内的像素 = 守卫漏检/未处理
    m = (prov == Prov.ORIGINAL) & glow_mask
    layers["original_in_glow"] = int(m.sum())
    return layers


def routing_epsilon(prov: np.ndarray, out: np.ndarray, truth: np.ndarray,
                    glow_mask: np.ndarray, dE_thr: float = 6.0) -> float:
    """路由 ε：被路由至保留/反演态但 ΔE94(out,GT)>6 的像素 / 发光域像素。"""
    lab_o = _lab(out)
    lab_t = _lab(truth)
    dE = delta_e94(lab_o, lab_t)
    escaped = glow_mask & (prov <= Prov.SUBTRACTED) & (dE > dE_thr)
    n = int(glow_mask.sum())
    if n == 0:
        return 0.0
    return float(escaped.sum()) / n


def evaluate_case(res, case) -> dict:
    """对单个 GT 用例跑完整评估。res 为 DeGlowResult。"""
    glow = case.glow_truth & ~case.M_text
    if res.has_glow and res.report.get("glow_pix", 0) == 0:
        res.has_glow = False  # 防御：报告与标志一致性
    out = np.clip(res.image, 0, 255).astype(np.uint8)
    layers = layered_metrics(out, res.prov, case.truth, glow, case.alpha)
    eps = routing_epsilon(res.prov, out, case.truth, glow)
    # 模式判别：域内多数 mode 是否与 GT 一致（M2 近似口径）
    modes = [d["mode"] for d in res.report.get("domains", [])
             if d.get("pix", 0) > 0]
    detected = res.has_glow and len(modes) > 0
    mode_ok = detected and (case.mode in modes or "unknown" in modes)
    return {
        "index": case.index,
        "axis": case.axis,
        "has_glow": res.has_glow,
        "glow_truth_px": int(glow.sum()),
        "tier": res.report.get("tier_pix"),
        "epsilon": round(eps, 5),
        "layers": layers,
        "modes_detected": modes,
        "mode_correct": bool(mode_ok),
        "guard_miss": bool(not res.has_glow and glow.sum() > 32),
    }


def gate_passes(ev: dict) -> tuple[bool, list[str]]:
    """单用例的门判定（规格 §6 任一轴任一层失败即阻断）。"""
    fails: list[str] = []
    layers = ev["layers"]
    gates = (
        ("inverted", (("rmse", 8.0, "lt"), ("dE94", 6.0, "lt"))),
        ("subtracted", (("rmse_low", 8.0, "lt"), ("rho_hp", 0.9, "ge"))),
        ("filled", (("dE94", 6.0, "lt"), ("ssim", 0.90, "ge"))),
        ("synth", (("dE94", 10.0, "lt"), ("ssim", 0.85, "ge"))),
    )
    for name, checks in gates:
        l = layers.get(name)
        if not l or l.get("pix", 0) == 0:
            continue
        for key, thr, op in checks:
            v = l.get(key)
            if v is None:
                continue
            ok = (v < thr) if op == "lt" else (v >= thr)
            if not ok:
                fails.append(f"{name}.{key}={v} (门 {op} {thr})")
    # 路由 ε 门
    if ev["epsilon"] > 0.001:
        fails.append(f"epsilon={ev['epsilon']} (门 <0.1%)")
    return (not fails, fails)


def aggregate(results: list[dict]) -> dict:
    """跨用例聚合：分层指标均值 + 按轴分组通过率 + 阻断计数。"""
    import collections
    by_axis: dict[str, dict] = collections.defaultdict(
        lambda: {"n": 0, "pass": 0, "eps_sum": 0.0, "glow_px": 0,
                 "guard_miss": 0, "mode_ok": 0})
    layers_all = collections.defaultdict(lambda: {"pix": 0, "rmse": 0.0,
                                                  "dE94": 0.0, "n": 0})
    eps_all = 0.0
    glow_px_all = 0
    n_block = 0
    for ev in results:
        ok, fails = gate_passes(ev)
        if not ok:
            n_block += 1
        ax = ev["axis"].get("mode", "?")
        b = by_axis[ax]
        b["n"] += 1
        b["pass"] += int(ok)
        b["eps_sum"] += ev["epsilon"] * ev["glow_truth_px"]
        b["glow_px"] += ev["glow_truth_px"]
        b["guard_miss"] += int(ev.get("guard_miss", False))
        b["mode_ok"] += int(ev.get("mode_correct", False))
        eps_all += ev["epsilon"] * ev["glow_truth_px"]
        glow_px_all += ev["glow_truth_px"]
        for name, l in ev["layers"].items():
            if isinstance(l, dict) and l.get("pix", 0):
                la = layers_all[name]
                la["pix"] += l["pix"]
                la["rmse"] += l.get("rmse", 0) * l["pix"]
                la["dE94"] += l.get("dE94", 0) * l["pix"]
                la["n"] += 1
    axes = {}
    for ax, b in by_axis.items():
        axes[ax] = {
            "n": b["n"],
            "pass_rate": round(b["pass"] / max(b["n"], 1), 4),
            "guard_miss": b["guard_miss"],
            "mode_acc": round(b["mode_ok"] / max(b["n"], 1), 4),
            "weighted_eps": round(b["eps_sum"] / max(b["glow_px"], 1), 5),
        }
    layers_agg = {}
    for name, la in layers_all.items():
        if la["n"]:
            layers_agg[name] = {
                "pix": la["pix"],
                "rmse": round(la["rmse"] / la["pix"], 3),
                "dE94": round(la["dE94"] / la["pix"], 3),
            }
    return {
        "n_cases": len(results),
        "n_blocking": n_block,
        "epsilon_all": round(eps_all / max(glow_px_all, 1), 5),
        "glow_px_all": glow_px_all,
        "layers": layers_agg,
        "by_mode": axes,
    }
