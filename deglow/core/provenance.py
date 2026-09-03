"""M-I · 溯源图与报告（规格 §3 M-I）。

prov 图像 = Prov 五值 uint8；conf = uint8×1（[0,1]→[0,255]）。
report JSON 含每域 {mode, calibrated, α 分布分位数, 三态像素计数, dye,
检出门通过率}；ε 监督钩子：无 GT 时以「校验门余量 <0.2 的像素占比」逼近 ε 趋势。
"""
from __future__ import annotations

import numpy as np

from deglow.core.types import Domain, Prov, TexStats

_PROV_NAMES = {
    Prov.ORIGINAL: "original",
    Prov.INVERTED: "inverted",
    Prov.SUBTRACTED: "subtracted",
    Prov.FILLED: "filled",
    Prov.SYNTH: "synth",
}


def domain_summary(dom: Domain, prov: np.ndarray, conf: np.ndarray) -> dict:
    m = dom.mask
    if not m.any():
        return {"id": dom.id, "pix": 0}
    alpha = dom.alpha
    alpha_q = None
    if alpha is not None:
        alpha_q = [round(float(x), 3) for x in
                   np.quantile(alpha[dom.mask], [0.25, 0.50, 0.75, 0.95])]
    return {
        "id": dom.id,
        "pix": int(m.sum()),
        "mode": dom.mode,
        "calibrated": bool(dom.calibrated),
        "alpha_max": round(dom.alpha_max, 3) if dom.alpha_max is not None else None,
        "sigma_g": round(dom.sigma_g, 2),
        "dye": dom.dye,
        "kv_local": round(dom.kv_local, 3),
        "alpha_quantiles": alpha_q,
        "tier_pix": {
            "inverted": int((m & (prov == Prov.INVERTED)).sum()),
            "subtracted": int((m & (prov == Prov.SUBTRACTED)).sum()),
            "filled": int((m & (prov == Prov.FILLED)).sum()),
            "synth": int((m & (prov == Prov.SYNTH)).sum()),
            "saturated": int((dom.saturated & m).sum()
                             if dom.saturated is not None else 0),
        },
        "verify_pass_rate": round(float((conf[m] > 0).mean()), 3)
        if m.any() else 0.0,
    }


def assemble_report(prov: np.ndarray, conf: np.ndarray,
                    doms: list[Domain], sig: TexStats) -> dict:
    total = prov.size
    n_glow = int((prov > Prov.ORIGINAL).sum())
    tier = {name: int((prov == code).sum())
            for code, name in _PROV_NAMES.items()}
    low_margin = int((conf[prov > Prov.ORIGINAL] < 0.2).sum())
    return {
        "has_glow": n_glow > 0,
        "glow_pix": n_glow,
        "glow_ratio": round(n_glow / max(total, 1), 5),
        "tier_pix": tier,
        "epsilon_proxy": round(low_margin / max(n_glow, 1), 5),
        "domains": [domain_summary(d, prov, conf) for d in doms],
        "sigma_bar": round(sig.bar, 3),
        "l_tex": sig.l_tex,
    }