"""夜间回归入口（规格 §8）：固定种子 GT 集 + 真实用例集，JSON 报告，阻断合入。"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def run_regression(cases, params: dict | None = None, calib: dict | None = None,
                   limit: int | None = None, verbose: bool = True,
                   ) -> dict:
    """在 GT 用例集上跑 pipeline 并聚合分层指标。

    返回 {n_cases, n_blocking, epsilon_all, layers, by_mode, fails, ...}。
    任一轴任一层失败（含 ε≥0.1%）→ n_blocking>0 → 调用方阻断合入。
    """
    from deglow.calib.pareto import check_frozen
    from deglow.gt.generator import hash_cases
    from deglow.gt.metrics import aggregate, evaluate_case, gate_passes
    from deglow.pipeline import run as run_pipe

    if calib is not None:
        ok, msg = check_frozen(calib, cases)
        if not ok:
            raise RuntimeError(f"[regression] 冻结断言失败：{msg}")

    if limit is not None:
        cases = cases[:limit]
    t0 = time.time()
    results = []
    for ci, case in enumerate(cases):
        res = run_pipe(case.image, carrier_mask=case.M_text, params=params,
                       calib=calib)
        results.append(evaluate_case(res, case))
        if verbose and (ci + 1) % 25 == 0:
            print(f"  regression {ci + 1}/{len(cases)} "
                  f"({time.time() - t0:.0f}s)")
    agg = aggregate(results)
    agg["runtime_s"] = round(time.time() - t0, 1)
    agg["frozen_gt_hash"] = calib.get("gt_hash") if calib else None
    agg["gt_hash"] = hash_cases(cases)
    fails = []
    for r in results:
        ok, fs = gate_passes(r)
        if not ok:
            fails.append({"index": r["index"], "axis": r["axis"], "fails": fs})
    agg["fails"] = fails[:50]
    return agg


def save_report(agg: dict, path: str | Path) -> Path:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    return path


def run_real_cases(targets, params=None, calib=None, max_side: int = 960,
                   verbose=True) -> dict:
    """真实用例集（无 GT）：记录四态占比 / 每域 mode / ε 代理趋势。"""
    import cv2
    from deglow.pipeline import run as run_pipe
    out = {}
    for tp in targets:
        tp = Path(tp)
        if tp.suffix == ".bin":
            from PIL import Image
            import io
            rgb = np.asarray(Image.open(io.BytesIO(tp.read_bytes()))
                             .convert("RGB"), np.uint8)
        else:
            img = cv2.imread(str(tp), cv2.IMREAD_COLOR)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        s = min(1.0, max_side / max(h, w))
        if s < 1.0:
            rgb = cv2.resize(rgb, (int(w * s), int(h * s)))
        res = run_pipe(rgb, params=params, calib=calib)
        rep = res.report
        out[tp.name] = {
            "has_glow": rep["has_glow"],
            "glow_pix": rep.get("glow_pix", 0),
            "tier": rep.get("tier_pix", {}),
            "epsilon_proxy": rep.get("epsilon_proxy"),
            "domains": [{k: d[k] for k in
                         ("id", "mode", "calibrated", "alpha_max", "sigma_g",
                          "dye", "pix") if k in d}
                        for d in rep.get("domains", [])],
        }
        if verbose:
            print(f"  {tp.name}: glow={out[tp.name]['has_glow']} "
                  f"ε_proxy={out[tp.name]['epsilon_proxy']}")
    return out
