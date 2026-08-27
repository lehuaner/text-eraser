"""M6 · k_v Pareto 标定与冻结（规格 §7）。

流程：标定子集（基础网格 1/3 + 全部重点交叉轴）→ k_v∈{1.5,2,2.5,3,4} 全跑
→ 记录 (ε, 重建面积率) → Pareto 曲线 → 取 ε<0.1% 约束下重建面积最小的工作点。
ε 工作点不可达 → 触发预案 R6（禁静默放水，返回 None 由调用方上报）。

冻结产物 calib/frozen.json = {kv, tau_dir, k, k_lf, margin_mode, s_min,
gt_hash, pareto_curve, ...}。gt_hash 对**全集**用例计算——回归集/标定集都取自
同一数据集，任一生成轴改动即漂移 → 冻结失效（规格 §4 导入断言由
regression / 驱动脚本执行，pipeline 无 GT 集可比对）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from deglow.core.types import Prov

_FROZEN_PATH = Path(__file__).resolve().parent / "frozen.json"

# 规格 §4 冻结字段（key 用规格命名；pipeline 消费时做 tau_dir→tau1/tau2 映射）
_FROZEN_KEYS = ("kv", "tau_dir", "k", "k_lf", "margin_mode", "s_min",
                "veil_chroma", "veil_tau", "guard_sig", "guard_chroma",
                "guard_align")
_EPS_STAR = 0.001          # 契约目标 ε* = 0.1%


def hash_cases(cases) -> str:
    """复用 gt.generator 的用例集哈希。"""
    from deglow.gt.generator import hash_cases as _h
    return _h(cases)


def load_frozen(path: Path | None = None) -> dict | None:
    path = Path(path) if path else _FROZEN_PATH
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_case(case, kv: float, params: dict | None):
    from deglow.pipeline import run as run_pipe
    p = dict(params or {})
    p["kv"] = kv
    return run_pipe(case.image, carrier_mask=case.M_text, params=p)


def sweep_kv(cases, kv_list=(1.5, 2.0, 2.5, 3.0, 4.0), params=None,
             verbose=True) -> list[dict]:
    """k_v 网格全跑，返回 [(kv, ε, 重建面积率)] 逐工作点。

    ε         = 全标定子集逃逸像素 / 发光域像素（规格 §6 路由 ε，含守卫漏检）
    重建面积率 = (FILLED+SYNTH) / 发光域像素
    """
    from deglow.gt.metrics import evaluate_case, routing_epsilon
    points = []
    t0 = time.time()
    for kv in kv_list:
        eps_acc = 0.0
        glow_px = 0
        rebuilt_px = 0
        for ci, case in enumerate(cases):
            res = _run_case(case, kv, params)
            glow = case.glow_truth & ~case.M_text
            n = int(glow.sum())
            if n == 0:
                continue
            eps_acc += routing_epsilon(res.prov, np.clip(res.image, 0, 255)
                                       .astype(np.uint8),
                                       case.truth, glow) * n
            glow_px += n
            rebuilt_px += int(((res.prov == Prov.FILLED)
                               | (res.prov == Prov.SYNTH))[glow].sum())
        eps = eps_acc / max(glow_px, 1)
        ratio = rebuilt_px / max(glow_px, 1)
        points.append({"kv": float(kv), "epsilon": round(eps, 6),
                       "rebuilt_ratio": round(ratio, 6),
                       "n_cases": len(cases), "glow_px": glow_px})
        if verbose:
            print(f"  kv={kv:<4} ε={eps:.4%}  重建面积率={ratio:.3%}  "
                  f"({time.time() - t0:.0f}s)")
    return points


def workpoint(points, eps_star: float = _EPS_STAR):
    """ε<ε* 约束下重建面积最小的工作点；不可达返回 None（R6）。"""
    feasible = [p for p in points if p["epsilon"] <= eps_star]
    if not feasible:
        return None
    return min(feasible, key=lambda p: p["rebuilt_ratio"])


def freeze(full_cases, points, work_pt, params: dict | None = None,
           path: Path | None = None, note: str = "") -> Path:
    """写入 frozen.json（含 gt_hash 与 Pareto 曲线）。"""
    gt_hash = hash_cases(full_cases)
    frozen = {
        "version": 1,
        "gt_hash": gt_hash,
        "gt_size": full_cases[0].size if full_cases else None,
        "gt_n": len(full_cases),
        "kv": work_pt["kv"],
        "tau_dir": [params.get("tau1", 0.5), params.get("tau2", 0.7)]
        if params else [0.5, 0.7],
        "k": params.get("k", 3.0) if params else 3.0,
        "k_lf": params.get("k_lf", 4.0) if params else 4.0,
        "margin_mode": 1.5,
        "s_min": params.get("s_min", 1.5) if params else 1.5,
        "veil_chroma": params.get("veil_chroma", 20.0) if params else 20.0,
        "veil_tau": params.get("veil_tau", 0.6) if params else 0.6,
        "guard_sig": params.get("guard_sig", 0.35) if params else 0.35,
        "guard_chroma": params.get("guard_chroma", 15.0) if params else 15.0,
        "guard_align": params.get("guard_align", 0.5) if params else 0.5,
        "pareto_curve": points,
        "note": note,
    }
    path = Path(path) if path else _FROZEN_PATH
    with open(path, "w", encoding="utf-8") as f:
        json.dump(frozen, f, ensure_ascii=False, indent=2)
    return path


def check_frozen(frozen: dict, full_cases) -> tuple[bool, str]:
    """gt_hash 一致性断言（规格 §4）：回归/标定前校验，防参数漂移。"""
    cur = hash_cases(full_cases)
    if frozen.get("gt_hash") != cur:
        return False, (f"gt_hash 漂移: frozen={frozen.get('gt_hash')} "
                       f"≠ 当前={cur}（生成器或数据集已变，冻结失效）")
    return True, "gt_hash 一致"


def apply_frozen(frozen: dict, params: dict | None = None) -> dict:
    """把 frozen.json 参数并入 pipeline params（tau_dir → tau1/tau2）。"""
    p = dict(params or {})
    for k in _FROZEN_KEYS:
        if k in frozen and frozen[k] is not None:
            p[k] = frozen[k]
    td = frozen.get("tau_dir")
    if isinstance(td, (list, tuple)) and len(td) == 2:
        p["tau1"], p["tau2"] = float(td[0]), float(td[1])
    return p
