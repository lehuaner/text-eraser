"""v4 工程标定驱动（M0 → M2 → M6 流水线）。

用法：
  py scripts/v4_calib.py [--size 384] [--limit 24] [--kv 1.5,2.0,2.5,3.0,4.0]
                         [--skip-det] [--n-contract 20] [--no-regression]
                         [--force-freeze] [--real]

流程：
  M0   GT 生成器自检（合成 vs 独立解析式 < 1e-3）
  M0   检测器确定性契约（同输入 n 次逐位一致）
  M2   HSL 扫描轴模式判别准确率（报告，信息性）
  M6   标定子集 → k_v Pareto 网格 → ε<0.1% 约束取重建面积最小工作点 → freeze
       工作点不可达 → 触发 R6 预案（默认不静默冻结；--force-freeze 才写）
  回归 acceptance 子集 → 分层指标 JSON（n_blocking>0 → 标红）

注意：--limit/--size 只用于交互验证；权威冻结须在夜间以全集跑（不带 limit）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deglow.calib import pareto                     # noqa: E402
from deglow.gt import generator as gtgen            # noqa: E402
from deglow.gt import metrics as gtmetrics          # noqa: E402


def m0_selfcheck(size: int):
    print("\n=== M0 · GT 生成器自检（<1e-3）===")
    worst = gtgen.self_check(size=max(128, size // 2), verbose=True)
    if worst >= 1e-3:
        print("M0 FAIL"); sys.exit(1)
    print("M0 PASS")


def m0_det_contract(n: int):
    print(f"\n=== M0 · 检测器确定性契约（{n} 次逐位一致）===")
    from deglow.regression.det_contract import test_det_contract
    ok = test_det_contract(n=n, verbose=True)
    if not ok:
        print("det contract FAIL"); sys.exit(1)
    print("det contract PASS")


def m2_mode_accuracy(cases):
    """HSL 扫描轴模式判别准确率（信息性，M2 验收门 ≥98%）。"""
    print("\n=== M2 · 模式判别准确率（门 ≥98%，信息性）===")
    from deglow.pipeline import run as run_pipe
    n_ok = n_total = 0
    for c in cases:
        res = run_pipe(c.image, carrier_mask=c.M_text)
        modes = [d["mode"] for d in res.report.get("domains", [])
                 if d.get("pix", 0) > 0]
        if res.has_glow and modes:
            n_total += 1
            n_ok += int(c.mode in modes or "unknown" in modes)
    acc = n_ok / max(n_total, 1)
    print(f"  可辨识域 {n_total}，判别正确 {n_ok}，准确率 {acc:.2%}"
          f"（门 ≥98%）")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--limit", type=int, default=24,
                    help="交互验证用标定子集大小（None=全集）")
    ap.add_argument("--kv", default="1.5,2.0,2.5,3.0,4.0")
    ap.add_argument("--skip-det", action="store_true")
    ap.add_argument("--n-contract", type=int, default=100)
    ap.add_argument("--no-regression", action="store_true")
    ap.add_argument("--force-freeze", action="store_true",
                    help="ε 工作点不可达时仍写 frozen.json（默认走 R6 不上报）")
    ap.add_argument("--reg-limit", type=int, default=24)
    ap.add_argument("--real", action="store_true", help="追加真实用例集报告")
    args = ap.parse_args()

    kv_list = [float(x) for x in args.kv.split(",")]
    t_all = time.time()

    m0_selfcheck(args.size)
    if not args.skip_det:
        m0_det_contract(args.n_contract)

    # 全集（hash 载体）+ 子集
    print(f"\n=== 用例集（size={args.size}）===")
    full = gtgen.full_cases(args.size)
    gh = gtgen.hash_cases(full)
    print(f"  全集 {len(full)} 例，gt_hash={gh}")

    m2_mode_accuracy(full[: min(48, len(full))])

    print(f"\n=== M6 · k_v Pareto 标定（{kv_list}）===")
    calib_cases = gtgen.iter_cases(args.size, subset="calib",
                                   limit=args.limit)
    print(f"  标定子集 {len(calib_cases)} 例")
    points = pareto.sweep_kv(calib_cases, kv_list=kv_list, verbose=True)
    wp = pareto.workpoint(points)
    if wp is None:
        eps_min = min(p["epsilon"] for p in points)
        print(f"  [R6 预案] ε 工作点不可达（ε_min={eps_min:.4%} > 0.1%）"
              f"——按规格收紧校验门重标或上报重定 ε*，禁静默放水。")
        if not args.force_freeze:
            print("  未写 frozen.json（--force-freeze 可强制写入用于调试）。")
            return
    else:
        print(f"  工作点: kv={wp['kv']}  ε={wp['epsilon']:.4%}  "
              f"重建面积率={wp['rebuilt_ratio']:.3%}")
    fp = pareto.freeze(full, points, wp,
                       params=None,
                       note=("SMOKE: 交互验证冻结，需以全集夜间重跑"
                             if (args.limit or args.size != 512) else
                             "权威冻结"))
    print(f"  已写 {fp}")

    if not args.no_regression:
        print("\n=== 回归（acceptance 子集）===")
        from deglow.regression.run import run_regression, save_report
        frozen = pareto.load_frozen(fp)
        acc_cases = gtgen.iter_cases(args.size, subset="acceptance",
                                     limit=args.reg_limit)
        rep = run_regression(acc_cases, calib=frozen, verbose=True)
        rep_path = ROOT / "data" / "v4_regression_report.json"
        save_report(rep, rep_path)
        print(f"  n_cases={rep['n_cases']}  n_blocking={rep['n_blocking']}  "
              f"ε_all={rep['epsilon_all']:.4%}")
        print(f"  按模式通过率: {json.dumps(rep['by_mode'], ensure_ascii=False)}")
        print(f"  分层: {json.dumps(rep['layers'], ensure_ascii=False)}")
        print(f"  报告已存 {rep_path}")
        if rep["n_blocking"] > 0:
            print(f"  [阻断] 有 {rep['n_blocking']} 例未过门（详见报告 fails）")

    if args.real:
        print("\n=== 真实用例集（ε 代理趋势）===")
        from deglow.regression.run import run_real_cases
        frozen = pareto.load_frozen()
        real = run_real_cases([
            ROOT / "data" / "history" / "1787768178725" / "orig.bin",
            ROOT / "data" / "final" / "needExtractAndPatch_orig.png",
            ROOT / "data" / "_batch_orig.png",
        ], calib=frozen, verbose=True)
        (ROOT / "data" / "v4_real_report.json").write_text(
            json.dumps(real, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n总耗时 {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
