# -*- coding: utf-8 -*-
"""接缝修复回归: 全部 history 图, git HEAD(旧) vs 工作区(新) 代码逐张对比。

用每张图 meta.json 里保存的前端参数完整跑 erase_text(v2), 对比:
  - deglow 中间图(去发光结果)逐像素差
  - 最终 mask 像素数
  - 最终结果图逐像素差
旧代码从 git HEAD 提取为独立模块, 经 monkeypatch 注入 eraser。
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OLD_PATH = ROOT / "data" / "_diag556seam" / "_text_select_old.py"
# 基线 = main 合并基(本分支改动之前的原始代码), 而不是分支自己的 HEAD
_merge_base = subprocess.run(
    ["git", "merge-base", "main", "HEAD"], cwd=ROOT,
    capture_output=True, text=True).stdout.strip()
OLD_PATH.write_text(
    subprocess.run(["git", "show", f"{_merge_base}:core/text_select.py"], cwd=ROOT,
                   capture_output=True, text=True, encoding="utf-8").stdout,
    encoding="utf-8")

import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("text_select_old", OLD_PATH)
old_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_mod)

import textpatch.eraser as eraser  # noqa: E402
new_fn = eraser._deglow_full_green_v2

PARAM_KEYS = ["edge", "q_off", "max_area_ratio", "max_box_ratio", "direction",
              "edge_aware", "glow_mode", "deglow_strength", "deglow_green_thr",
              "deglow_range", "deglow_glo", "deglow_protect", "deglow_mask_soft",
              "deglow_zone_ratio", "deglow_zone_expand", "deglow_protect_px",
              "deglow_chroma_keep", "deglow_scheme", "fill_white",
              "fill_max_dist", "auto_edge", "auto_max_edge"]


def run_one(rgb, params):
    kw = {k: params[k] for k in PARAM_KEYS if k in params}
    res, mask, meta = eraser.erase_text(rgb, return_mask=True, **kw)
    return res, mask, meta


def diffstats(a, b):
    d = cv2.absdiff(a, b).max(axis=2)
    return int((d > 2).sum()), float(d.mean()), int(d.max())


hdr = (f"{'图':<28s} {'mask旧':>7s} {'mask新':>7s} | "
       f"{'deglow差px':>9s} {'均值':>6s} {'max':>4s} | "
       f"{'结果差px':>9s} {'均值':>6s}")
print(hdr)
print("-" * len(hdr * 2))
worst = []
for d in sorted((ROOT / "data" / "history").iterdir()):
    meta_p = d / "meta.json"
    if not meta_p.exists():
        continue
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    raw = (d / "orig.bin").read_bytes()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    p = meta["params"]

    res_n, _, meta_n = run_one(rgb, p)
    eraser._deglow_full_green_v2 = old_mod._deglow_full_green_v2
    try:
        res_o, _, meta_o = run_one(rgb, p)
    finally:
        eraser._deglow_full_green_v2 = new_fn

    dn, dm, dx = diffstats(meta_o["deglow_img"], meta_n["deglow_img"])
    rn, rm, rx = diffstats(res_o, res_n)
    name = meta.get("name", d.name)
    mp_o, mp_n = meta_o["mask_pix"], meta_n["mask_pix"]
    flag = ""
    if mp_n < mp_o:
        flag = "  ← mask 减少!"
    elif dn > 50 or rn > 50:
        flag = "  ← 检查"
    print(f"{name:<28s} {mp_o:>7d} {mp_n:>7d} | "
          f"{dn:>9d} {dm:>6.2f} {dx:>4d} | {rn:>9d} {rm:>6.2f}{flag}")
    worst.append((dn + rn, name))

# 556 专项: 保存新旧去发光图对比
d556 = ROOT / "data" / "history" / "1787822778556"
meta = json.loads((d556 / "meta.json").read_text(encoding="utf-8"))
raw = (d556 / "orig.bin").read_bytes()
rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR),
                   cv2.COLOR_BGR2RGB)
res_n, _, meta_n = run_one(rgb, meta["params"])
res_o, _, meta_o = run_one(rgb, {**meta["params"]})
print(f"\n556 deglow 新旧一致? {np.array_equal(meta_o['deglow_img'], meta_n['deglow_img'])}")
