# -*- coding: utf-8 -*-
"""端到端 parity 验收：浏览器流(wasm erase_text_glyphs) vs 后端(编排式)。

⚠️ 两个历史教训（写死在脚本里防再犯）:
1. edge 必须用后端 auto_edge 的选择(meta["edge_used"]), 不能硬编码 ——
   auto_edge 会按图选 1 或 2, 传错 edge 会产生上千像素的假分歧
   (1787980309628 实测)。
2. 手工复刻后端链路时, CLOSE 必须走 _cv shim(dilate(erode), 交换语义),
   不能用真 cv2.morphologyEx(正确语义) —— 两者差 ~80px(1787980309628)。

用法: python scripts/parity_check.py [iid ...]
不传 iid 则跑默认四图。
"""
import sys
import numpy as np
import cv2

ROOT = r"D:/Code/Project/Python/TextPatch"
sys.path.insert(0, ROOT)
import text_eraser.eraser as er
from text_eraser import _shared_core

DEFAULT_IIDS = ["1787767556635", "1787767611178", "1787822778556", "1787980309628"]
KW = dict(edge=1, auto_edge=True, auto_max_edge=2, q_off=55.0, max_area_ratio=0.4,
          max_box_ratio=0.4, direction=None, edge_aware=False, return_mask=True,
          tint_fill=True, fill_white=True, fill_max_dist=12, glow_mode="auto",
          deglow_scheme="v2", deglow_strength=1.0, deglow_zone_ratio=0.6,
          deglow_zone_expand=10, deglow_protect_px=1, deglow_chroma_keep=True)
DET = dict(q_off=55.0, max_area_ratio=0.4, max_box_ratio=0.4, tint_fill=False,
           fill_white=True, fill_max_dist=12)


def check(iid):
    bgr = cv2.imread(rf"{ROOT}/data/history/{iid}/orig.bin", cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res_be, mask_be, meta = er.erase_text(rgb, **KW)
    edge = int(meta.get("edge_used", 1))
    tmask, _ = er.detect_text_mask(rgb, method="ml", **DET)
    clean_t0, _, _ = _shared_core.deglow_full_green_v2(
        rgb, tmask, strength=1.15, zone_ratio=0.6, zone_expand=10,
        protect_px=1, chroma_keep=1)
    tm_clean, _ = er.detect_text_mask(clean_t0, method="ml",
                                      **{**DET, "tint_fill": True})
    result_rs, fill_rs, clean_rs, _ = _shared_core.erase_text_glyphs(
        rgb, tmask, tm_clean, strength=1.15, zone_ratio=0.6, zone_expand=10,
        protect_px=1, chroma_keep=1, edge=edge, direction_deg=-1.0, seed=0,
        edge_aware=0, soft_expand=0.0)
    rd = np.abs(res_be.astype(np.int16) - result_rs.astype(np.int16)).sum(2)
    md = int(((mask_be > 0) ^ (fill_rs > 0)).sum())
    ok = int((rd > 0).sum()) == 0 and md == 0
    print(f"{iid}: edge_used={edge}  RESULT diff={int((rd > 0).sum())}  "
          f"MASK diff={md}  {'OK' if ok else 'MISMATCH'}")
    return ok


if __name__ == "__main__":
    iids = sys.argv[1:] or DEFAULT_IIDS
    all_ok = all(check(i) for i in iids)
    print("ALL PARITY OK" if all_ok else "PARITY FAILURE")
    sys.exit(0 if all_ok else 1)
