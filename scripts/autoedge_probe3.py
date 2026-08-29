"""验证 auto_edge：79188 应自动选 2，其余图应保持在 1。"""
import sys, json
import numpy as np
import cv2

sys.path.insert(0, ".")
from textpatch.eraser import erase_text

DEFAULTS = dict(
    q_off=55.0, max_area_ratio=0.40, max_box_ratio=0.40,
    ml_max_side=960, direction=None, edge_aware=False,
    tint_fill=True, fill_white=True, fill_max_dist=12,
    glow_mode="auto", deglow_strength=1.0, deglow_green_thr=6.0,
    deglow_range=24, deglow_glo=85.0, deglow_protect=1.0,
    deglow_mask_soft=0.0, deglow_scheme="channel",
)


def load_params(meta_path):
    p = dict(DEFAULTS)
    try:
        m = json.load(open(meta_path, encoding="utf-8"))
        for k in p:
            if k in m.get("params", {}):
                p[k] = m["params"][k]
    except Exception:
        pass
    return p


ids = ["1787765979188", "1787765716464", "1787766251689"]
for tid in ids:
    bgr = cv2.imread(f"data/_hist3_now/{tid}.png")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    params = load_params(f"data/history/{tid}/meta.json")
    res, mask, meta = erase_text(rgb, edge=1, auto_edge=True, return_mask=True, **params)
    print(f"{tid}: auto_edge -> edge_used={meta.get('edge_used')}  "
          f"mask_filled_pix={meta.get('mask_filled_pix')}")
    # 对照：固定 edge
    for e in (1, 2):
        _, _, m2 = erase_text(rgb, edge=e, return_mask=True, **params)
        print(f"    fixed edge={e} -> mask_filled_pix={m2.get('mask_filled_pix')}")
    cv2.imwrite(f"data/_diag_autoedge/{tid}_auto.png", cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
print("done")
