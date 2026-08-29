"""自动 edge 探针：验证 "残留文字分数" 能否区分 edge=1(差) 与 edge=2(好)。

核心思路（对齐用户标准「文字看不出来」）：
  擦除后，在原文字邻域内重新做文字检测；残留面积占原文字面积比例越低，说明擦得越干净。
  自动策略 = 从首选 edge(默认1) 起，逐个 +1 试，取第一个残留比例低于阈值的最小 edge。
"""
import sys, json, time
import numpy as np
import cv2

sys.path.insert(0, ".")
from text_eraser.eraser import erase_text, _ellipse
from text_eraser.text_select import detect_text_mask

DEFAULTS = dict(
    q_off=55.0, max_area_ratio=0.40, max_box_ratio=0.40,
    ml_max_side=960, direction=None, edge_aware=False,
    tint_fill=True, fill_white=True, fill_max_dist=12,
    glow_mode="auto", deglow_strength=1.0, deglow_green_thr=6.0,
    deglow_range=24, deglow_glo=85.0, deglow_protect=1.0,
    deglow_mask_soft=0.0, deglow_scheme="channel",
)


def load_params(meta_path):
    params = dict(DEFAULTS)
    try:
        m = json.load(open(meta_path, encoding="utf-8"))
        p = m.get("params", {})
        for k in ("q_off", "max_area_ratio", "max_box_ratio", "edge_aware",
                 "glow_mode", "deglow_strength", "deglow_green_thr",
                 "deglow_range", "deglow_glo", "deglow_protect",
                 "deglow_mask_soft", "deglow_scheme", "fill_white",
                 "fill_max_dist"):
            if k in p:
                params[k] = p[k]
    except Exception as e:
        print("  (meta read fail, use defaults:", e, ")")
    return params


def residual_score(rgb_orig, result, core_mask, params):
    """返回 (残留比例, 残留px, 原文字px)。"""
    core_pix = int((core_mask > 0).sum())
    if core_pix == 0:
        return 0.0, 0, 0
    edit_zone = cv2.dilate(core_mask, _ellipse(14)) > 0
    res_mask, _ = detect_text_mask(
        result, method="ml", q_off=params["q_off"],
        max_area_ratio=params["max_area_ratio"], max_box_ratio=params["max_box_ratio"],
        fill_white=params["fill_white"], fill_max_dist=params["fill_max_dist"],
        tint_fill=params["tint_fill"])
    residual = (res_mask > 0) & edit_zone
    rpix = int(residual.sum())
    return rpix / core_pix, rpix, core_pix


def main():
    ids = ["1787765979188", "1787765716464", "1787766251689"]
    for i, tid in enumerate(ids):
        src = f"data/_hist3_now/{tid}.png"
        try:
            bgr = cv2.imread(src)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(tid, "read fail", e); continue
        params = load_params(f"data/history/{tid}/meta.json")
        print(f"\n=== {tid}  size={rgb.shape[1]}x{rgb.shape[0]}  params.edge={params.get('edge')} ===")
        core, _ = detect_text_mask(
            rgb, method="ml", q_off=params["q_off"],
            max_area_ratio=params["max_area_ratio"], max_box_ratio=params["max_box_ratio"],
            fill_white=params["fill_white"], fill_max_dist=params["fill_max_dist"],
            tint_fill=params["tint_fill"])
        print(f"  core text pix = {int((core>0).sum())}")
        for edge in (1, 2, 3):
            t = time.time()
            res, mask, meta = erase_text(rgb, edge=edge, return_mask=True, **params)
            dt = time.time() - t
            ratio, rpix, cpix = residual_score(rgb, res, core, params)
            print(f"  edge={edge}  mask_filled_pix={meta.get('mask_filled_pix'):>7}  "
                  f"residual_ratio={ratio:6.3f}  res_pix={rpix:>6}  core_pix={cpix:>6}  ({dt:.2f}s)")
            cv2.imwrite(f"data/_diag_autoedge/{tid}_e{edge}.png",
                        cv2.cvtColor(res, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
