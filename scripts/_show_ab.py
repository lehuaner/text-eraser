"""临时出图: 本轮 A+B 改动的前后对比(合成图 + 真实 low-res 图)。"""
import cv2
import numpy as np

from textpatch.text_select import detect_text_mask, _deglow_full_green_v2
from textpatch.eraser import _erase_deglow_v2, _run_fill


def load(p):
    return cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def det(rgb, tint_fill=True, **kw):
    return detect_text_mask(rgb, method="ml", tint_fill=tint_fill,
                            max_area_ratio=0.40, max_box_ratio=0.40,
                            **kw)


def red_overlay(img, mask, alpha=0.55):
    out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.float32)
    m = (mask > 0)
    out[..., 2][m] = out[..., 2][m] * (1 - alpha) + 255 * alpha
    return out.clip(0, 255).astype(np.uint8)


def hstack(*imgs):
    ims = [cv2.cvtColor(i, cv2.COLOR_RGB2BGR) if i.ndim == 3 else
           cv2.cvtColor(i, cv2.COLOR_GRAY2BGR) for i in imgs]
    h = max(i.shape[0] for i in ims)
    ims = [cv2.resize(i, (int(i.shape[1] * h / i.shape[0]), h)) for i in ims]
    return cv2.hconcat(ims)


def save(name, bgr):
    cv2.imwrite(name, bgr)
    print("saved:", name)


def label_bar(n, labels, bg=20):
    """顶部标签条: 按面板宽度均分, 白字标注。"""
    cols = [labels[i] for i in range(n)]
    w = 400 * n
    bar = np.full((26, w, 3), bg, np.uint8)
    for i, lab in enumerate(cols):
        cv2.putText(bar, lab, (i * 400 + 12, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return bar


def main():
    # 合成图: 3 面板, 统一放大到每个 400px 宽
    H, W = 46, 46
    img = np.full((H, W, 3), 40, np.uint8)
    cv2.rectangle(img, (12, 8), (16, 32), (235, 238, 230), -1)
    cv2.rectangle(img, (8, 8), (24, 9), (235, 238, 230), -1)
    cv2.rectangle(img, (20, 8), (21, 13), (235, 238, 230), -1)
    boxes = [{"x0": 2, "y0": 2, "x1": W - 2, "y1": H - 2}]
    from textpatch.text_select import _detect_text_mask_classic, _fill_bright_near_mask
    m_off = _detect_text_mask_classic(img, boxes=boxes, upscale=False)
    m_a = _detect_text_mask_classic(img, boxes=boxes, upscale=True)
    m_ab = _fill_bright_near_mask(img, m_a)
    src = cv2.resize(img, (400, 400), interpolation=cv2.INTER_NEAREST)
    p1 = cv2.resize(red_overlay(img, m_off), (400, 400),
                    interpolation=cv2.INTER_NEAREST)
    p2 = cv2.resize(red_overlay(img, m_ab), (400, 400),
                    interpolation=cv2.INTER_NEAREST)
    row = hstack(src, p1, p2)
    bar = label_bar(3, ["SYNTH ORIG", "OLD MASK", "NEW MASK  A+B"])
    save("data/_ab_synth.png", cv2.vconcat([bar, row]))

    # 真实 low-res 图(668 '新'): 上排 原/旧蒙版/新蒙版, 下排 旧结果/新结果/差异
    rgb = load("data/history/1787834009668/thumb.png")
    tmask_old, _ = det(rgb, tint_fill=False, upscale=False)
    clean_old, _ = _deglow_full_green_v2(rgb, tmask_old, strength=1.15,
                                         zone_ratio=0.6, zone_expand=24)
    tc_old, _ = det(clean_old, upscale=False)
    m_old = ((tmask_old > 0) | (tc_old > 0)).astype(np.uint8) * 255
    res_old, mf_old, _ = _run_fill(clean_old, m_old, [], edge=1,
                                   direction=None, edge_aware=False,
                                   return_mask=True, t0=0.0)
    res_new, mf_new, _ = _erase_deglow_v2(
        rgb, edge=1, q_off=55, max_area_ratio=0.40, max_box_ratio=0.40,
        ml_max_side=960, direction=None, edge_aware=False,
        return_mask=True, deglow_strength=1.15, alpha_core=0.65,
        deglow_zone_ratio=0.6, deglow_zone_expand=24, soft_expand=0.0)
    up = 3
    src = cv2.resize(rgb, (rgb.shape[1] * up, rgb.shape[0] * up),
                     interpolation=cv2.INTER_NEAREST)
    o1 = cv2.resize(red_overlay(rgb, mf_old), (rgb.shape[1] * up,
                                               rgb.shape[0] * up),
                    interpolation=cv2.INTER_NEAREST)
    o2 = cv2.resize(red_overlay(rgb, mf_new), (rgb.shape[1] * up,
                                               rgb.shape[0] * up),
                    interpolation=cv2.INTER_NEAREST)
    r1 = cv2.resize(res_old, (rgb.shape[1] * up, rgb.shape[0] * up),
                    interpolation=cv2.INTER_NEAREST)
    r2 = cv2.resize(res_new, (rgb.shape[1] * up, rgb.shape[0] * up),
                    interpolation=cv2.INTER_NEAREST)
    diff = np.max(cv2.absdiff(res_old.astype(np.int16),
                              res_new.astype(np.int16)), axis=2).astype(np.uint8)
    diff = cv2.resize(diff, (rgb.shape[1] * up, rgb.shape[0] * up),
                      interpolation=cv2.INTER_NEAREST)
    diff3 = cv2.cvtColor(cv2.applyColorMap(diff, cv2.COLORMAP_JET),
                         cv2.COLOR_BGR2RGB)
    row0 = hstack(src, o1, o2)
    row1 = hstack(r1, r2, diff3)
    bar0 = cv2.resize(label_bar(3, ["ORIG 668", "OLD MASK",
                                    "NEW MASK (A+B)"], 24),
                      (row0.shape[1], 26), interpolation=cv2.INTER_AREA)
    bar1 = cv2.resize(label_bar(3, ["OLD RESULT", "NEW RESULT",
                                    "DIFF (jet)"], 20),
                      (row1.shape[1], 26), interpolation=cv2.INTER_AREA)
    save("data/_ab_thumb668.png", cv2.vconcat([bar0, row0, bar1, row1]))


if __name__ == "__main__":
    main()