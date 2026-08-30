"""方案B「预算耗尽未收敛」判据实验:
方案B(白字AA环带补全)从蒙版沿亮候选生长 ≤rounds 轮。真 AA 环带宽 1~3px,
6 轮内被吃干净(收敛); 背景亮纹理(石纹亮带/光斑)超过 6px 厚, 预算耗尽时
紧邻仍未吸收的候选还在(未收敛)。

判据: 生长结束后, 若某候选连通块仍与已吸收区相邻(还想长但没预算) →
该连通块不是 AA 环带, 回退其全部新增。

验证: 7 张图(武器 + 6 基线)上量:
  A. 每图方案B新增 px, 其中「未收敛连通块」占多少;
  B. 基线图上被回退的像素是否伤召回(668 两横/白块走的是吸收步, 方案B只管AA环带)。
"""
import sys
import numpy as np
import cv2
from PIL import Image

ROOT = "D:/Code/Project/Python/TextEraser"
sys.path.insert(0, ROOT)
from text_eraser.text_select import (detect_text_mask, _deglow_full_green_v2,
                              _fill_bright_near_mask)

IMGS = [
    ("武器1787", f"{ROOT}/data/history/1787767429309/orig.bin"),
    ("178",     f"{ROOT}/data/_glowcheck/178.png"),
    ("556",     f"{ROOT}/data/_glowcheck/556.png"),
    ("635",     f"{ROOT}/data/_glowcheck/635.png"),
    ("668",     f"{ROOT}/data/_glowcheck/668.png"),
    ("换装",    f"{ROOT}/data/_glowcheck/_huanzhang_new.png"),
    ("展台",    f"{ROOT}/data/_glowcheck/_s4462.png"),
]

def schemeB_traced(rgb, mask, bg_lo=25, lum_off=24, min_rgb=118,
                   green_gate=26, rounds=6):
    """方案B + 记账: 返回 (生长后mask, 新增bool, 未收敛连通块bool)。"""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    outside = (mask == 0)
    bg = float(np.percentile(gray[outside], bg_lo)) if outside.sum() else 90.0
    r = rgb[..., 0].astype(np.int16); g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    min_rgb_im = np.minimum(np.minimum(r, g), b)
    cand = ((gray > (bg + lum_off)) & (min_rgb_im >= min_rgb) &
            ((g - np.maximum(r, b)) < green_gate))
    if not cand.any():
        return mask, np.zeros_like(cand), np.zeros_like(cand)
    cur = (mask > 0).astype(np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    for _ in range(rounds):
        dil = cv2.dilate(cur, k3) > 0
        add = dil & cand & (cur == 0)
        if not add.any():
            break
        cur[add] = 1
    grown = np.where(cur > 0, 255, 0).astype(np.uint8)
    added = grown.astype(bool) & ~mask.astype(bool)
    # 未收敛检测: 已吸收区的 1px 膨胀仍压着候选 → 预算耗尽还想长
    frontier = (cv2.dilate(grown, k3) > 0) & cand & ~grown.astype(bool)
    unconverged = np.zeros_like(cand)
    if frontier.any() and added.any():
        # 把「已吸收∪前沿」做连通域, 找同时含两者的块
        seed = (grown | (frontier.astype(np.uint8) * 255))
        n, lab = cv2.connectedComponents((seed > 0).astype(np.uint8), 8)
        add_lab = set(np.unique(lab[added])) - {0}
        fr_lab = set(np.unique(lab[frontier])) - {0}
        bad = add_lab & fr_lab
        if bad:
            unconverged = np.isin(lab, list(bad)) & added
    return grown, added, unconverged

print(f"{'图':<8} {'union前':>7} {'+方案B':>7} {'新增':>6} {'未收敛':>7} {'回退后':>7}")
for tag, path in IMGS:
    rgb = np.array(Image.open(path).convert("RGB"))
    tmask, _ = detect_text_mask(rgb, method="ml", q_off=55.0,
                                max_area_ratio=0.4, max_box_ratio=0.4,
                                max_side=960, tint_fill=False,
                                fill_white=True, fill_max_dist=12)
    clean, _, zone = _deglow_full_green_v2(
        rgb, tmask, strength=1.15, zone_ratio=0.6, zone_expand=10,
        protect_px=1, deglow_chroma_keep=False, return_zone=True)
    tm_clean, _ = detect_text_mask(clean, method="ml", q_off=55.0,
                                   max_area_ratio=0.4, max_box_ratio=0.4,
                                   max_side=960, tint_fill=True,
                                   fill_white=True, fill_max_dist=12)
    mask = ((tmask > 0) | (tm_clean > 0)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    grown, added, unconv = schemeB_traced(clean, mask)
    kept = added & ~unconv
    print(f"{tag:<8} {int((mask>0).sum()):>7} {int((mask>0).sum()+added.sum()):>7} "
          f"{int(added.sum()):>6} {int(unconv.sum()):>7} "
          f"{int((mask>0).sum()+kept.sum()):>7}")
