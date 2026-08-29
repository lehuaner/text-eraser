"""方案B 新增连通块特征表: 找「AA环带(该吃) vs 背景亮纹理(不该吃)」的判据。

每张图: 跑到并集蒙版 → 方案B生长(带记账) → 对每个新增连通块量:
  area        新增面积
  thick       厚度(能被3x3腐蚀几次还有剩余, +1)
  frontier    预算耗尽后紧邻未吸收候选 px
  cc_total    该块所在「全量候选连通块」总面积(不限预算)
  reach       候选块超出已吸收区的最大测地距离 px
  gray_p50    新增像素灰度中位
  lbg         新增像素周边(21窗, 非候选)灰度中位 → 局部对比
"""
import sys
import numpy as np
import cv2
from PIL import Image

ROOT = "D:/Code/Project/Python/TextPatch"
sys.path.insert(0, ROOT)
from textpatch.text_select import detect_text_mask, _deglow_full_green_v2

IMGS = [
    ("武器1787", f"{ROOT}/data/history/1787767429309/orig.bin"),
    ("178",     f"{ROOT}/data/_glowcheck/178.png"),
    ("556",     f"{ROOT}/data/_glowcheck/556.png"),
    ("635",     f"{ROOT}/data/_glowcheck/635.png"),
    ("668",     f"{ROOT}/data/_glowcheck/668.png"),
    ("换装",    f"{ROOT}/data/_glowcheck/_huanzhang_new.png"),
    ("展台",    f"{ROOT}/data/_glowcheck/_s4462.png"),
]

def union_mask(rgb):
    tmask, _ = detect_text_mask(rgb, method="ml", q_off=55.0, max_area_ratio=0.4,
                                max_box_ratio=0.4, max_side=960, tint_fill=False,
                                fill_white=True, fill_max_dist=12)
    clean, _, _ = _deglow_full_green_v2(rgb, tmask, strength=1.15, zone_ratio=0.6,
                                        zone_expand=10, protect_px=1,
                                        deglow_chroma_keep=False, return_zone=True)
    tmc, _ = detect_text_mask(clean, method="ml", q_off=55.0, max_area_ratio=0.4,
                              max_box_ratio=0.4, max_side=960, tint_fill=True,
                              fill_white=True, fill_max_dist=12)
    m = ((tmask > 0) | (tmc > 0)).astype(np.uint8) * 255
    return clean, cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

k3 = np.ones((3, 3), np.uint8)
hdr = f"{'图':<9}{'area':>5}{'thick':>6}{'front':>6}{'cc_tot':>7}{'reach':>6}{'gray':>5}{'lbg':>5}{'对比':>5}  bbox"
print(hdr)
for tag, path in IMGS:
    rgb = np.array(Image.open(path).convert("RGB"))
    clean, mask = union_mask(rgb)
    gray = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY).astype(np.float32)
    outside = (mask == 0)
    bg = float(np.percentile(gray[outside], 25))
    r = clean[..., 0].astype(np.int16); g = clean[..., 1].astype(np.int16)
    b = clean[..., 2].astype(np.int16)
    min_rgb_im = np.minimum(np.minimum(r, g), b)
    cand = ((gray > (bg + 24)) & (min_rgb_im >= 118) &
            ((g - np.maximum(r, b)) < 26))
    if not cand.any():
        continue
    cur = (mask > 0).astype(np.uint8)
    for _ in range(6):
        dil = cv2.dilate(cur, k3) > 0
        add = dil & cand & (cur == 0)
        if not add.any():
            break
        cur[add] = 1
    grown = cur.astype(bool)
    added = grown & ~mask.astype(bool)
    if not added.any():
        print(f"{tag:<9} (无新增)")
        continue
    frontier = (cv2.dilate(cur, k3) > 0) & cand & ~grown
    # 全量候选连通块(不受预算限制)
    n_cc, lab_cc = cv2.connectedComponents(cand.astype(np.uint8), 8)
    # 测地延展: 从已吸收区在候选块内走 24 步, 看还剩多远
    reach_map = grown.astype(np.uint8)
    for step in range(24):
        nxt = (cv2.dilate(reach_map, k3) > 0) & cand & ~mask.astype(bool)
        if (nxt.astype(np.uint8) == reach_map).all() if nxt.shape else True:
            pass
        new = nxt.astype(np.uint8) & ~reach_map
        if not new.any():
            break
        reach_map |= new
    extra_reach = (reach_map.astype(bool) & cand & ~grown)
    # 厚度
    thick = added.astype(np.uint8)
    tdepth = 0
    while True:
        er = cv2.erode(thick, k3)
        if not er.any():
            break
        thick = er; tdepth += 1
    n, lab, stats, cent = cv2.connectedComponentsWithStats(added.astype(np.uint8), 8)
    # 局部背景: 灰度高斯低通(排除 mask∪added 后的环形中位)简化: 非候选像素 31 窗中位
    inv = (~cand).astype(np.uint8)
    lbg_im = cv2.blur(inv * gray, (31, 31)) / np.maximum(cv2.blur(inv, (31, 31)), 1)
    print(f"-- {tag} 新增 {int(added.sum())}px, 未收敛前沿 {int(frontier.sum())}px, 候选块超延展 {int(extra_reach.sum())}px")
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 3:
            continue
        sel = lab == i
        cc_id = int(np.bincount(lab_cc[sel]).argmax())
        cc_tot = int((lab_cc == cc_id).sum())
        fr = int((sel & frontier).sum())
        er_thick = 0
        t = sel.astype(np.uint8)
        while True:
            e = cv2.erode(t, k3)
            if not e.any():
                break
            t = e; er_thick += 1
        # reach: 该 CC 的候选块里, 在 extra_reach 上的最大距离变换值
        cc_sel = (lab_cc == cc_id)
        dm = cv2.distanceTransform((~(grown | extra_reach)).astype(np.uint8),
                                   cv2.DIST_L2, 3)
        reach = float(dm[cc_sel].max()) if cc_sel.any() else 0.0
        gp = float(np.median(gray[sel]))
        lb = float(np.median(lbg_im[sel]))
        x, y, w, h = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3]
        print(f"{tag:<9}{int(sel.sum()):>5}{er_thick:>6}{fr:>6}{cc_tot:>7}"
              f"{reach:>6.1f}{gp:>5.0f}{lb:>5.0f}{gp-lb:>5.0f}  x{x:.0f} y{y:.0f} {w:.0f}x{h:.0f}")
