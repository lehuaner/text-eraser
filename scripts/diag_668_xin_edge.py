"""诊断 668 文字蒙版两个问题:
A. 文字上方远离文字处的一小点蒙版: 逐级追溯(tmask/tm_clean/closed/方案B/吸收)首次出现位置;
B. 文字灰边保留较多(文字图层不完整): clean 上文字 AA 灰边的 min_rgb 分布,
   方案B 门限(min_rgb>=118)挡了多少, mask 外灰边有多少。
"""
import sys
import numpy as np
import cv2
from PIL import Image

ROOT = "D:/Code/Project/Python/TextPatch"
sys.path.insert(0, ROOT)
from textpatch.text_select import (detect_text_mask, _deglow_full_green_v2,
                              _fill_bright_near_mask, _absorb_zone_bright_core,
                              _fill_nearby_white, _grow_color_tint, _detect_text_mask_classic)

rgb = np.array(Image.open(f"{ROOT}/data/_glowcheck/668.png").convert("RGB"))
kw = dict(method="ml", q_off=55.0, max_area_ratio=0.40, max_box_ratio=0.40,
          max_side=1280, fill_white=True, fill_max_dist=12)
tmask, _ = detect_text_mask(rgb, tint_fill=False, **kw)
clean, _, zone = _deglow_full_green_v2(
    rgb, tmask, strength=1.15, alpha_core=0.65, zone_ratio=0.6,
    zone_expand=24, protect_px=1, deglow_chroma_keep=False, return_zone=True)
tm_clean, _ = detect_text_mask(clean, tint_fill=True, **kw)
union = ((tmask > 0) | (tm_clean > 0)).astype(np.uint8) * 255
closed = cv2.morphologyEx(union, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
bright = _fill_bright_near_mask(clean, closed)
absorbed = _absorb_zone_bright_core(clean, rgb, bright, zone)

# ---- A: 文字上方(y<112, 即浅色块分界附近/以上)的蒙版连通块追溯 ----
print("== A. 文字上方(y<112)的蒙版连通块 ==")
text_bbox_y0 = 116  # 主文字蒙版顶部
region = np.zeros(rgb.shape[:2], bool); region[:112, :] = True
for name, m in [("tmask", tmask), ("tm_clean", tm_clean), ("union", union),
                ("closed", closed), ("brightB", bright), ("absorbed", absorbed)]:
    top = (m > 0) & region
    if not top.any():
        print(f"  {name:9s}: 无")
        continue
    n, lab, stats, _ = cv2.connectedComponentsWithStats(top.astype(np.uint8), 8)
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        print(f"  {name:9s}: 连通块 (x={x},y={y},w={w},h={h},area={a})")

# ---- B: clean 上文字灰边 vs 方案B 门限 ----
print("\n== B. clean 上文字灰边分析 ==")
gclean = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY).astype(np.float32)
cr, cg, cb = clean[..., 0].astype(np.int16), clean[..., 1].astype(np.int16), clean[..., 2].astype(np.int16)
cmin = np.minimum(np.minimum(cr, cg), cb)
cgreen = cg - np.maximum(cr, cb)
# 文字灰边定义: clean 上亮于背景+24(方案B亮度门) 但在最终 mask 外
bg = float(np.percentile(gclean[absorbed == 0], 25))
outside = absorbed == 0
edge_band = outside & (gclean > bg + 24) & (gclean < 190)   # 亮但不到白核
print(f"bg(25分位)={bg:.0f}  mask外亮带(灰边候选)px={int(edge_band.sum())}")
if edge_band.any():
    print(f"  灰边 min_rgb 分布 p10/25/50/75/90 = {np.percentile(cmin[edge_band], [10,25,50,75,90]).round(0)}")
    blocked_minrgb = int((cmin[edge_band] < 118).sum())
    print(f"  被方案B近白门(min_rgb>=118)挡住: {blocked_minrgb}px ({blocked_minrgb/max(edge_band.sum(),1)*100:.0f}%)")
    blocked_dist = edge_band.copy()
    dist = cv2.distanceTransform((absorbed == 0).astype(np.uint8), cv2.DIST_L2, 3)
    far = int((dist[edge_band] > 6).sum())
    print(f"  距蒙版>6px(方案B轮数够不着): {far}px")
    both = int(((cmin[edge_band] < 118) & (dist[edge_band] > 6)).sum())
    print(f"  两道门都过不了(彻底救不回): {both}px")
# 文字图层不完整的直观量化: 文字 bbox 内 mask 外的"文字感"像素
ys, xs = np.nonzero(absorbed)
tb = np.zeros(rgb.shape[:2], bool); tb[ys.min():ys.max()+1, xs.min():xs.max()+1] = True
gray_edge = tb & outside & (gclean > 110)
print(f"  文字bbox内 mask外 亮于110 的灰边 px: {int(gray_edge.sum())}")

# 可视化: 灰边候选标红叠在 clean 上
vis = clean.copy()
vis[edge_band] = [255, 0, 0]
z = 3
Image.fromarray(cv2.resize(vis, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
                ).save(f"{ROOT}/data/_glowcheck/_xin_edgeband.png")
# 顶部小蒙版点可视化: absorbed 的 top 区域 + 主蒙版
vis2 = rgb.copy()
vis2[absorbed > 0] = [0, 200, 255]
vis2[(absorbed > 0) & region] = [255, 0, 255]
Image.fromarray(cv2.resize(vis2, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
                ).save(f"{ROOT}/data/_glowcheck/_xin_topmask.png")
print("saved _xin_edgeband.png (灰边候选红) / _xin_topmask.png (顶部蒙版点洋红)")
