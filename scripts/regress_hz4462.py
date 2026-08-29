"""换装.png + 展台(4462) 修复回归(前端默认参数):
1. 换装: 不判发光(zone=0)、蒙版纯字形(无方案B过度生长)、黑碎块仍为 0(TELEA 自适应);
2. 展台4462: 不再误判发光(连通块门)、蒙版/结果正常;
3. 四发光样图: 发光判定不受影响、蒙版/残迹回归。
"""
import sys
import numpy as np
import cv2
from PIL import Image

ROOT = "D:/Code/Project/Python/TextPatch"
sys.path.insert(0, ROOT)
from core.eraser import erase_text
from core.text_select import detect_text_mask

def run(path):
    rgb = np.array(Image.open(path).convert("RGB"))
    res, m, meta = erase_text(
        rgb, deglow_scheme="v2", glow_mode="auto", deglow_mask_soft=0.0,
        edge=1, deglow_strength=1, fill_white=True, fill_max_dist=12,
        deglow_zone_ratio=0.6, deglow_zone_expand=10, deglow_protect_px=1,
        return_mask=True, tint_fill=True)
    gz = meta.get("glow_zone")
    zone_px = int((gz > 0).sum()) if gz is not None else 0
    return rgb, res, meta, zone_px, m

# 1) 换装
rgb, res, meta, zone_px, m = run(f"{ROOT}/data/_glowcheck/_huanzhang_new.png")
mp = meta["mask_pre_edge"]
n, lab, stats, _ = cv2.connectedComponentsWithStats((mp > 0).astype(np.uint8), 8)
comps = sorted(range(1, n), key=lambda i: -stats[i, 4])
print("换装: zone =", zone_px, "(应0)  mask连通块:",
      [(int(stats[i,0]), int(stats[i,1]), int(stats[i,4])) for i in comps[:4]])
g = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY).astype(np.float32)
ring = (cv2.dilate(m, np.ones((25, 25), np.uint8)) > 0) & (m == 0)
dark = int(((g < np.median(g[ring]) - 30) & (m > 0)).sum())
print(f"      洞内暗碎块: {dark}px (应0)")

# 2) 展台4462
rgb2, res2, meta2, zone2, m2 = run(f"{ROOT}/data/_glowcheck/_s4462.png")
mp2 = meta2["mask_pre_edge"]
n2, l2, st2, _ = cv2.connectedComponentsWithStats((mp2 > 0).astype(np.uint8), 8)
c2 = sorted(range(1, n2), key=lambda i: -st2[i, 4])
print("展台: zone =", zone2, "(应0)  mask连通块:",
      [(int(st2[i,0]), int(st2[i,1]), int(st2[i,4])) for i in c2[:5]])
Image.fromarray(cv2.resize(np.hstack([rgb2, res2]), None, fx=4, fy=4,
                           interpolation=cv2.INTER_NEAREST)
                ).save(f"{ROOT}/data/_glowcheck/_z4462_final.png")

# 3) 四发光样图
print(f"\n{'tag':>5} {'zone_px':>8} {'mask_pix':>9} {'resid_px':>9}")
for tag in ["178", "556", "635", "668"]:
    im = np.array(Image.open(f"{ROOT}/data/_glowcheck/{tag}.png").convert("RGB"))
    r3, m3, meta3, z3, _mm = run(f"{ROOT}/data/_glowcheck/{tag}.png")
    gres = cv2.cvtColor(r3, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gorg = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
    tm, _ = detect_text_mask(im, method="ml", tint_fill=False,
                             max_area_ratio=0.40, q_off=55,
                             fill_white=True, fill_max_dist=12)
    roi = np.zeros(im.shape[:2], bool)
    if tm.any():
        ys, xs = np.nonzero(tm)
        roi[max(0, ys.min()-30):ys.max()+30, max(0, xs.min()-30):xs.max()+30] = True
    resid = int(((gres > 130) & roi & (gorg < 110)).sum())
    print(f"{tag:>5} {z3:>8} {meta3['mask_pix']:>9} {resid:>9}")

# 4) 668 两横/白块/顶部小点专项
rgb4, res4, meta4, _, mm4 = run(f"{ROOT}/data/_glowcheck/668.png")
mp4 = meta4["mask_pre_edge"]
clean4 = meta4["deglow_img"]
g4 = cv2.cvtColor(clean4, cv2.COLOR_RGB2GRAY).astype(np.float32)
hole = np.zeros(rgb4.shape[:2], bool); hole[163:208, 138:205] = True
stroke = hole & (g4 > 100)
cov = int(((mp4 > 0) & stroke).sum())
white = np.zeros(rgb4.shape[:2], bool); white[171:178, 148:157] = True
top = (mp4 > 0); top[110:, :] = False
print(f"\n668 专项: 两横覆盖 {cov}/{int(stroke.sum())} ({cov/stroke.sum()*100:.0f}%), "
      f"白块 {int((mp4[white]>0).sum())}/63, 顶部小点 {int(top.sum())}px")
