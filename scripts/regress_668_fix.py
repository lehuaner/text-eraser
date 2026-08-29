"""方案A 修复回归: 178/556/635/668 四图 erase_text(v2) + 668 白块专项核对。
输出: 各图 mask_pix / 残迹像素数(结果图中亮于背景的像素, 限文字邻域深色区) + 668 对比图。
"""
import sys
import numpy as np
import cv2
from PIL import Image

ROOT = "D:/Code/Project/Python/TextPatch"
sys.path.insert(0, ROOT)
from text_eraser.eraser import erase_text
from text_eraser.text_select import detect_text_mask

def load(tag):
    return np.array(Image.open(f"{ROOT}/data/_glowcheck/{tag}.png").convert("RGB"))

print(f"{'tag':>5} {'mask_pix':>9} {'sec':>6} {'resid_px':>9}")
results = {}
for tag in ["178", "556", "635", "668"]:
    rgb = load(tag)
    res, m, meta = erase_text(
        rgb, deglow_scheme="v2", glow_mode="auto", deglow_mask_soft=0.0,
        edge=1, deglow_strength=1.15, fill_white=True, fill_max_dist=12,
        deglow_zone_ratio=0.6, deglow_zone_expand=24,
        return_mask=True, tint_fill=True)
    # 残迹: 文字邻域(原蒙版 bbox 外扩)内、结果亮度>130 且原图对应处不是浅色块背景
    gres = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gorg = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    tmask, _ = detect_text_mask(rgb, method="ml", tint_fill=False,
                                max_area_ratio=0.40, q_off=55,
                                fill_white=True, fill_max_dist=12)
    roi = np.zeros(rgb.shape[:2], bool)
    if tmask.any():
        ys, xs = np.nonzero(tmask)
        roi[max(0, ys.min()-30):ys.max()+30, max(0, xs.min()-30):xs.max()+30] = True
    resid = (gres > 130) & roi & (gorg < 110)   # 原图暗处(排除浅色块背景本身)
    results[tag] = (res, m, meta, resid)
    print(f"{tag:>5} {meta['mask_pix']:>9} {meta['inpaint_seconds']:>6} {int(resid.sum()):>9}")

# 668 专项: 白块区域核对
res, m, meta, resid = results["668"]
hole = np.zeros(res.shape[:2], bool); hole[171:178, 148:157] = True
print("\n668 白块(148..156,171..177): mask覆盖",
      f"{int((m[hole]>0).sum())}/{hole.sum()}",
      "结果灰度均值", round(float(cv2.cvtColor(res, cv2.COLOR_RGB2GRAY).astype(np.float32)[hole].mean()), 1))

# 对比图: 修复前(已有) | 修复后 | 修复后蒙版
before = np.array(Image.open(f"{ROOT}/data/_glowcheck/_xin_full_result.png").convert("RGB"))
z = 3
tiles = [before, res, np.stack([(m > 0).astype(np.uint8)] * 3, -1) * 180]
big = cv2.resize(np.hstack(tiles), None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
Image.fromarray(big).save(f"{ROOT}/data/_glowcheck/_xin_fix_compare.png")
# 放大红框区
vis = res.copy()
cv2.rectangle(vis, (148, 171), (156, 177), (255, 0, 0), 1)
Image.fromarray(np.hstack([before, vis])).save(f"{ROOT}/data/_glowcheck/_xin_fix_zoom.png")
print("saved _xin_fix_compare.png / _xin_fix_zoom.png")
