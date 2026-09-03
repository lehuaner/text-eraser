"""诊断 668「新」字蒙版覆盖不全：逐级导出 v2 管线的蒙版中间产物。

复现 eraser._erase_deglow_v2 的蒙版生成链：
  tmask(原图,tint=False) → clean=去发光 → tm_clean(clean,tint=True)
  → union → close3x3 → 方案B亮侧补全 → 膨胀edge=1
并在「新」字区域放大叠加：把最终 mask 没盖住的"亮/文字感"像素标红。
"""
import sys
import numpy as np
import cv2
from PIL import Image

ROOT = "D:/Code/Project/Python/TextEraser"
sys.path.insert(0, ROOT)
from text_eraser.text_select import detect_text_mask, _deglow_full_green_v2, _fill_bright_near_mask
from text_eraser.eraser import erase_text

rgb = np.array(Image.open(f"{ROOT}/data/_glowcheck/668.png").convert("RGB"))
H, W = rgb.shape[:2]
print("668 size:", rgb.shape)

# ---- 逐级复现 v2 蒙版链(与 _erase_deglow_v2 相同参数) ----
kw = dict(method="ml", q_off=55.0, max_area_ratio=0.40, max_box_ratio=0.40,
          max_side=1280, fill_white=True, fill_max_dist=12)
tmask, _ = detect_text_mask(rgb, tint_fill=False, **kw)
clean, core = _deglow_full_green_v2(
    rgb, tmask, strength=1.15, alpha_core=0.65,
    zone_ratio=0.6, zone_expand=24, protect_px=1, deglow_chroma_keep=False)
tm_clean, _ = detect_text_mask(clean, tint_fill=True, **kw)
union = ((tmask > 0) | (tm_clean > 0)).astype(np.uint8) * 255
closed = cv2.morphologyEx(union, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
bright = _fill_bright_near_mask(clean, closed)
mask_filled = cv2.dilate(bright, cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (3, 3)))  # edge=1

for name, m in [("tmask", tmask), ("tm_clean", tm_clean), ("union", union),
                ("closed", closed), ("brightB", bright), ("filled", mask_filled)]:
    print(f"{name:9s} px={int(np.count_nonzero(m))}")

# ---- 与 erase_text 全流程结果对照 ----
res, m_erase, meta = erase_text(
    rgb, deglow_scheme="v2", glow_mode="auto", deglow_mask_soft=0.0,
    edge=1, deglow_strength=1.15, fill_white=True, fill_max_dist=12,
    deglow_zone_ratio=0.6, deglow_zone_expand=24,
    return_mask=True, tint_fill=True)
print("erase meta mask_pix:", meta.get("mask_pix"), "boxes:", meta.get("boxes"))
Image.fromarray(res).save(f"{ROOT}/data/_glowcheck/_xin_full_result.png")
Image.fromarray(m_erase).save(f"{ROOT}/data/_glowcheck/_xin_erase_mask.png")

# ---- 「新」字区域放大: 漏覆盖像素标红 ----
# 求文字 bbox(取并集蒙版范围再外扩)
ys, xs = np.nonzero(union)
y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
pad = 30
y0, y1 = max(0, y0 - pad), min(H, y1 + pad)
x0, x1 = max(0, x0 - pad), min(W, x1 + pad)
print("xin bbox:", (x0, y0, x1, y1))

gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
r, g, b = rgb[..., 0].astype(np.int16), rgb[..., 1].astype(np.int16), rgb[..., 2].astype(np.int16)
# 「文字感」像素定义(笔画核心, 供人工核对):
#   原图近白: gray >= 170 且 min_rgb >= 140 (光晕是绿色, 绿度门排除)
cclean = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY).astype(np.float32)
min_rgb = np.minimum(np.minimum(r, g), np.minimum(r, b))
text_like = ((gray >= 170) & (min_rgb >= 140) &
             ((g - np.maximum(r, b)) < 26))
miss = text_like & (mask_filled == 0)
print(f"text_like={int(text_like.sum())} miss={int(miss.sum())}")

def zoom_overlay(base_img, over_mask, title_pad=True):
    crop = base_img[y0:y1, x0:x1].copy()
    ov = over_mask[y0:y1, x0:x1] > 0
    crop[ov] = [255, 0, 0]
    z = 5
    return cv2.resize(crop, (crop.shape[1] * z, crop.shape[0] * z),
                      interpolation=cv2.INTER_NEAREST)

Image.fromarray(zoom_overlay(rgb, miss)).save(f"{ROOT}/data/_glowcheck/_xin_miss_on_orig.png")
Image.fromarray(zoom_overlay(clean, miss)).save(f"{ROOT}/data/_glowcheck/_xin_miss_on_clean.png")
# mask 本体 + 蒙版边界 vs 文字感
Image.fromarray(zoom_overlay(np.stack([(mask_filled > 0).astype(np.uint8)] * 3, -1) * 180,
                             miss)).save(f"{ROOT}/data/_glowcheck/_xin_mask_with_miss.png")
# 分级对比图
def side(side_masks):
    tiles = []
    for label, m in side_masks:
        tiles.append(zoom_overlay(rgb, m))
    hmax = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0,
                                cv2.BORDER_CONSTANT, value=(30, 30, 30)) for t in tiles]
    return np.hstack(tiles)

grid = side([("tmask", tmask), ("tm_clean", tm_clean), ("union", union)])
Image.fromarray(grid).save(f"{ROOT}/data/_glowcheck/_xin_stages1.png")
grid2 = side([("closed", closed), ("brightB", bright), ("filled+miss", mask_filled)])
Image.fromarray(grid2).save(f"{ROOT}/data/_glowcheck/_xin_stages2.png")
print("saved diag images")
