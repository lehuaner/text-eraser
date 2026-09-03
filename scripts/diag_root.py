"""座驾问题根因诊断: 看 DBNet 是否检出框, Otsu 是否保留白字身."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from text_eraser.text_select import to_rgb_uint8, detect_text_mask, detect_text, _detect_text_mask_classic
from text_eraser.ml_text_select import detect_text_mask_ml, _dbnet_infer, detect_text_ml, _get_session

P = ROOT / "data" / "needExtractAndPatch2.png"
rgb = to_rgb_uint8(Image.open(P))
H, W = rgb.shape[:2]
print(f"image = {W}x{H}, total={H*W}")

# 1) DBNet 直接 boxes
boxes_dbnet = detect_text(rgb, method="ml", max_area_ratio=0.40,
                          max_box_ratio=0.40, max_side=960, min_area=30)
print(f"\n[DBNet boxes] {len(boxes_dbnet)} boxes:")
for b in boxes_dbnet:
    area = (b["x1"]-b["x0"]) * (b["y1"]-b["y0"])
    print(f"  {b}  area={area}  ratio={area/(H*W):.3f}")

# 2) DBNet 直接 boxes (default max_area_ratio=0.05)
boxes_dbnet2 = detect_text(rgb, method="ml")
print(f"\n[DBNet boxes default] {len(boxes_dbnet2)} boxes:")
for b in boxes_dbnet2:
    area = (b["x1"]-b["x0"]) * (b["y1"]-b["y0"])
    print(f"  {b}  area={area}  ratio={area/(H*W):.3f}")

# 3) ML mask (DBNet prob map 直接)
mask_ml, boxes_ml = detect_text_mask_ml(rgb, min_area=30, max_area_ratio=0.40,
                                          mask_threshold=0.4, mask_max_side=1600)
mp_ml = int(mask_ml.sum()//255)
print(f"\n[ML mask] mask_pix={mp_ml}  boxes={len(boxes_ml)}")

# 4) Classic 路径 detect_text_mask (当前默认走的就是这个)
mask_otsu, boxes_otsu = detect_text_mask(rgb, method="ml", q_off=70.0,
                                          max_area_ratio=0.40, max_box_ratio=0.40)
mp_otsu = int(mask_otsu.sum()//255)
print(f"\n[Current ML+Otsu] mask_pix={mp_otsu}  boxes={len(boxes_otsu)}")

# 5) Pure classic (CV) boxes
boxes_classic = detect_text(rgb, method="classic", max_area_ratio=0.40, max_box_ratio=0.40)
print(f"\n[Classic boxes] {len(boxes_classic)} boxes:")
for b in boxes_classic:
    area = (b["x1"]-b["x0"]) * (b["y1"]-b["y0"])
    print(f"  {b}  area={area}  ratio={area/(H*W):.3f}")

# 6) 统计 DBNet 概率分布
prob, nw, nh, _, _, _ = _dbnet_infer(rgb, 1.0, 0.3, 960)
print(f"\n[DBNet prob stats] shape={prob.shape}  min={prob.min():.3f} max={prob.max():.3f} mean={prob.mean():.3f} p95={np.percentile(prob,95):.3f} p99={np.percentile(prob,99):.3f}")
# 取阈值 0.3 / 0.5 看联通域
for thr in [0.1, 0.3, 0.5]:
    bin255 = (prob*255 > thr*255).astype(np.uint8)*255
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(bin255, connectivity=8)
    sizes = sorted([int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)], reverse=True)[:5]
    print(f"  thr={thr:.1f}: comps={n-1}  top_sizes={sizes}")

# 7) DBNet 高分辨率推理 (mask_max_side=1600)
prob_hi, nw_hi, nh_hi, _, _, _ = _dbnet_infer(rgb, 1.0, 0.4, 1600)
print(f"\n[DBNet HI prob stats] shape={prob_hi.shape}  min={prob_hi.min():.3f} max={prob_hi.max():.3f} mean={prob_hi.mean():.3f} p95={np.percentile(prob_hi,95):.3f} p99={np.percentile(prob_hi,99):.3f}")
for thr in [0.3, 0.4, 0.5]:
    bin255 = (prob_hi*255 > thr*255).astype(np.uint8)*255
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(bin255, connectivity=8)
    sizes = sorted([int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)], reverse=True)[:5]
    print(f"  thr={thr:.1f}: comps={n-1}  top_sizes={sizes}")

# Save visualizations
out_dir = ROOT / "data" / "diag_root"
out_dir.mkdir(exist_ok=True)

# Orig (4x)
Image.fromarray(rgb).resize((W*4, H*4), Image.NEAREST).save(out_dir / "orig_4x.png")

# ML mask
Image.fromarray(mask_ml).resize((W*4, H*4), Image.NEAREST).save(out_dir / "mask_ml_4x.png")

# Current Otsu mask
Image.fromarray(mask_otsu).resize((W*4, H*4), Image.NEAREST).save(out_dir / "mask_otsu_4x.png")

# DBNet prob map upscaled
prob_full = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
Image.fromarray((prob_full*255).astype(np.uint8)).save(out_dir / "dbnet_prob.png")

prob_hi_full = cv2.resize(prob_hi, (W, H), interpolation=cv2.INTER_LINEAR)
Image.fromarray((prob_hi_full*255).astype(np.uint8)).save(out_dir / "dbnet_prob_hi.png")

# Draw boxes on orig
import PIL.ImageDraw as ImageDraw
img_box = Image.fromarray(rgb).convert("RGB").resize((W*4, H*4), Image.NEAREST)
dr = ImageDraw.Draw(img_box)
for b in boxes_dbnet:
    dr.rectangle([b["x0"]*4, b["y0"]*4, b["x1"]*4, b["y1"]*4], outline=(255,0,0), width=2)
img_box.save(out_dir / "dbnet_boxes_4x.png")

img_box2 = Image.fromarray(rgb).convert("RGB").resize((W*4, H*4), Image.NEAREST)
dr = ImageDraw.Draw(img_box2)
for b in boxes_classic:
    dr.rectangle([b["x0"]*4, b["y0"]*4, b["x1"]*4, b["y1"]*4], outline=(0,255,0), width=2)
img_box2.save(out_dir / "classic_boxes_4x.png")

print(f"\nsaved diag to {out_dir}")