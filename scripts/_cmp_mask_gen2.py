"""Generate backend reference text masks + DBNet prob maps for cross-validating the
browser JS port of detect_text_mask(method='ml').

For each image we dump:
  rgb.raw       float32 H*W*3   (original-resolution RGB, what the Otsu/fill/tint steps use)
  prob.raw      float32 nw*nh   (DBNet prob map at max_side=960, what box detection uses)
  refmask.raw   uint8   H*W     (255=text) backend reference mask
  dims.txt      "H W"
  meta.txt      "nw nh thr"     (prob dims + box threshold used by backend)
  boxes.json    text boxes (for sanity)
  refmask.png   visual

The JS port consumes prob.raw + rgb.raw + meta to reproduce the mask and we compare via IoU.
"""
import os, json
import numpy as np
from PIL import Image
import cv2
from text_eraser.text_select import detect_text_mask
from text_eraser.ml_text_select import _dbnet_infer

IMAGES = [
    r"D:/Code/Project/Python/TextPatch/data/_glowcheck/556_orig.png",
    r"D:/Code/Project/Python/TextPatch/data/_glowcheck/668_orig.png",
    r"D:/Code/Project/Python/TextPatch/data/_glowcheck/178_orig.png",
    r"D:/Code/Project/Python/TextPatch/data/_glowcheck/635_orig.png",
    r"D:/Code/Project/Python/TextPatch/data/_d5814_orig.png",
    r"D:/Code/Project/Python/TextPatch/data/_I_fix.png",
    r"D:/Code/Project/Python/TextPatch/data/_ab_synth.png",
]
OUT = r"D:/Code/Project/Python/TextPatch/scripts/_cmp_work"

# Exact backend frontend defaults (see meta.json / _webapp_impl.py / eraser.py)
KW = dict(method="ml", q_off=55.0, max_area_ratio=0.40, max_box_ratio=0.40,
          max_side=960, tint_fill=True, fill_white=True, fill_max_dist=12,
          strength=1.0, min_area=30, upscale=True)

os.makedirs(OUT, exist_ok=True)
for p in IMAGES:
    if not os.path.isfile(p):
        print("MISSING", p); continue
    name = os.path.splitext(os.path.basename(p))[0]
    rgb = np.asarray(Image.open(p).convert("RGB"))
    H, W = rgb.shape[:2]
    prob, nw, nh, _, _, thr = _dbnet_infer(rgb, 1.0, 0.3, 960)
    mask, boxes = detect_text_mask(rgb, **KW)
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    rgb.astype(np.float32).tofile(os.path.join(d, "rgb.raw"))
    prob.astype(np.float32).tofile(os.path.join(d, "prob.raw"))
    mask.astype(np.uint8).tofile(os.path.join(d, "refmask.raw"))
    with open(os.path.join(d, "dims.txt"), "w") as f:
        f.write(f"{H} {W}")
    with open(os.path.join(d, "meta.txt"), "w") as f:
        f.write(f"{nw} {nh} {thr:.6f}")
    with open(os.path.join(d, "boxes.json"), "w") as f:
        json.dump(boxes, f)
    cv2.imwrite(os.path.join(d, "refmask.png"), mask)
    print(f"{name}: {H}x{W} prob={nw}x{nh} thr={thr:.4f} "
          f"mask_pix={int(mask.sum()//255)} boxes={len(boxes)}")
