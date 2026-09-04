"""Generate backend text-mask reference + original-image base64 for browser parity check.
Outputs into data/_glowcheck/ (temp, prefixed _cmp_):
  _cmp_backend_mask.png  : red glyphs on transparent (matches browser maskTransparent style)
  _cmp_orig_b64.txt      : base64 dataURL of the original (for the in-browser run)
"""
import base64
import sys
import numpy as np
import cv2

sys.path.insert(0, r"D:\Code\Project\Python\TextPatch")
from text_eraser.text_select import detect_text_mask

SRC = r"D:\Code\Project\Python\TextPatch\data\_glowcheck\556_orig.png"
OUT_DIR = r"D:\Code\Project\Python\TextPatch\data\_glowcheck"

rgb = cv2.cvtColor(cv2.imread(SRC), cv2.COLOR_BGR2RGB)
H, W = rgb.shape[:2]
print("image:", W, "x", H)

# Backend reference mask (the hybrid box+Otsu pipeline eraser.py actually uses)
mask, boxes = detect_text_mask(
    rgb, method="ml", strength=1.0,
    min_area=30, max_area_ratio=0.05, max_box_ratio=0.40, max_side=960,
    q_off=50.0, tint_fill=True, fill_white=True, fill_max_dist=12, upscale=True,
)
pix = int((mask > 0).sum())
print("backend mask pixels:", pix, "boxes:", len(boxes))

# Red-on-transparent PNG (mirrors browser maskTransparent: red 255,60,60 alpha 150)
rgba = np.zeros((H, W, 4), dtype=np.uint8)
rgba[..., 0] = 255
rgba[..., 1] = 60
rgba[..., 2] = 60
rgba[..., 3] = (mask > 0) * 150
cv2.imwrite(f"{OUT_DIR}/_cmp_backend_mask.png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

# Original as base64 dataURL for the browser run
with open(SRC, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("ascii")
with open(f"{OUT_DIR}/_cmp_orig_b64.txt", "w") as f:
    f.write("data:image/png;base64," + b64)
print("orig b64 len:", len(b64))
print("DONE")
