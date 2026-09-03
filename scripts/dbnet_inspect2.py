"""对单图跑 DBNet，直接导出概率图，看为啥 boxes=[]."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from text_eraser.text_select import to_rgb_uint8

img_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "needExtractAndPatch2.png")
rgb = to_rgb_uint8(Image.open(img_path).convert("RGB"))
H, W = rgb.shape[:2]
print(f"image = {W}x{H}")

from text_eraser.ml_text_select import _dbnet_infer, detect_text_ml, _get_session
sess = _get_session()
print(f"providers = {sess.get_providers()}")
prob, nw, nh, H, W, thr = _dbnet_infer(rgb, 1.0, 0.3, 960)
print(f"prob shape={prob.shape}  min={prob.min():.3f}  max={prob.max():.3f}  mean={prob.mean():.3f}  p99={np.percentile(prob,99):.3f}  p95={np.percentile(prob,95):.3f}  default_thr={thr}")
# thresh variants
for thr in [0.10, 0.20, 0.30, 0.50]:
    bin255 = (prob * 255 > thr * 255).astype(np.uint8) * 255
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(bin255, connectivity=8)
    sizes = sorted([int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)], reverse=True)[:5]
    print(f"  thr={thr:.2f}: components={n-1} top_sizes={sizes}")

# save full-resolution prob map (upscaled to img size if needed)
prob_full = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
out_dir = ROOT / "data" / "dryrun_out"
out_dir.mkdir(exist_ok=True)
Image.fromarray((prob_full * 255).astype(np.uint8)).save(out_dir / "_img2_dbnet_prob.png")
# Also save thresholded at 0.3
bin255 = (prob_full * 255 > 76).astype(np.uint8) * 255
Image.fromarray(bin255).save(out_dir / "_img2_dbnet_thr076.png")
print(f"saved {out_dir}/_img2_dbnet_prob.png and _img2_dbnet_thr076.png")