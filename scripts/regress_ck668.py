"""色度结构保留去绿回归(1787980309628 = 668 开 deglow_chroma_keep):
1. 保留区不残留绿光(修复前 891px 绿度 +9~14, 与同行 +1 可见色差);
2. 黄条结构仍保留(|R−B| 接近原图, 远高于关开关的重建结果);
3. 关开关路径与基线七图不受影响。
"""
import sys
import numpy as np
import cv2
from PIL import Image

ROOT = "D:/Code/Project/Python/TextPatch"
sys.path.insert(0, ROOT)
from core.eraser import erase_text

def run(keep):
    rgb = np.array(Image.open(f"{ROOT}/data/history/1787980309628/orig.bin").convert("RGB"))
    res, m, meta = erase_text(
        rgb, deglow_scheme="v2", glow_mode="auto", deglow_mask_soft=0.0,
        edge=1, q_off=55.0, max_area_ratio=0.4, max_box_ratio=0.4,
        deglow_strength=1.0, fill_white=True, fill_max_dist=12,
        deglow_zone_ratio=0.6, deglow_zone_expand=10, deglow_protect_px=1,
        deglow_chroma_keep=keep, return_mask=True, tint_fill=True, auto_edge=True)
    return rgb, res, meta

fails = []

# 1) 开保留: 全图绿像素=0, 黄条带结构保留
rgb, res, meta = run(True)
clean = meta["deglow_img"]
r = clean[..., 0].astype(int); g = clean[..., 1].astype(int); b = clean[..., 2].astype(int)
n_green = int(((g - np.maximum(r, b)) > 8).sum())
band = (slice(86, 104), slice(150, 290))
rb90 = float(np.percentile(np.abs(r - b)[band], 90))
print(f"开保留: 绿像素={n_green} (修复前891, 应0)  黄条带|R-B| p90={rb90:.0f} (应≈16, 关开关仅5)")
if n_green != 0 or rb90 < 10:
    fails.append("chroma_keep_on")

# 2) 关保留: 行为不变(绿像素0, 结构被重建 |R-B| 低)
_, _, meta0 = run(False)
clean0 = meta0["deglow_img"]
r0 = clean0[..., 0].astype(int); g0 = clean0[..., 1].astype(int); b0 = clean0[..., 2].astype(int)
n0 = int(((g0 - np.maximum(r, b)) > 8).sum())
rb0 = float(np.percentile(np.abs(r0 - b0)[band], 90))
print(f"关保留: 绿像素={n0}  黄条带|R-B| p90={rb0:.0f} (重建抹平, 应<10)")
if n0 != 0 or rb0 > 10:
    fails.append("chroma_keep_off")

# 3) 基线七图(全走关保留路径, 逐像素不变)
import subprocess
r_ = subprocess.run([sys.executable, f"{ROOT}/scripts/regress_hz4462.py"],
                    capture_output=True, text=True)
tail = r_.stdout.strip().splitlines()[-8:]
print("\n".join(tail))
for line, want in [("178", "1273"), ("556", "5518"), ("635", "1325"), ("668", "10995")]:
    if not any(line in l and want in l for l in tail):
        fails.append(line)

print("\n==>", "全部通过" if not fails else f"失败: {fails}")
sys.exit(1 if fails else 0)
