"""填充边界小白块修复回归(1787980309628 = 668 纯色背景去发光字):
1. 填充区白块(>填充区中位+30)≈0(修复前 329px, 全贴蒙版边界);
2. 纹理图(武器石头)填充不变糊 —— std 对齐保留纹理对比度;
3. 基线七图不受影响。
"""
import sys
import numpy as np
import cv2
from PIL import Image

ROOT = "D:/Code/Project/Python/TextEraser"
sys.path.insert(0, ROOT)
from text_eraser.eraser import erase_text

fails = []

def run(path):
    rgb = np.array(Image.open(path).convert("RGB"))
    res, m, meta = erase_text(
        rgb, deglow_scheme="v2", glow_mode="auto", deglow_mask_soft=0.0,
        edge=1, q_off=55.0, max_area_ratio=0.4, max_box_ratio=0.4,
        deglow_strength=1.0, fill_white=True, fill_max_dist=12,
        deglow_zone_ratio=0.6, deglow_zone_expand=10, deglow_protect_px=1,
        return_mask=True, tint_fill=True, auto_edge=True)
    return res, m, meta

# 1) 668 白块
res, m, meta = run(f"{ROOT}/data/history/1787980309628/orig.bin")
rg = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
fill = m > 0
vals = rg[fill]
n_white = int((fill & (rg > float(np.median(vals)) + 30)).sum())
p99 = int(np.percentile(vals, 99))
print(f"668: 填充区白块={n_white} (修复前329, 应≤5)  填充区灰度 p99={p99} (修复前128, 应≤110)")
if n_white > 5 or p99 > 110:
    fails.append("668_white_blocks")

# 2) 武器: 纹理保持(填充区局部 std 不塌陷 —— 防「用模糊修白块」回归)
res2, m2, _ = run(f"{ROOT}/data/history/1787767429309/orig.bin")
fill2 = m2 > 0
g2 = cv2.cvtColor(res2, cv2.COLOR_RGB2GRAY).astype(np.float32)
# 填充区内 9x9 窗口 std 中位 —— 纹理对比度
loc_std = cv2.blur((g2 ** 2), (9, 9)) - cv2.blur(g2, (9, 9)) ** 2
std_fill = float(np.median(np.sqrt(np.clip(loc_std[fill2], 0, None))))
print(f"武器: 填充区局部纹理 std 中位={std_fill:.1f} (石头纹理应≥3)")
if std_fill < 3.0:
    fails.append("weapon_texture")

# 3) 基线七图
import subprocess
r = subprocess.run([sys.executable, f"{ROOT}/scripts/regress_hz4462.py"],
                   capture_output=True, text=True)
tail = r.stdout.strip().splitlines()[-8:]
print("\n".join(tail))
for line, want in [("178", "1273"), ("556", "5518"), ("635", "1325"), ("668", "10995")]:
    if not any(line in l and want in l for l in tail):
        fails.append(line)

print("\n==>", "全部通过" if not fails else f"失败: {fails}")
sys.exit(1 if fails else 0)
