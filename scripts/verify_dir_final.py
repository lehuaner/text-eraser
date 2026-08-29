"""验证 direction 修复：多角度不崩 + 无方向/方向 效果对比图。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from textpatch.eraser import erase_text
import time

IMGS = {
    "武器": Path("D:/Code/Project/Python/ExtractRole/data/needExtractAndPatch.png"),
    "座驾": ROOT / "data" / "diag_root" / "needExtractAndPatch2_orig.png",
}

OUT = ROOT / "data" / "verify_dir"
OUT.mkdir(parents=True, exist_ok=True)

def blur_ratio(result, mask):
    """填充区 vs 周边原纹理 的拉普拉斯方差比(>1 锐利)。"""
    g = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    inside = mask > 0
    if not inside.any():
        return float('nan')
    # 周边环带 = 距离填充区 <= 12px 的非填充像素
    dist = cv2.distanceTransform((~inside).astype(np.uint8), cv2.DIST_L2, 5)
    ring = (~inside) & (dist <= 12)
    if not ring.any():
        return float('nan')
    return float(lap[inside].var() / max(lap[ring].var(), 1e-6))

for name, p in IMGS.items():
    rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
    print(f"\n===== {name} =====")
    # 多角度稳健性
    for ang in (0, 30, 60, 90, 135):
        try:
            t = time.time()
            erase_text(rgb, direction=float(ang), glow_mode="off")
            print(f"  [OK] {ang:3d}°  用时 {time.time()-t:.2f}s")
        except Exception as e:
            print(f"  [FAIL] {ang:3d}°  {type(e).__name__}: {e}")
    # 对比图：无方向 vs 方向60
    r0, m0, _ = erase_text(rgb, direction=None, glow_mode="off", return_mask=True)
    r60, m60, _ = erase_text(rgb, direction=60.0, glow_mode="off", return_mask=True)
    print(f"  blur_ratio: 无方向={blur_ratio(r0,m0):.2f}  方向60={blur_ratio(r60,m60):.2f}")
    # 拼图: 左=无方向结果, 右=方向60结果, 中=红蒙版占位
    pad = np.full((rgb.shape[0], 6, 3), 255, np.uint8)
    comp = np.hstack([r0, pad, r60])
    Image.fromarray(comp).save(OUT / f"{name}_nodir_vs_dir60.png")
    Image.fromarray(r60).save(OUT / f"{name}_dir60.png")
print(f"\n对比图已存: {OUT}")
