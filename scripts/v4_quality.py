"""v4 结果质量验证：光晕区去除量 / 文字区保护 / 残留绿。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PIL import Image
import io

ROOT = Path(__file__).resolve().parent.parent

from deglow import pipeline

tp = ROOT / "data" / "history" / "1787768178725" / "orig.bin"
raw = tp.read_bytes()
rgb = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), np.uint8)
P = rgb.astype(np.float32)
res = pipeline.run(rgb, deglow_strength=1.0)
out = res.image.astype(np.float32)
print("has_glow:", res.report["has_glow"], "glow_pix:", res.report["glow_pix"])

r_, g_, b_ = (rgb[..., 0].astype(np.int16), rgb[..., 1].astype(np.int16),
              rgb[..., 2].astype(np.int16))
for dom in res.domains[:3]:
    m = dom.mask
    if not m.any():
        continue
    # 环带背景均值（近似真实背景）
    w = max(int(np.ceil(1.5 * dom.sigma_g)) + 2, 3)
    band = cv2.dilate(m.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * w + 1, 2 * w + 1))) > 0
    band &= ~m
    bg = rgb[band].mean(0).astype(np.float32) if band.any() else None
    print("=" * 60)
    print("dom", dom.id, "mode", dom.mode, "pix", int(m.sum()))
    if bg is not None:
        d_orig = np.abs(P[m] - bg).mean()
        d_out = np.abs(out[m] - bg).mean()
        print(f"  |orig−bg|={d_orig:.1f}  |out−bg|={d_out:.1f}  (去除量={d_orig - d_out:.1f})")
    resid_g = float(np.mean((out[..., 1] - np.maximum(out[..., 0], out[..., 2]))[m] > 6))
    print(f"  残留绿(out g−max>6): {resid_g:.3%}")
    bright = float(np.mean((out[m].max(-1) - P[m].max(-1)) > 2))
    print(f"  域内变亮(out>orig 亮度+2): {bright:.3%}")
# 文字区保护：carrier 像素原图=结果？
save = cv2.cvtColor(out, cv2.COLOR_RGB2BGR) if out.ndim == 3 else out
cv2.imwrite(str(ROOT / "data" / "_v4_verify_orig.png"), cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))
cv2.imwrite(str(ROOT / "data" / "_v4_verify_out.png"), cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2BGR))
print("saved _v4_verify_orig/out.png")