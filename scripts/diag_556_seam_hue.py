# -*- coding: utf-8 -*-
"""B场逐扇区色相 vs 真实背景; 浅色带进出 zone 对比; protect2 亮环统计。"""
import sys
import io
import contextlib
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

with contextlib.redirect_stdout(io.StringIO()):
    import diag_556_seam_band as _band
rgb, clean, cap = _band.rgb, _band.clean_real, _band.cap
H, W = _band.H, _band.W
zone, fb, prot = cap["zone"], cap["fb"], cap["protect2"]
B = cap["B"]  # shift 前的 B 场
a_map = cap.get("a_map", np.zeros((H, W), np.float32))

r = rgb[..., 0].astype(np.float32)
g = rgb[..., 1].astype(np.float32)
b = rgb[..., 2].astype(np.float32)
dout = cv2.distanceTransform((~zone).astype(np.uint8), cv2.DIST_L2, 5)
ys, xs = np.nonzero(fb)
ang = np.degrees(np.arctan2(ys - 123, xs - 184))
yy2, xx2 = np.nonzero((dout > 6) & (dout < 14))
ang2 = np.degrees(np.arctan2(yy2 - 123, xx2 - 184))
print("扇区        B场(shift前) R-G  G-B | 同向真实背景 R-G  G-B | hue差(R-G)")
for name, lo, hi in [("right", -30, 30), ("downright", 30, 75), ("down", 75, 105),
                     ("downleft", 105, 160), ("left", 160, 200),
                     ("upleft", -160, -110), ("up", -70, -20)]:
    m = (ang >= lo) & (ang < hi)
    if not m.any():
        continue
    br = (B[..., 0] - B[..., 1])[ys[m], xs[m]]
    gb = (B[..., 1] - B[..., 2])[ys[m], xs[m]]
    m2 = (ang2 >= lo) & (ang2 < hi)
    bgr = (r - g)[yy2[m2], xx2[m2]]
    bgb = (g - b)[yy2[m2], xx2[m2]]
    print(f"{name:>9s}   {np.median(br):+6.1f} {np.median(gb):+6.1f}   "
          f"|   {np.median(bgr):+6.1f} {np.median(bgb):+6.1f}   |   "
          f"{np.median(br) - np.median(bgr):+5.1f}")
print("(代码里 B 场还整体 G-4: shift 后各扇区 R-G 再 +4)")

strip = np.zeros((H, W), bool)
strip[:56, :] = True
sr = clean.astype(np.int16)
m_in = strip & zone & (a_map > 0.3)
m_out = strip & ~zone
if m_in.any() and m_out.any():
    print(f"\n浅色带(顶部米白横带) zone内(chroma-keep) {int(m_in.sum())}px  "
          f"结果RGB=({sr[...,0][m_in].mean():.0f},{sr[...,1][m_in].mean():.0f},{sr[...,2][m_in].mean():.0f})")
    print(f"                     zone外              {int(m_out.sum())}px  "
          f"结果RGB=({sr[...,0][m_out].mean():.0f},{sr[...,1][m_out].mean():.0f},{sr[...,2][m_out].mean():.0f})")
    print(f"                     原图 zone内        RGB="
          f"({r[m_in].mean():.0f},{g[m_in].mean():.0f},{b[m_in].mean():.0f})")

pm = prot & zone
lum_p = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY).astype(np.float32)
print(f"\nprotect2(只减绿区) 结果亮度: p50={np.median(lum_p[pm]):.0f} "
      f"p90={np.percentile(lum_p[pm], 90):.0f}")
lum_fb = lum_p[fb & (a_map <= 0.3)]
print(f"B场核心区 结果亮度: p50={np.median(lum_fb):.0f}")
lum_out = lum_p[dout > 14]
print(f"zone外背景 结果亮度: p50={np.median(lum_out):.0f}")
