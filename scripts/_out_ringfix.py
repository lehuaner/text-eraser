"""临时: 668 去发光「旧(纯减绿) vs 新(含环带中和)」放大对比 + 默认定稿。"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.text_select import detect_text_mask, _deglow_full_green_v2

rgb = np.array(Image.open("data/_glowcheck/668.png").convert("RGB"))
H, W, _ = rgb.shape
tmask, _ = detect_text_mask(rgb, method="ml", tint_fill=False,
                            max_area_ratio=0.40, q_off=55,
                            fill_white=True, fill_max_dist=12)
clean, core, dbg = _deglow_full_green_v2(
    rgb, tmask, strength=1.15, alpha_core=0.65,
    zone_ratio=0.6, zone_expand=24, debug=True)

# 旧版: 纯减绿(仅 G 减绿度)
zone = dbg["zone"]
r0 = rgb[..., 0].astype(np.int16); g0 = rgb[..., 1].astype(np.int16)
b0 = rgb[..., 2].astype(np.int16)
old = rgb.copy().astype(np.int16)
grn = np.maximum(g0 - np.maximum(r0, b0), 0)
old[zone, 1] = np.clip(g0[zone].astype(np.float32) - grn[zone] * 1.15,
                        0, 255).astype(np.int16)
old = old.clip(0, 255).astype(np.uint8)

FONT = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
UP = 2


def panel(im, lab):
    im = cv2.resize(im, (W * UP, H * UP), interpolation=cv2.INTER_NEAREST)
    bar = np.full((34, W * UP, 3), 16, np.uint8)
    p = Image.fromarray(cv2.cvtColor(np.vstack([bar, im]), cv2.COLOR_RGB2BGR))
    ImageDraw.Draw(p).text((8, 6), lab, font=FONT, fill=(255, 225, 90))
    return p


panels = [panel(rgb, "① 原图 (668, 黄绿光晕)"),
          panel(old, "② 旧: 纯减绿 (黑/红残带)"),
          panel(clean, "③ 新: 减绿 + 环带中和 (向背景靠谱)")]
row = Image.new("RGB", (sum(p.width for p in panels), panels[0].height), 12)
x = 0
for p in panels:
    row.paste(p, (x, 0))
    x += p.width
out = "data/_glowcheck/cmp_ringfix_668.png"
row.save(out)
print("saved", out, row.size)