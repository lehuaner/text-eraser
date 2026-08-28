"""流程可视化：对每张发光样图，竖向展示算法每一步并标注步骤名。
步骤：①原图 → ②发光区检测 → ③Alpha分解去光晕 → ④文字笔画 → ⑤填充Mask → ⑥最终结果
顶部标注算法名；同时打印中心区域像素均值用于诊断纯黑。
"""
import sys, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = "D:/Code/Project/Python/TextPatch"
sys.path.insert(0, ROOT)
from core.text_select import detect_text_mask, _deglow_full_green_v2
from core.eraser import erase_text

try:
    FONT_T = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 19)
    FONT_S = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 13)
    FONT_TITLE = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 25)
except Exception:
    FONT_T = FONT_S = FONT_TITLE = ImageFont.load_default()

TH = 235        # 每步图高度
STEPBAR = 72    # 每步标题栏高度
BAR = 62        # 顶部算法标题高度
PAD = 6

TAGS = ["556", "668", "635", "178"]


def load_rgb(tag):
    return np.array(Image.open(f"{ROOT}/data/_glowcheck/{tag}.png").convert("RGB"))


def overlay_mask(rgb, mask, color=(255, 40, 40), alpha=0.55):
    vis = rgb.copy().astype(np.float32)
    m = mask > 0 if mask.dtype != np.uint8 else mask.astype(bool)
    for c, v in enumerate(color):
        vis[m, c] = vis[m, c] * (1 - alpha) + v * alpha
    return vis.clip(0, 255).astype(np.uint8)


def panel(im, target_h=TH):
    h, w = im.shape[:2]
    s = target_h / h
    return np.array(Image.fromarray(im).resize((int(round(w * s)), target_h), Image.LANCZOS))


def center_mean(rgb, r=42):
    H, W, _ = rgb.shape
    y0, y1 = max(0, H // 2 - r // 2), min(H, H // 2 + r // 2)
    x0, x1 = max(0, W // 2 - r // 2), min(W, W // 2 + r // 2)
    return tuple(int(round(v)) for v in rgb[y0:y1, x0:x1].mean(axis=(0, 1)))


def make_step(img, title, sub):
    img = panel(img)
    h, w, _ = img.shape
    canvas = np.full((h + STEPBAR, w, 3), 30, np.uint8)
    canvas[STEPBAR:STEPBAR + h] = img
    pil = Image.fromarray(canvas)
    d = ImageDraw.Draw(pil)
    d.text((6, 6), title, font=FONT_T, fill=(255, 225, 90))
    d.text((6, 34), sub, font=FONT_S, fill=(205, 215, 225))
    return np.array(pil)


def build(tag):
    rgb = load_rgb(tag)
    H, W, _ = rgb.shape
    tmask, _ = detect_text_mask(rgb, method="ml", tint_fill=False,
                                max_area_ratio=0.40, q_off=55)
    clean, core, dbg = _deglow_full_green_v2(
        rgb, tmask, strength=1.0, alpha_core=0.65, debug=True)
    res, m, meta = erase_text(
        rgb, deglow_scheme="v2", glow_mode="auto", deglow_mask_soft=0.0,
        edge=1, deglow_strength=1.0, fill_white=True, fill_max_dist=12,
        return_mask=True, tint_fill=True)

    green_ov = overlay_mask(rgb, dbg["strong_green"], color=(255, 40, 40))
    text_ov = overlay_mask(rgb, dbg["text_stroke"], color=(40, 170, 255))
    fill_ov = overlay_mask(rgb, core, color=(255, 120, 0))

    steps = [
        (rgb, "① 原图", "输入：白字 + 绿色光晕"),
        (green_ov, "② 发光区检测", "绿度 g-max(r,b)>15 且 g>100 标红"),
        (clean, "③ Alpha 分解去光晕", "T=I-α·Glow，外圈保留底层纹理"),
        (text_ov, "④ 文字笔画", "白字 min(RGB)>120 且非强绿 标蓝"),
        (fill_ov, "⑤ 填充 Mask", "白字(+高α核心)→patchmatch 标橙"),
        (res, "⑥ 最终结果", "去光晕 + 去文字"),
    ]
    panels = [make_step(im, t, s) for im, t, s in steps]
    cw = max(p.shape[1] for p in panels)
    rows = [np.array(Image.fromarray(p).resize((cw, p.shape[0]), Image.LANCZOS))
            for p in panels]
    stacked = np.vstack(rows)

    top = np.full((BAR, cw, 3), 16, np.uint8)
    pil = Image.fromarray(top)
    d = ImageDraw.Draw(pil)
    d.text((8, 8), f"算法 v2 · 发光区 Alpha 分解（保留纹理）— 样图 {tag} ({W}x{H})",
           font=FONT_TITLE, fill=(120, 220, 160))
    d.text((8, 40), "流程：原图 → 发光检测 → 分解去光晕 → 文字检测 → 填充 → 结果",
           font=FONT_S, fill=(205, 215, 225))
    out = np.vstack([np.array(pil), stacked])
    Image.fromarray(out).save(f"{ROOT}/data/_glowcheck/flow_{tag}.png")
    print(f"{tag}: 原图中心={center_mean(rgb)} 分解后中心={center_mean(clean)} "
          f"结果中心={center_mean(res)} fill={int((m > 0).sum())} "
          f"zone={int(dbg['zone'].sum())} text_stroke={int(dbg['text_stroke'].sum())}")


if __name__ == "__main__":
    for t in TAGS:
        build(t)
    print("done")
