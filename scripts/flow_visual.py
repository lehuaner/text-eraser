"""流程可视化：对每张发光样图，横向展示算法每一步并标注步骤名与说明。

步骤（文案按需求）：
 ① 原图
 ② 发光区检测：将检测到的发光区域用红色高亮标出。
 ③ 去发光结果：单独展示去除了绿色光晕后的文字图像。
 ④ 文字蒙版(去发光图)：在去除了发光效果的图像上检测文字，并将文字区域标记为红色。
 ⑤ 填充 Mask：将蓝色文字区域进行膨胀并填充，标记为橙色。
 ⑥ 最终结果：展示了算法最终的输出。

布局：黑色背景，6 步横向一行；每步顶部为标签区（编号标题 + 自动换行的说明），
下方图片为**原始分辨率，不缩放**。输出 data/_glowcheck/flow_{tag}.png 四张。
"""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = "D:/Code/Project/Python/TextPatch"
sys.path.insert(0, ROOT)
from core.text_select import detect_text_mask, _deglow_full_green_v2
from core.eraser import erase_text

try:
    FONT_T = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 17)
    FONT_S = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 12)
    FONT_TITLE = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 24)
except Exception:
    FONT_T = FONT_S = FONT_TITLE = ImageFont.load_default()

PAD = 16          # 格间黑边
GAP = 8           # 标签区内边距
BAR = 46          # 顶部算法标题区高度
LABEL_H_MIN = 56  # 每步标签区最小高度

TAGS = ["178", "556", "635", "668"]

STEPS = [
    ("① 原图", "输入：白字 + 绿色光晕"),
    ("② 发光区检测", "将检测到的发光区域用红色高亮标出。"),
    ("③ 去发光结果", "单独展示去除了绿色光晕后的文字图像。"),
    ("④ 文字蒙版(去发光图)", "在去除了发光效果的图像上检测文字，并将文字区域标记为红色。"),
    ("⑤ 填充 Mask", "将蓝色文字区域进行膨胀并填充，标记为橙色。"),
    ("⑥ 最终结果", "展示了算法最终的输出。"),
]


def load_rgb(tag):
    return np.array(Image.open(f"{ROOT}/data/_glowcheck/{tag}.png").convert("RGB"))


def overlay_mask(rgb, mask, color=(255, 40, 40), alpha=0.55):
    vis = rgb.copy().astype(np.float32)
    m = mask > 0 if mask.dtype != np.uint8 else mask.astype(bool)
    for c, v in enumerate(color):
        vis[m, c] = vis[m, c] * (1 - alpha) + v * alpha
    return vis.clip(0, 255).astype(np.uint8)


def wrap_lines(draw, text, font, max_w):
    """按像素宽度自动换行。"""
    lines, cur = [], ""
    for ch in text:
        if draw.textlength(cur + ch, font=font) <= max_w:
            cur += ch
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


def make_cell(img_nd, title, desc):
    """单步格子：黑底标签区(标题+说明, 自动换行) + 原分辨率图片。返回 PIL 图。"""
    img = Image.fromarray(img_nd)
    ih, iw = img.size[1], img.size[0]

    # 1) 先按宽松宽度算说明行数 → 确定格宽(标签区至少容纳说明一行)
    tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    probe = tmp.textlength(desc, font=FONT_S)
    cell_w = max(iw, int(probe) + 24, 150)

    # 2) 说明行数
    d = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines = wrap_lines(d, desc, FONT_S, cell_w - 2 * GAP)
    label_h = max(LABEL_H_MIN, 8 + 24 + len(lines) * 17 + 6)

    canvas = Image.new("RGB", (cell_w, label_h + ih), (18, 18, 18))
    dr = ImageDraw.Draw(canvas)
    dr.text((GAP, 6), title, font=FONT_T, fill=(255, 225, 90))
    yy = 6 + 24
    for ln in lines:
        dr.text((GAP, yy), ln, font=FONT_S, fill=(200, 210, 220))
        yy += 17
    # 图片左对齐放置(留出黑边)
    canvas.paste(img, (GAP, label_h))
    return canvas


def build(tag):
    rgb = load_rgb(tag)
    H, W, _ = rgb.shape
    # 1) 文字检测(保护 + 生长种子；不做色偏生长)
    tmask, _ = detect_text_mask(rgb, method="ml", tint_fill=False,
                                max_area_ratio=0.40, q_off=55,
                                fill_white=True, fill_max_dist=12)
    # 2) 先去发光(减绿度)：绿晕→中性灰（单独一步展示）
    clean, core, dbg = _deglow_full_green_v2(
        rgb, tmask, strength=1.15, alpha_core=0.65,
        zone_ratio=0.6, zone_expand=24, debug=True)
    # 3) 在「去发光图」上检测文字(非高亮算法：tint_fill=True 自动并入残余绿)
    mask, boxes = detect_text_mask(clean, method="ml", tint_fill=True,
                                   max_area_ratio=0.40, q_off=55,
                                   fill_white=True, fill_max_dist=12)
    # 4) 完整擦除(先去发光 → 再去字)取最终结果 + 中间 deglow
    res, m, meta = erase_text(
        rgb, deglow_scheme="v2", glow_mode="auto", deglow_mask_soft=0.0,
        edge=1, deglow_strength=1.15, fill_white=True, fill_max_dist=12,
        deglow_zone_ratio=0.6, deglow_zone_expand=24,
        return_mask=True, tint_fill=True)

    # 步骤图：②发光区(红：强绿种子检测区, 收敛不泛红) / ④文字蒙版(红) / ⑤填充区(橙)
    glow_ov = overlay_mask(rgb, dbg["strong_green"], color=(255, 30, 30),
                           alpha=0.55)
    text_ov = overlay_mask(clean, mask, color=(255, 30, 30), alpha=0.55)
    fill_ov = overlay_mask(clean, m, color=(255, 120, 0), alpha=0.55)

    imgs = [rgb, glow_ov, clean, text_ov, fill_ov, res]
    cells = [make_cell(np.array(i), t, s)
             for i, (t, s) in zip(imgs, STEPS)]

    # 横向拼一行, 格间黑边, 整体黑底
    gap = Image.new("RGB", (PAD, 1), (12, 12, 12))
    row = Image.new("RGB", (sum(c.width for c in cells) + PAD * (len(cells) - 1),
                            max(c.height for c in cells)), (12, 12, 12))
    xx = 0
    for c in cells:
        row.paste(gap, (xx, 0))
        xx += PAD
        row.paste(c, (xx, 0))
        xx += c.width

    top = Image.new("RGB", (row.width, BAR), (14, 14, 14))
    dr = ImageDraw.Draw(top)
    dr.text((10, 9),
            f"算法 v2 · 减绿度去发光 → 非高亮算法去字 — 样图 {tag} ({W}x{H} 原分辨率)",
            font=FONT_TITLE, fill=(120, 220, 160))
    out = Image.new("RGB", (row.width, BAR + row.height), (12, 12, 12))
    out.paste(top, (0, 0))
    out.paste(row, (0, BAR))
    out.save(f"{ROOT}/data/_glowcheck/flow_{tag}.png")
    print(f"saved flow_{tag}.png  size={out.size}  "
          f"zone={int(dbg['zone'].sum())} mask={int((mask > 0).sum())} "
          f"fill={int((m > 0).sum())}")


if __name__ == "__main__":
    for t in TAGS:
        build(t)
    print("done")