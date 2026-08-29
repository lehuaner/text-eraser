"""生成发布用演示图与包内示例图（合成图，不含任何用户素材）。

输出:
  docs/assets/demo.png       README 演示图 (原图 | 擦除结果)
  textpatch/assets/example.png  包内示例图 (web 界面「示例图」按钮用)

用法: python scripts/make_release_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from textpatch import erase_text, to_rgb_uint8  # noqa: E402


def _find_font(size: int) -> ImageFont.FreeTypeFont:
    """找一个可用的中文字体；找不到就退回默认字体(仅英文)。"""
    candidates = [
        "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_input(w=520, h=300, glow: bool = False) -> Image.Image:
    """合成一张「游戏 UI 截图」风格的图：暖色渐变背景 + 噪声 + 白字(可选绿光晕)。"""
    rng = np.random.default_rng(20260830)

    # 暖色对角渐变 + 细噪声纹理
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = (xx / w * 0.6 + yy / h * 0.4)
    base = np.stack([
        96 + 52 * t,          # R
        84 + 40 * t,          # G
        70 + 30 * t,          # B
    ], axis=-1)
    noise = rng.normal(0, 4.5, (h, w, 1)).repeat(3, axis=2)
    # 低频斑块让背景不那么"平"
    patchy = cv2_gauss(noise)
    bg = np.clip(base + patchy, 0, 255).astype(np.uint8)

    img = Image.fromarray(bg, "RGB")
    draw = ImageDraw.Draw(img)
    font_big = _find_font(56)
    font_small = _find_font(28)

    # 主标题 + 属性行 (白字, 可选绿光晕)
    title = "强化成功"
    sub = "攻击力 +125"
    tw = draw.textlength(title, font=font_big)
    sw = draw.textlength(sub, font=font_small)
    tpos = ((w - tw) / 2, 78)
    spos = ((w - sw) / 2, 178)

    if glow:
        glow_img = Image.new("RGB", img.size, (0, 0, 0))
        gdraw = ImageDraw.Draw(glow_img)
        gdraw.text(tpos, title, font=font_big, fill=(0, 255, 60))
        gdraw.text(spos, sub, font=font_small, fill=(0, 255, 60))
        glow_img = glow_img.filter(ImageFilter.GaussianBlur(6))
        garr = np.asarray(glow_img, np.float32)
        strength = (garr.max(axis=-1, keepdims=True) / 255.0)
        out = np.clip(np.asarray(img, np.float32) + garr * 0.85 * strength, 0, 255)
        img = Image.fromarray(out.astype(np.uint8), "RGB")
        draw = ImageDraw.Draw(img)

    # 白字芯叠在光晕上
    draw.text((tpos[0] - 2, tpos[1] - 2), title, font=font_big, fill=(255, 255, 255))
    draw.text((spos[0] - 1, spos[1] - 1), sub, font=font_small, fill=(252, 252, 252))
    return img


def cv2_gauss(noise: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.GaussianBlur(noise, (0, 0), 9)


def main() -> None:
    img = make_input()
    rgb = to_rgb_uint8(img)
    result, _mask, meta = erase_text(rgb, return_mask=True)
    print("mask_px:", meta.get("mask_pix"), " sec:", round(meta.get("inpaint_seconds", 0), 2))

    # 拼接: 原图 | 结果, 4px 白色分隔
    sep = np.full((rgb.shape[0], 4, 3), 255, np.uint8)
    combo = np.concatenate([rgb, sep, result], axis=1)
    out_dir = ROOT / "docs" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(combo).save(out_dir / "demo.png")
    print("saved:", out_dir / "demo.png")

    # 包内示例图 (web「示例图」按钮)
    asset_dir = ROOT / "textpatch" / "assets"
    asset_dir.mkdir(exist_ok=True)
    img.resize((int(img.width * 0.6), int(img.height * 0.6))).save(asset_dir / "example.png")
    print("saved:", asset_dir / "example.png")


if __name__ == "__main__":
    main()
