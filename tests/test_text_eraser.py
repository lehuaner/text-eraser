"""text_eraser 包级冒烟测试（全部合成图，不依赖仓库内样图与历史数据）。"""
from __future__ import annotations

import numpy as np
import pytest


def _synthetic_rgb(w=320, h=200, seed=7) -> np.ndarray:
    """带纹理的中性渐变背景，用于填充/蒙版测试。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = (xx / w * 0.6 + yy / h * 0.4)
    base = np.stack([110 + 40 * t, 108 + 39 * t, 104 + 37 * t], axis=-1)
    noise = rng.normal(0, 3.5, (h, w, 1)).repeat(3, axis=2)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def test_version_and_exports():
    import text_eraser
    assert text_eraser.__version__
    for name in text_eraser.__all__:
        assert hasattr(text_eraser, name), name


def test_patch_fill_inpaints_hole():
    """纯色洞应被周围纹理背景填回，不残留黑洞/亮斑。"""
    rgb = _synthetic_rgb()
    hole = np.zeros(rgb.shape[:2], np.uint8)
    hole[80:120, 120:200] = 255          # 中央 40x80 洞, 填 0(黑)
    rgb_hole = rgb.copy()
    rgb_hole[hole > 0] = (0, 0, 0)

    from text_eraser import inpaint
    filled = inpaint(rgb_hole, hole, sample_mask=(255 - hole))

    center = filled[95:105, 150:170].astype(float)
    ring = rgb[95:105, 150:170].astype(float)   # 原图同位置 ≈ 真背景
    assert abs(center.mean() - ring.mean()) < 12.0


def test_detect_text_mask_classic_synthetic():
    """经典路径：合成图上大块非文字区域不应被当成文字。"""
    from text_eraser import detect_text_mask
    rgb = _synthetic_rgb()
    mask, _ = detect_text_mask(rgb, method="classic",
                               max_area_ratio=0.40, max_box_ratio=0.40)
    assert mask.shape == rgb.shape[:2]
    assert mask.dtype == np.uint8
    # 纯纹理背景无文字 → 蒙版应接近空
    assert mask.mean() < 1.0


def test_erase_text_end_to_end():
    """全流程冒烟：中央大字块被擦除且不残留白色（ml 路径，模型缺失则跳过）。"""
    pytest.importorskip("onnxruntime")
    from text_eraser import erase_text
    try:
        from text_eraser.ml_text_select import get_model_path
        get_model_path()
    except Exception as e:  # 离线/网络受限环境: 跳过而非失败
        pytest.skip(f"DBNet model unavailable: {e}")

    rgb = _synthetic_rgb(w=480, h=280)
    # 画一个粗白字块("口"形方框, 结构简单且 DBNet 稳定可检)
    rgb[100:180, 190:290] = (250, 250, 250)
    rgb[110:170, 200:280] = _synthetic_rgb(w=80, h=60, seed=9)[20]

    result, mask, meta = erase_text(rgb, return_mask=True)
    assert mask.any(), "文字未被检出"
    # 原白框区域不应残留大片纯白
    resid = (result[100:180, 190:290].min(axis=-1) > 235).mean()
    assert resid < 0.05, f"白框残留 {resid:.2%}"


def test_deglow_v2_large_zone_no_red_dominance():
    """v2 减绿在大发光区(zone≥80% H*W)不应留下 R>G 的「红偏」。

    回归：1788077005814(81x84, zone 86.7%)等小图在过冲 strength=1.15 下
    被压到 G<max(R,B) → R 主导 → 填充呈暗红。本测试构造 zone≈100% 的合成
    绿光图, 断言去发光后 R-G ≤ 1(中性感), 抓回退。
    """
    from text_eraser._shared_core import deglow_full_green_v2 as _sc_deglow_v2
    # 90x90: 整片绿光(R=35,G=120,B=30, R!=B 模拟真实背景微暖) + 中心白字。
    # strong_green 覆盖全图 → zone≈100% → 走不到「背景场重建」分支,
    # 这是过冲残留的精确放大版。
    H, W = 90, 90
    rgb = np.tile(np.array([35, 120, 30], np.uint8), (H, W, 1))
    # 中心 20x20 白字(触发 text_stroke 保护)
    rgb[35:55, 35:55] = (255, 255, 255)
    tmask = np.zeros((H, W), np.uint8)
    tmask[35:55, 35:55] = 255

    clean, _core, _zone = _sc_deglow_v2(
        rgb, tmask, strength=1.15, zone_ratio=0.6, zone_expand=10,
        protect_px=1, chroma_keep=1)

    # 绿光本体(非白字)的 R-G 必须中性和(允许 ±1 噪声)。
    body = np.ones((H, W), bool)
    body[35:55, 35:55] = False
    RmG = clean[body, 0].astype(np.int16) - clean[body, 1].astype(np.int16)
    assert float(RmG.mean()) <= 1.0, (
        f"大发光区出现 R 主导(红偏): mean R-G={RmG.mean():.2f} "
        f"(修复前 ~+14)")
    # 去色到中性灰: R-B 也必须中性和(否则残留黄绿)
    RmB = clean[body, 0].astype(np.int16) - clean[body, 2].astype(np.int16)
    assert float(RmB.mean()) <= 1.0, (
        f"大发光区残留黄绿(R-B 非中性): mean R-B={RmB.mean():.2f}")
    # 绿也应被去净: G 不应明显高于 max(R,B)
    R, G, B = clean[body, 0], clean[body, 1], clean[body, 2]
    excess = (G.astype(np.int16) - np.maximum(R, B).astype(np.int16))
    assert float(excess.mean()) <= 1.0, (
        f"绿未去净: mean(G-max(R,B))={excess.mean():.2f}")


# 注: 大发光区的「亮度匹配背景灰阶」与「纹理恢复(fill_lum_std>0)」是真实
# 1788077005814 等 3 张图(zone=86~99%)通过 B+detail 重建达成的, 由仓库内
# 14 图回归脚本(data/_diag_1788077005814/regress)验证: 3 张问题图 fill 亮度
# std 由 2.0 恢复到 ~15-25(与真背景 std 14-20 一致), 其余 11 张字节一致。
# 合成图无法可靠触发该路径(整图均匀绿光的合成图无真实非 zone 像素, 走
# 上面的回退逐像素去色, 亮度仍偏暗但中性, 由 test_deglow_v2_large_zone_no_red_dominance
# 守住 R-G/R-B 中性)。
