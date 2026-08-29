"""textpatch 包级冒烟测试（全部合成图，不依赖仓库内样图与历史数据）。"""
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
    import textpatch
    assert textpatch.__version__
    for name in textpatch.__all__:
        assert hasattr(textpatch, name), name


def test_patch_fill_inpaints_hole():
    """纯色洞应被周围纹理背景填回，不残留黑洞/亮斑。"""
    rgb = _synthetic_rgb()
    hole = np.zeros(rgb.shape[:2], np.uint8)
    hole[80:120, 120:200] = 255          # 中央 40x80 洞, 填 0(黑)
    rgb_hole = rgb.copy()
    rgb_hole[hole > 0] = (0, 0, 0)

    from textpatch import inpaint
    filled = inpaint(rgb_hole, hole, sample_mask=(255 - hole))

    center = filled[95:105, 150:170].astype(float)
    ring = rgb[95:105, 150:170].astype(float)   # 原图同位置 ≈ 真背景
    assert abs(center.mean() - ring.mean()) < 12.0


def test_detect_text_mask_classic_synthetic():
    """经典路径：合成图上大块非文字区域不应被当成文字。"""
    from textpatch import detect_text_mask
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
    from textpatch import erase_text
    try:
        from textpatch.ml_text_select import get_model_path
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
