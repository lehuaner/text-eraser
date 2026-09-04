"""``text_eraser.core`` — 共享算法核的公开门面（自定义管线入口）。

0.3.0 起算法核心只有一份实现：textcore.wasm（后端经 wasmtime、浏览器经
WebAssembly 调同一份字节码，逐字节一致）。这个模块把常用算子直接暴露给
调用者，便于在 demo 之外**自由编排自己的管线** —— web 界面只是参考实现。

引擎选择（算法在哪里跑）::

    # 后端引擎（本包）
    from text_eraser import erase_text
    result, mask, meta = erase_text(rgb, return_mask=True)

    # 浏览器引擎（ESM 包 text-eraser-browser，见 browser/ 目录）
    import { erase, eraseTextGlyphs, inpaint } from 'text-eraser-browser';

    # 深度自定义（后端）：直接取共享核算子，自己编排
    from text_eraser import core
    tmask, boxes = core.detect_text_mask(rgb)          # DBNet 检测（可选）
    clean, core_mask, zone = core.deglow_full_green_v2(rgb, tmask, strength=1.0)
    result, fill, clean, zone = core.erase_text_glyphs(rgb, tmask, tmask2)
    filled = core.patchmatch_inpaint_fill(roi, roi_mask)   # 纯填充

管线参考（erase_text 的编排顺序，可直接增删步骤）:
    detect → deglow(v2) → re-detect → mask 修复 → patchmatch 填充
"""
from text_eraser._shared_core import (
    using_shared_core,
    # 底层原语（与浏览器 cv-bridge 一致）
    rgb2gray, threshold_otsu, connected_components,
    connected_components_with_stats, edt_to_nearest_zero,
    dilate, erode, morphology_ex,
    resize_gray_cubic, resize_float_linear,
    # 高层算子（整条共享管线的构建块）
    patchmatch_inpaint_fill,
    smooth_telea_full,
    grow_color_tint,
    deglow_full_green_v2,
    erase_text_glyphs,
)
from text_eraser._textcore import CoreLoadError, get_core, reset_core

__all__ = [
    "using_shared_core", "get_core", "reset_core", "CoreLoadError",
    # 原语
    "rgb2gray", "threshold_otsu", "connected_components",
    "connected_components_with_stats", "edt_to_nearest_zero",
    "dilate", "erode", "morphology_ex",
    "resize_gray_cubic", "resize_float_linear",
    # 高层
    "patchmatch_inpaint_fill", "smooth_telea_full", "grow_color_tint",
    "deglow_full_green_v2", "erase_text_glyphs",
]
