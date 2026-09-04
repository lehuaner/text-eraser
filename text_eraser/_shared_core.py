"""textcore.wasm 共享算法核的后端集成点（0.3.0 起 wasm 是唯一算法核心）。

这里是 Python 后端调用共享算子的**唯一**入口。加载与浏览器 Worker 完全相同的
wasm（打包在 ``text_eraser/assets/textcore.wasm``），保证前后端逐字节一致。

0.3.0 破坏性变更：
- Python/cv2 回退实现已全部删除 —— 核加载失败直接抛 ``CoreLoadError``，
  不再静默降级（静默降级会重新引入前后端分歧）。
- ``TEXTCORE_BACKEND=0`` 调试开关已移除（无 Python 核心可切）。

保留的 cv2 使用面（非核心算法，检测链必需）:
- DBNet 检测前/后处理（resize INTER_AREA/GaussianBlur 等 float 算子，位级
  复刻不可达，见 docs/HANDOFF-wasm-python-fill-unify.md §13/§14）。
"""
from __future__ import annotations

import numpy as np

from text_eraser._textcore import CoreLoadError, get_core

__all__ = [
    "CoreLoadError", "using_shared_core", "get_core",
    "rgb2gray", "threshold_otsu", "connected_components",
    "connected_components_with_stats", "edt_to_nearest_zero",
    "dilate", "erode", "morphology_ex",
    "resize_gray_cubic", "resize_float_linear",
    "patchmatch_inpaint_fill", "smooth_telea_full", "grow_color_tint",
    "deglow_full_green_v2", "erase_text_glyphs",
]


def _get_core():
    """返回当前线程的 wasm 核实例；加载失败抛 CoreLoadError（快速、明确）。"""
    return get_core()


def using_shared_core() -> bool:
    """True 表示后端正经共享核分发（wasm 就绪）。失败返回 False 而不抛。"""
    try:
        return get_core() is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 算子（drop-in cv2 兼容；全部 wasm 实现，无 Python 回退）
# ---------------------------------------------------------------------------
def rgb2gray(rgb):
    """RGB HxWx3 (uint8/float) -> uint8 HxW，与 cv2.cvtColor(COLOR_RGB2GRAY) 逐字节一致。"""
    arr = np.asarray(rgb)
    if arr.ndim == 3:
        arr = arr[..., :3]
    H, W = arr.shape[:2]
    return _get_core().rgb_to_gray(arr, H, W)


def threshold_otsu(gray, *_, **__):
    """单通道 Otsu 阈值。返回 (thr: float, bin: uint8 0/255)。"""
    g = np.asarray(gray)
    H, W = g.shape[:2]
    thr, bin = _get_core().threshold_otsu(g.astype(np.uint8), H, W)
    return float(thr), bin


def connected_components(mask, connectivity=8):
    """与 cv2.connectedComponents(mask, connectivity) 匹配。返回 (n, labels HxW int32)。"""
    m = np.asarray(mask)
    H, W = m.shape[:2]
    n, labels, _stats = _get_core().connected_components(m.astype(np.uint8), H, W)
    return n, labels


def connected_components_with_stats(mask, connectivity=8):
    """与 cv2.connectedComponentsWithStats 匹配。返回 (n, labels, stats(n,5), centroids(n,2))。

    stats 列序与 cv2 一致: [CC_STAT_LEFT, CC_STAT_TOP, CC_STAT_WIDTH,
    CC_STAT_HEIGHT, CC_STAT_AREA] = [0,1,2,3,4]。
    """
    m = np.asarray(mask)
    H, W = m.shape[:2]
    n, labels, stats = _get_core().connected_components(m.astype(np.uint8), H, W)
    stat_arr = np.zeros((n, 5), np.int32)
    for i, s in enumerate(stats):
        stat_arr[i, 0] = s["left"]
        stat_arr[i, 1] = s["top"]
        stat_arr[i, 2] = s["width"]
        stat_arr[i, 3] = s["height"]
        stat_arr[i, 4] = s["area"]
    centroids = np.zeros((n, 2), np.float64)
    if n > 1:
        flat = labels.ravel()
        cnt = np.bincount(flat, minlength=n)
        sx = np.bincount(flat, weights=np.broadcast_to(np.arange(W, dtype=np.float64), (H, W)).ravel(), minlength=n)
        sy = np.bincount(flat, weights=np.broadcast_to(np.arange(H, dtype=np.float64)[:, None], (H, W)).ravel(), minlength=n)
        nz = cnt[1:] > 0
        centroids[1:, 0] = np.where(nz, sx[1:] / cnt[1:], 0.0)
        centroids[1:, 1] = np.where(nz, sy[1:] / cnt[1:], 0.0)
    return n, labels, stat_arr, centroids


def edt_to_nearest_zero(src):
    """到最近 ZERO 像素的距离 —— 与 cv2.distanceTransform(src, DIST_L2, 3) 匹配。

    cv2 度量每个像素到最近 src==0 像素的距离。我们以 (src==0) 作为源掩码
    (nonzero=源) 喂给 wasm EDT，结果即到最近 src==0 像素的距离。返回 float32 HxW。
    """
    s = np.asarray(src)
    H, W = s.shape[:2]
    seed = (s != 0).astype(np.uint8)  # src 为 0 处作为源
    return _get_core().distance_transform_edt(seed, H, W)


def dilate(mask, kernel, iterations=1):
    """与 cv2.dilate(mask, kernel, iterations) 匹配。

    cv2.dilate 保留输入值域（输出 = 邻域 max），因此 0/1 掩码膨胀后仍是 0/1、
    0/255 仍是 0/255。wasm 核总返回二值 0/1，这里按输入 on-value 重新缩放以保持
    cv2 语义。结构元（rect/ellipse/任意）以位图转发，两端核完全一致。
    """
    m = np.asarray(mask)
    on = int(np.max(m)) if m.size else 0
    k = np.asarray(kernel)
    kh, kw = k.shape[:2]
    kbits = (k != 0).astype(np.uint8)
    H, W = m.shape[:2]
    core = _get_core()
    out = m.astype(np.uint8)
    for _ in range(max(1, int(iterations))):
        out = core.morphology(out, H, W, kbits, kh, kw, "dilate")
    return (out.astype(np.uint8) * on).astype(np.uint8)


def erode(mask, kernel, iterations=1):
    """与 cv2.erode(mask, kernel, iterations) 匹配。值域说明见 dilate()。"""
    m = np.asarray(mask)
    on = int(np.max(m)) if m.size else 0
    k = np.asarray(kernel)
    kh, kw = k.shape[:2]
    kbits = (k != 0).astype(np.uint8)
    H, W = m.shape[:2]
    core = _get_core()
    out = m.astype(np.uint8)
    for _ in range(max(1, int(iterations))):
        out = core.morphology(out, H, W, kbits, kh, kw, "erode")
    return (out.astype(np.uint8) * on).astype(np.uint8)


def morphology_ex(mask, op, kernel):
    """与 cv2.morphologyEx(mask, op, kernel) 匹配。op: cv2.MORPH_OPEN / MORPH_CLOSE。

    注意：用分离式 dilate/erode 组合（与浏览器 cv-bridge 相同），
    而非 cv2 内部 morphologyEx（边界上与分离式最多差 1）—— 这是前后端一致的约定。
    2026-09-04 修正: CLOSE 的旧实现 dilate(erode(x)) 实为「开」的组合, 会吃掉
    1px 细笔画(实测"台"底横/"周"顶横从蒙版漏掉), 已改为教科书语义
    CLOSE=erode(dilate(x))、OPEN=dilate(erode(x)), 并与 shared/src/deglow.rs
    的 mask_close 同步修正, 两端保持逐位一致。
    """
    import cv2
    if op == cv2.MORPH_CLOSE:
        return erode(dilate(mask, kernel), kernel)
    return dilate(erode(mask, kernel), kernel)


def resize_gray_cubic(u8, h2, w2):
    """单通道 uint8，与 cv2.resize(u8, (w2,h2), INTER_CUBIC) 匹配。"""
    u = np.asarray(u8)
    return _get_core().resize_gray_cubic(u, h2, w2)


def resize_float_linear(f32, h2, w2):
    """单通道 float32，与 cv2.resize(float32, (w2,h2), INTER_LINEAR) 匹配。"""
    f = np.asarray(f32, dtype=np.float32)
    return _get_core().resize_float_linear(f, h2, w2)


def patchmatch_inpaint_fill(sub_f32, subm, subsm=None, p: int = 7,
                            direction_deg: float = -1.0, seed: int = 0):
    """共享 PatchMatch 填充（已裁剪 ROI）。patch_fill.inpaint 使用。

    sub_f32 : HxWx3 float32（调用方已完成 pad/ROI 裁剪）。
    subm     : HxW bool/0-255, >0 = 洞。subsm : 可选 HxW 取样区。
    返回填充后的 HxWx3 float32 ROI。
    """
    core = _get_core()
    H, W = sub_f32.shape[:2]
    m = np.ascontiguousarray(subm, dtype=np.uint8)
    m = np.where(m > 0, 255, 0).astype(np.uint8)
    sm = None
    if subsm is not None:
        s = np.ascontiguousarray(subsm, dtype=np.uint8)
        sm = np.where(s > 0, 255, 0).astype(np.uint8)
    return core.patchmatch_inpaint(sub_f32, H, W, m, sm, p, direction_deg, seed)


def smooth_telea_full(rgb, mask, flat_tex: float = 20.0):
    """平滑背景 TELEA 判定 + 填充，与浏览器 `erase_text_glyphs` 平滑分支
    逐字节一致（同一 wasm 实现、相同输入）。

    rgb  : HxWx3 uint8/float 完整图（未 pad）。
    mask : HxW bool/0-255, >0 = 洞（移动边缘后的填充蒙版）。
    返回填充后 HxWx3 uint8 图；平滑分支未触发（tex >= flat_tex → 走 PatchMatch）
    时返回 None。
    """
    core = _get_core()
    H, W = rgb.shape[:2]
    n = H * W
    arr = np.ascontiguousarray(rgb, dtype=np.float32)
    m = np.ascontiguousarray(mask, dtype=np.uint8)
    m = np.where(m > 0, 255, 0).astype(np.uint8)
    p_in = core._alloc(n * 12)
    p_m = core._alloc(n)
    p_out = core._alloc(n * 12)
    try:
        core.mem.write(core.store, arr.tobytes(), p_in)
        core.mem.write(core.store, m.tobytes(), p_m)
        hit = int(core.ex["pm_smooth_telea_full"](
            core.store, p_in, p_m, H, W, float(flat_tex), p_out))
        if hit != 1:
            return None
        filled = np.frombuffer(
            bytes(core.mem.read(core.store, p_out, p_out + n * 12)),
            dtype=np.float32).reshape(H, W, 3).copy()
        return np.clip(filled, 0, 255).astype(np.uint8)
    finally:
        core._free(p_in, n * 12)
        core._free(p_m, n)
        core._free(p_out, n * 12)


def grow_color_tint(rgb, mask, red_thr: int = 30, green_thr: int = 15,
                    green_g: int = 100, rounds_max: int = 120,
                    max_grow_ratio: float = 5.0):
    """共享色偏生长闭合（text_select 蒙版检测的 tint 步骤，逐位镜像 Python 语义）。"""
    core = _get_core()
    H, W = rgb.shape[:2]
    n = H * W
    arr = np.ascontiguousarray(rgb, dtype=np.float32)
    m = np.ascontiguousarray(mask, dtype=np.uint8)
    p_in = core._alloc(n * 12)
    p_m = core._alloc(n)
    p_out = core._alloc(n)
    try:
        core.mem.write(core.store, arr.tobytes(), p_in)
        core.mem.write(core.store, m.tobytes(), p_m)
        core.ex["grow_color_tint"](
            core.store, p_in, p_m, H, W, float(red_thr), float(green_thr),
            float(green_g), int(rounds_max), float(max_grow_ratio), p_out)
        return np.frombuffer(
            bytes(core.mem.read(core.store, p_out, p_out + n)),
            dtype=np.uint8).reshape(H, W).copy()
    finally:
        core._free(p_in, n * 12)
        core._free(p_m, n)
        core._free(p_out, n)


def deglow_full_green_v2(rgb, tmask, strength: float = 1.0, zone_ratio: float = 0.6,
                         zone_expand: int = 0, protect_px: int = 0,
                         chroma_keep: int = 0):
    """共享去发光（整片减绿度 v2）。返回 (clean HxWx3 u8, core_mask HxW u8, zone HxW u8)。"""
    core = _get_core()
    H, W = rgb.shape[:2]
    rgb_f = np.ascontiguousarray(rgb, dtype=np.float32)
    tm = np.ascontiguousarray(tmask, dtype=np.uint8)
    return core.deglow_full_green_v2(rgb_f, H, W, tm, strength, zone_ratio,
                                     zone_expand, protect_px, chroma_keep)


def erase_text_glyphs(rgb, tmask, tmask2=None, strength: float = 1.0,
                      zone_ratio: float = 0.6, zone_expand: int = 0,
                      protect_px: int = 0, chroma_keep: int = 0, edge: int = 0,
                      direction_deg: float = -1.0, seed: int = 0,
                      edge_aware: int = 0, soft_expand: float = 0.0):
    """单一共享管线入口 —— 去发光 + 蒙版修复 + PatchMatch 填充一次完成
    （浏览器与后端逐字节一致）。

    rgb    : HxWx3 uint8。
    tmask  : HxW, >0 = 原图文字检测蒙版。tmask2 : 可选 HxW，去发光图上的检测蒙版。
    返回 (result HxWx3 u8, fill HxW u8, clean HxWx3 u8, zone HxW u8)。

    NOTE: 这是发光管线的唯一权威实现。文字*检测*（DBNet）不属于共享核 ——
    后端/浏览器各自推理后，把两张蒙版交给本算子。
    """
    core = _get_core()
    H, W = rgb.shape[:2]
    rgb_f = np.ascontiguousarray(rgb, dtype=np.float32)
    tm = np.ascontiguousarray(tmask, dtype=np.uint8)
    tm2 = None
    if tmask2 is not None:
        tm2 = np.ascontiguousarray(tmask2, dtype=np.uint8)
    return core.erase_text_glyphs(rgb_f, H, W, tm, tm2, strength, zone_ratio,
                                  zone_expand, protect_px, chroma_keep, edge,
                                  direction_deg, seed, edge_aware, soft_expand)
