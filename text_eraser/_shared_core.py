"""Backend integration point for the shared WASM algorithm core (textcore.wasm).

This module is the SINGLE place the Python backend calls the shared operators. It loads
the *same* .wasm the browser Worker loads (`shared/build/textcore.wasm`), so when this
module is used, the backend runs the exact same algorithms as the browser — that is the
"前后端共用一套算法" (frontend/backend share one algorithm set) guarantee.

Every helper tries the wasm core first and transparently falls back to a cv2/numpy
implementation on any failure (missing .wasm, wasmtime not installed, operator error),
so existing behavior is preserved if the core cannot be loaded.

Toggle: set environment `TEXTCORE_BACKEND=0` to disable wasm and force the cv2 fallback
for the whole backend. By default wasm is used when available.
"""
from __future__ import annotations

import os
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except Exception:  # pragma: no cover - cv2 is a hard dep of the backend
    _HAS_CV2 = False

_USE_WASM = os.environ.get("TEXTCORE_BACKEND", "1") != "0"

_core = None
_core_ok = None


def _get_core():
    """Return the loaded wasm core, or None if wasm is disabled / failed to load."""
    global _core, _core_ok
    if _core_ok is not None:
        return _core if _core_ok else None
    if not _USE_WASM:
        _core_ok = False
        return None
    try:
        from shared.bindings.textcore import get_core
        _core = get_core()
        _core_ok = True
    except Exception:
        _core = None
        _core_ok = False
    return _core if _core_ok else None


def using_shared_core() -> bool:
    """True if the backend is currently dispatching through the wasm core."""
    return _get_core() is not None


# ---------------------------------------------------------------------------
# Operators (drop-in cv2-compatible)
# ---------------------------------------------------------------------------
def rgb2gray(rgb):
    """RGB HxWx3 (uint8/float) -> uint8 HxW, matching cv2.cvtColor(COLOR_RGB2GRAY)."""
    arr = np.asarray(rgb)
    if arr.ndim == 3:
        arr = arr[..., :3]
    core = _get_core()
    if core is not None:
        try:
            H, W = arr.shape[:2]
            return core.rgb_to_gray(arr, H, W)
        except Exception:
            pass
    if _HAS_CV2:
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    f = arr.astype(np.float32)
    return np.clip((f[..., 0] * 0.299 + f[..., 1] * 0.587 + f[..., 2] * 0.114), 0, 255).astype(np.uint8)


def threshold_otsu(gray, *_, **__):
    """Otsu threshold of a single-channel array. Returns (thr: float, bin: uint8 0/255)."""
    g = np.asarray(gray)
    core = _get_core()
    if core is not None:
        try:
            H, W = g.shape[:2]
            thr, bin = core.threshold_otsu(g.astype(np.uint8), H, W)
            return float(thr), bin
        except Exception:
            pass
    if _HAS_CV2:
        thr, bin = cv2.threshold(g.astype(np.uint8), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return float(thr), bin
    # fallback (rare): simple inter-means
    hist = np.bincount(g.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    sumv = (np.arange(256) * hist).sum()
    sum_b = w_b = 0.0
    best, bthr = -1.0, 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sumv - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between > best:
            best, bthr = between, t
    return float(bthr), (g > bthr).astype(np.uint8) * 255


def connected_components(mask, connectivity=8):
    """Match cv2.connectedComponents(mask, connectivity). Returns (n, labels HxW int32)."""
    m = np.asarray(mask)
    core = _get_core()
    if core is not None:
        try:
            H, W = m.shape[:2]
            n, labels, _stats = core.connected_components(m.astype(np.uint8), H, W)
            return n, labels
        except Exception:
            pass
    if _HAS_CV2:
        return cv2.connectedComponents(m.astype(np.uint8), connectivity=connectivity)
    # minimal fallback
    from scipy import ndimage
    labels, n = ndimage.label(m.astype(np.uint8))
    return int(n), labels.astype(np.int32)


def connected_components_with_stats(mask, connectivity=8):
    """Match cv2.connectedComponentsWithStats. Returns (n, labels, stats(n,5), centroids(n,2)).

    stats columns follow cv2: [CC_STAT_LEFT, CC_STAT_TOP, CC_STAT_WIDTH,
    CC_STAT_HEIGHT, CC_STAT_AREA] = [0,1,2,3,4].
    """
    m = np.asarray(mask)
    core = _get_core()
    if core is not None:
        try:
            H, W = m.shape[:2]
            n, labels, stats = core.connected_components(m.astype(np.uint8), H, W)
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
        except Exception:
            pass
    if _HAS_CV2:
        return cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=connectivity)
    n, labels = connected_components(m, connectivity)
    stat_arr = np.zeros((n, 5), np.int32)
    centroids = np.zeros((n, 2), np.float64)
    return n, labels, stat_arr, centroids


def edt_to_nearest_zero(src):
    """Distance to nearest ZERO pixel — matches cv2.distanceTransform(src, DIST_L2, 3).

    cv2 measures distance from each pixel to the nearest src==0 pixel. We seed the
    wasm EDT with (src==0) as the source mask (nonzero=source), so the result is the
    distance to the nearest src==0 pixel. Returns float32 HxW.
    """
    s = np.asarray(src)
    core = _get_core()
    if core is not None:
        try:
            H, W = s.shape[:2]
            seed = (s != 0).astype(np.uint8)  # nonzero where src is ZERO (the sources)
            return core.distance_transform_edt(seed, H, W)
        except Exception:
            pass
    if _HAS_CV2:
        return cv2.distanceTransform(s.astype(np.uint8), cv2.DIST_L2, 3)
    from scipy import ndimage
    return ndimage.distance_transform_edt(s == 0).astype(np.float32)


def dilate(mask, kernel, iterations=1):
    """Match cv2.dilate(mask, kernel, iterations).

    cv2.dilate preserves the input's value range (output = max over neighborhood), so a
    0/1 mask dilates to 0/1 and a 0/255 mask to 0/255. The wasm core always returns a
    binary 0/1 mask, so we re-scale by the input's on-value to stay cv2-faithful. The
    exact structuring element (rect/ellipse/arbitrary) is forwarded as a bitmap so both
    ends honor the identical kernel.
    """
    m = np.asarray(mask)
    on = int(np.max(m)) if m.size else 0
    core = _get_core()
    if core is not None:
        try:
            k = np.asarray(kernel)
            kh, kw = k.shape[:2]
            kbits = (k != 0).astype(np.uint8)
            H, W = m.shape[:2]
            out = m.astype(np.uint8)
            for _ in range(max(1, int(iterations))):
                out = core.morphology(out, H, W, kbits, kh, kw, "dilate")
            return (out.astype(np.uint8) * on).astype(np.uint8)
        except Exception:
            pass
    if _HAS_CV2:
        return cv2.dilate(m.astype(np.uint8), np.asarray(kernel), iterations=iterations)
    # fallback: simple box dilate
    from scipy import ndimage
    return (ndimage.binary_dilation(m.astype(bool), iterations=iterations).astype(np.uint8) * on)


def erode(mask, kernel, iterations=1):
    """Match cv2.erode(mask, kernel, iterations). See dilate() for value-range note."""
    m = np.asarray(mask)
    on = int(np.max(m)) if m.size else 0
    core = _get_core()
    if core is not None:
        try:
            k = np.asarray(kernel)
            kh, kw = k.shape[:2]
            kbits = (k != 0).astype(np.uint8)
            H, W = m.shape[:2]
            out = m.astype(np.uint8)
            for _ in range(max(1, int(iterations))):
                out = core.morphology(out, H, W, kbits, kh, kw, "erode")
            return (out.astype(np.uint8) * on).astype(np.uint8)
        except Exception:
            pass
    if _HAS_CV2:
        return cv2.erode(m.astype(np.uint8), np.asarray(kernel), iterations=iterations)
    from scipy import ndimage
    return (ndimage.binary_erosion(m.astype(bool), iterations=iterations).astype(np.uint8) * on)


def morphology_ex(mask, op, kernel):
    """Match cv2.morphologyEx(mask, op, kernel). op: cv2.MORPH_OPEN / MORPH_CLOSE."""
    if _HAS_CV2 and op == cv2.MORPH_CLOSE:
        return dilate(erode(mask, kernel), kernel)
    return erode(dilate(mask, kernel), kernel)


def resize_gray_cubic(u8, h2, w2):
    """Match cv2.resize(u8, (w2,h2), INTER_CUBIC) for a single-channel array."""
    u = np.asarray(u8)
    h, w = u.shape[:2]
    core = _get_core()
    if core is not None:
        try:
            return core.resize_gray_cubic(u, h2, w2)
        except Exception:
            pass
    if _HAS_CV2:
        return cv2.resize(u, (w2, h2), interpolation=cv2.INTER_CUBIC)
    from scipy import ndimage
    return ndimage.zoom(u, (h2 / h, w2 / w), order=3).astype(np.uint8)


def resize_float_linear(f32, h2, w2):
    """Match cv2.resize(float32, (w2,h2), INTER_LINEAR) for a single-channel array."""
    f = np.asarray(f32, dtype=np.float32)
    h, w = f.shape[:2]
    core = _get_core()
    if core is not None:
        try:
            return core.resize_float_linear(f, h2, w2)
        except Exception:
            pass
    if _HAS_CV2:
        return cv2.resize(f, (w2, h2), interpolation=cv2.INTER_LINEAR)
    from scipy import ndimage
    return ndimage.zoom(f, (h2 / h, w2 / w), order=1).astype(np.float32)


def patchmatch_inpaint_fill(sub_f32, subm, subsm=None, p: int = 7,
                           direction_deg: float = -1.0, seed: int = 0):
    """Shared PatchMatch fill of a prepared ROI. Used by patch_fill.inpaint.

    sub_f32 : HxWx3 float32 (the ROI, already padded/ROI-cropped by the caller).
    subm     : HxW bool/0-255, >0 = hole. subsm : optional HxW sample region.
    Returns the filled HxWx3 float32 ROI, or None if the core is unavailable
    (caller should then run its own cv2/numpy fallback).
    """
    core = _get_core()
    if core is None:
        return None
    try:
        H, W = sub_f32.shape[:2]
        m = np.ascontiguousarray(subm, dtype=np.uint8)
        m = np.where(m > 0, 255, 0).astype(np.uint8)
        sm = None
        if subsm is not None:
            s = np.ascontiguousarray(subsm, dtype=np.uint8)
            sm = np.where(s > 0, 255, 0).astype(np.uint8)
        return core.patchmatch_inpaint(sub_f32, H, W, m, sm, p, direction_deg, seed)
    except Exception:
        return None


def deglow_full_green_v2(rgb, tmask, strength: float = 1.0, zone_ratio: float = 0.6,
                         zone_expand: int = 0, protect_px: int = 0,
                         chroma_keep: int = 0):
    """Shared de-glow (full green v2). Returns (clean HxWx3 u8, core_mask HxW u8)
    or None if the core is unavailable (caller should run its cv2 fallback)."""
    core = _get_core()
    if core is None:
        return None
    try:
        H, W = rgb.shape[:2]
        rgb_f = np.ascontiguousarray(rgb, dtype=np.float32)
        tm = np.ascontiguousarray(tmask, dtype=np.uint8)
        return core.deglow_full_green_v2(rgb_f, H, W, tm, strength, zone_ratio,
                                         zone_expand, protect_px, chroma_keep)
    except Exception:
        return None


def erase_text_glyphs(rgb, tmask, tmask2=None, strength: float = 1.0,
                      zone_ratio: float = 0.6, zone_expand: int = 0,
                      protect_px: int = 0, chroma_keep: int = 0, edge: int = 0,
                      direction_deg: float = -1.0, seed: int = 0):
    """Single shared pipeline entry — run the FULL de-glow + mask-surgery +
    PatchMatch fill (browser + backend call this identically).

    rgb    : HxWx3 uint8.
    tmask  : HxW, >0 = raw text detect. tmask2 : optional HxW, >0 = detect on clean.
    Returns (result HxWx3 u8, fill HxW u8, clean HxWx3 u8, zone HxW u8) or None.

    NOTE: this is the SINGLE source of truth for the glow pipeline. The backend
    still performs text *detection* (cv2 detect_text_mask) on both the raw image
    and the de-glowed image — detection is not part of the shared core — and hands
    the resulting masks to this operator.
    """
    core = _get_core()
    if core is None:
        return None
    try:
        H, W = rgb.shape[:2]
        rgb_f = np.ascontiguousarray(rgb, dtype=np.float32)
        tm = np.ascontiguousarray(tmask, dtype=np.uint8)
        tm2 = None
        if tmask2 is not None:
            tm2 = np.ascontiguousarray(tmask2, dtype=np.uint8)
        return core.erase_text_glyphs(rgb_f, H, W, tm, tm2, strength, zone_ratio,
                                      zone_expand, protect_px, chroma_keep, edge,
                                      direction_deg, seed)
    except Exception:
        return None
