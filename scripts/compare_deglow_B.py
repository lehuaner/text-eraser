# -*- coding: utf-8 -*-
"""B 场跨端逐子层对比 (handover §13.5-1c)。

读 deglow_case.bin → wasm deglow_debug_geodesic 全中间量 → 与 cv2 对应子层对比。
"""
import sys
import struct
import heapq
import numpy as np
import cv2

sys.path.insert(0, r"D:/Code/Project/Python/TextPatch")
from shared.bindings.textcore import get_core

OUT = r"D:/Code/Project/Python/TextPatch/data/_pmparity"


def dijkstra_py(lum, src_mask):
    """1:1 复刻 text_eraser/text_select._geodesic_sources。"""
    H2, W2 = lum.shape
    INF = float("inf")
    dist = np.full((H2, W2), INF, np.float32)
    src_y = np.zeros((H2, W2), np.int32)
    src_x = np.zeros((H2, W2), np.int32)
    heap = []
    ys, xs = np.nonzero(src_mask)
    for yy, xx in zip(ys, xs):
        dist[int(yy), int(xx)] = 0.0
        src_y[int(yy), int(xx)] = int(yy)
        src_x[int(yy), int(xx)] = int(xx)
        heap.append((0.0, int(yy), int(xx)))
    heapq.heapify(heap)
    while heap:
        d, y, x = heapq.heappop(heap)
        if d > dist[y, x]:
            continue
        sy, sx = src_y[y, x], src_x[y, x]
        lv = lum[y, x]
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if ny < 0 or ny >= H2 or nx < 0 or nx >= W2:
                continue
            nd = np.float32(d + 1.0 + 3.0 * abs(lum[ny, nx] - lv))
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                src_y[ny, nx] = sy
                src_x[ny, nx] = sx
                heapq.heappush(heap, (float(nd), int(ny), int(nx)))
    return src_y, src_x


def stat(name, a, b):
    bit = np.array_equal(a, b) if a.dtype.kind in "bu" else \
        np.array_equal(a.view(np.uint32) if a.dtype == np.float32 else a.view(np.int32), b.view(np.uint32) if b.dtype == np.float32 else b.view(np.int32))
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    nz = d > 0
    print(f"{name:8s}: bit={bit} diff={int(nz.sum())}/{a.size} max={d.max():.6g}")


def main():
    with open(rf"{OUT}/deglow_case.bin", "rb") as f:
        buf = f.read()
    h, w = struct.unpack("<ii", buf[:8])
    n = h * w
    scale = 4 if min(h, w) >= 160 else 2
    h2, w2 = max(2, h // scale), max(2, w // scale)
    n2 = h2 * w2
    off = 8
    rgb = np.frombuffer(buf[off:off + n * 12], np.float32).reshape(h, w, 3).copy(); off += n * 12
    zone = np.frombuffer(buf[off:off + n], np.uint8).copy(); off += n
    ring = np.frombuffer(buf[off:off + n], np.uint8).copy()

    # ---- call wasm hook ----
    core = get_core()
    out_size = 4 * (n2 + n2 * 3 + n2 + n2 + n2 * 3 + n * 3 + n * 3) + n2 + 4 * (n2 + n2 + n + n)
    p_in = core._alloc(n * 12)
    p_z = core._alloc(n)
    p_r = core._alloc(n)
    p_o = core._alloc(out_size)
    assert p_o > 0
    try:
        core.mem.write(core.store, rgb.tobytes(), p_in)
        core.mem.write(core.store, zone.tobytes(), p_z)
        core.mem.write(core.store, ring.tobytes(), p_r)
        core.ex["deglow_debug_geodesic"](core.store, p_in, h, w, p_z, p_r, p_o)
        raw = bytes(core.mem.read(core.store, p_o, p_o + out_size))
    finally:
        for p, sz in [(p_in, n * 12), (p_z, n), (p_r, n), (p_o, out_size)]:
            core._free(p, sz)

    o = 0
    def take_f32(k, shape):
        nonlocal o
        a = np.frombuffer(raw[o:o + k * 4], np.float32).copy(); o += k * 4
        return a.reshape(shape)
    def take_i32(k, shape):
        nonlocal o
        a = np.frombuffer(raw[o:o + k * 4], np.int32).copy(); o += k * 4
        return a.reshape(shape)
    def take_u8(k, shape):
        nonlocal o
        a = np.frombuffer(raw[o:o + k], np.uint8).copy(); o += k
        return a.reshape(shape)

    lum_rs = take_f32(n2, (h2, w2))
    rz_rs = take_u8(n2, (h2, w2))
    rgb_s_rs = take_f32(n2 * 3, (h2, w2, 3))
    sy_rs = take_i32(n2, (h2, w2))
    sx_rs = take_i32(n2, (h2, w2))
    b_s_rs = take_f32(n2 * 3, (h2, w2, 3))
    b_up_rs = take_f32(n * 3, (h, w, 3))
    b_sm_rs = take_f32(n * 3, (h, w, 3))
    e_s_rg_rs = take_f32(n2, (h2, w2))
    e_s_gb_rs = take_f32(n2, (h2, w2))
    e_f_rg_rs = take_f32(n, (h, w))
    e_f_gb_rs = take_f32(n, (h, w))

    # ---- cv2 references ----
    rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lum_cv = cv2.resize(gray, (w2, h2), interpolation=cv2.INTER_AREA)
    rz_cv = cv2.resize(zone.astype(np.uint8) * 255, (w2, h2),
                       interpolation=cv2.INTER_NEAREST) > 127
    rgb_s_cv = cv2.resize(rgb_u8, (w2, h2), interpolation=cv2.INTER_AREA)
    src_mask = ~rz_cv
    sy_cv, sx_cv = dijkstra_py(lum_cv, src_mask)
    b_s_cv = rgb_s_cv[sy_cv, sx_cv].astype(np.float32)
    b_up_cv = cv2.resize(b_s_cv, (w, h), interpolation=cv2.INTER_CUBIC)
    b_sm_cv = cv2.GaussianBlur(b_up_cv, (0, 0), 4.0)
    # extras
    r16 = rgb_u8[..., 0].astype(np.int16); g16 = rgb_u8[..., 1].astype(np.int16); b16 = rgb_u8[..., 2].astype(np.int16)
    rg_f = (r16 - g16).astype(np.float32); gb_f = (g16 - b16).astype(np.float32)
    ring_bool = ring > 0
    es_mask = cv2.resize(ring_bool.astype(np.uint8) * 255, (w2, h2),
                         interpolation=cv2.INTER_NEAREST) > 127
    ey, ex = dijkstra_py(lum_cv, es_mask)
    e_s_rg_cv = cv2.resize(rg_f, (w2, h2), interpolation=cv2.INTER_AREA)[ey, ex]
    e_s_gb_cv = cv2.resize(gb_f, (w2, h2), interpolation=cv2.INTER_AREA)[ey, ex]
    e_f_rg_cv = cv2.resize(e_s_rg_cv, (w, h), interpolation=cv2.INTER_CUBIC)
    e_f_gb_cv = cv2.resize(e_s_gb_cv, (w, h), interpolation=cv2.INTER_CUBIC)

    print(f"img {w}x{h}  low-res {w2}x{h2}")
    stat("lum", lum_cv, lum_rs)
    stat("rz", rz_cv.astype(np.uint8), (rz_rs > 0).astype(np.uint8))
    stat("rgb_s", rgb_s_cv.astype(np.float32), rgb_s_rs)
    stat("sy", sy_cv, sy_rs)
    stat("sx", sx_cv, sx_rs)
    stat("b_s", b_s_cv, b_s_rs)
    stat("b_up", b_up_cv, b_up_rs)
    stat("b_sm", b_sm_cv, b_sm_rs)
    stat("e_s_rg", e_s_rg_cv, e_s_rg_rs)
    stat("e_s_gb", e_s_gb_cv, e_s_gb_rs)
    stat("e_f_rg", e_f_rg_cv, e_f_rg_rs)
    stat("e_f_gb", e_f_gb_cv, e_f_gb_rs)


if __name__ == "__main__":
    main()
