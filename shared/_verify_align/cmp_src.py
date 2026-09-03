"""Isolate routing + single-channel resize_area for BOTH background (mode 0, used
by B) and ring (mode 1, used by D_rg/D_gb). Feed Rust's dumped (lum, src_mask)
into Python's exact `_geodesic_sources` and compare (sy,sx). Also compare the
single-channel resize_area(rg) against cv2.resize(rg, INTER_AREA)."""
import os, json
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deglow_io")
IDS = ["1787767611178", "1787822778556"]


def geodesic_sources(lum, src_mask):
    import heapq
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
            nd = d + 1.0 + 3.0 * abs(lum[ny, nx] - lv)
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                src_y[ny, nx] = sy
                src_x[ny, nx] = sx
                heapq.heappush(heap, (nd, int(ny), int(nx)))
    return src_y, src_x


def load_i32(path, n):
    return np.fromfile(path, dtype="<i4", count=n)


def load_f32(path, n):
    return np.fromfile(path, dtype="<f4", count=n)


def load_u8(path, n):
    return np.fromfile(path, dtype=np.uint8, count=n)


def cmp_routing(hid, h2, w2, mask_name, sy_name, sx_name, label):
    mask = load_u8(os.path.join(IO, f"{hid}_{mask_name}.bin"), h2 * w2).reshape(h2, w2)
    rsy = load_i32(os.path.join(IO, f"{hid}_{sy_name}.bin"), h2 * w2).reshape(h2, w2)
    rsx = load_i32(os.path.join(IO, f"{hid}_{sx_name}.bin"), h2 * w2).reshape(h2, w2)
    lum = load_f32(os.path.join(IO, f"{hid}_src_lum.bin"), h2 * w2).reshape(h2, w2)
    psy, psx = geodesic_sources(lum, mask > 0)
    dy = np.abs(rsy.astype(np.int64) - psy.astype(np.int64))
    dx = np.abs(rsx.astype(np.int64) - psx.astype(np.int64))
    print(f"  [{label}] n_src={int((mask>0).sum())} sy_mismatch={int((dy>0).sum())}/{h2*w2} sy_max={dy.max()}")
    print(f"          sx_mismatch={int((dx>0).sum())}/{h2*w2} sx_max={dx.max()}")


for hid in IDS:
    meta = json.load(open(os.path.join(IO, f"{hid}_meta.json")))
    H, W = meta["H"], meta["W"]
    scale = 4 if min(H, W) >= 160 else 2
    h2 = max(2, H // scale)
    w2 = max(2, W // scale)
    print(f"=== {hid} h2,w2={h2},{w2} ===")
    cmp_routing(hid, h2, w2, "src_mask", "src_sy", "src_sx", "background(B)")
    cmp_routing(hid, h2, w2, "rng_mask", "rng_sy", "rng_sx", "ring(D_rg)")

    # single-channel resize_area of (r-g)
    rg = load_f32(os.path.join(IO, f"{hid}_rg_rsz.bin"), h2 * w2).reshape(h2, w2)
    orig = cv2.imdecode(np.frombuffer(open(os.path.join(ROOT, "data", "history", hid, "orig.bin"), "rb").read(), np.uint8), cv2.IMREAD_COLOR)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    rg_py = (orig[..., 0].astype(np.float32) - orig[..., 1].astype(np.float32))
    rg_cv = cv2.resize(rg_py, (w2, h2), interpolation=cv2.INTER_AREA)
    d = np.abs(rg.astype(np.float64) - rg_cv.astype(np.float64))
    print(f"  [resize_area rg ch=1] max={d.max():.3f} mean={d.mean():.4f} #>1={int((d>1).sum())}")
print("done")
