"""Bisect D_rg's e_up. Two checks:
(1) Rust es_mask (resize_mask) vs cv2 INTER_NEAREST ring mask — if they differ,
    that's why D_rg diverges (sparse ring routing is mask-sensitive).
(2) e_up using Rust's OWN es_mask fed to Python's _geodesic_sources — if THIS
    matches Rust's e_up, then routing+resize are correct and the D_rg gap is
    purely the resize_mask vs cv2 discrepancy."""
import os, json
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deglow_io")
IDS = ["1787822778556"]


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


for hid in IDS:
    orig = cv2.imdecode(np.frombuffer(open(os.path.join(ROOT, "data", "history", hid, "orig.bin"), "rb").read(), np.uint8), cv2.IMREAD_COLOR)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    H, W = orig.shape[:2]
    r = orig[..., 0].astype(np.float32)
    g = orig[..., 1].astype(np.float32)
    b = orig[..., 2].astype(np.float32)
    gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY).astype(np.float32)
    scale = 4 if min(H, W) >= 160 else 2
    h2, w2 = max(2, H // scale), max(2, W // scale)
    lum = cv2.resize(gray, (w2, h2), interpolation=cv2.INTER_AREA)
    rg = (r - g).astype(np.float32)
    e_s = cv2.resize(rg, (w2, h2), interpolation=cv2.INTER_AREA)
    zone = np.fromfile(os.path.join(IO, f"{hid}_zone.bin"), dtype=np.uint8).reshape(H, W)
    k3 = np.ones((3, 3), np.uint8)
    geo_mask = cv2.erode(zone.astype(np.uint8), k3, iterations=3) > 0
    dout = cv2.distanceTransform((~zone).astype(np.uint8), cv2.DIST_L2, 5)
    greenness = np.maximum(g.astype(np.int16) - np.maximum(r, b).astype(np.int16), 0).astype(np.float32)
    ring_clean = (~zone) & (dout >= 10.0) & (dout <= 26.0) & (greenness <= 6)
    es_cv = cv2.resize(ring_clean.astype(np.uint8) * 255, (w2, h2), interpolation=cv2.INTER_NEAREST) > 127
    es_rust = np.fromfile(os.path.join(IO, f"{hid}_eup_esmask.bin"), dtype=np.uint8).reshape(h2, w2) > 0

    # (1) mask discrepancy
    md = np.abs(es_rust.astype(np.int16) - es_cv.astype(np.int16))
    print(f"=== {hid} ===")
    print(f"  [es_mask] rust_vs_cv2_nn mismatch={int((md>0).sum())}/{h2*w2} ({100*(md>0).mean():.2f}%) n_src_rust={int(es_rust.sum())} n_src_cv={int(es_cv.sum())}")

    # (2) e_up with RUST's own mask -> compare to Rust's dumped e_up
    ey, ex = geodesic_sources(lum, es_rust)
    e_up_py = e_s[ey, ex]
    e_up_rs = np.fromfile(os.path.join(IO, f"{hid}_eup.bin"), dtype=np.float32).reshape(h2, w2)
    d = np.abs(e_up_rs.astype(np.float64) - e_up_py.astype(np.float64))
    print(f"  [e_up via rust mask] max={d.max():.3f} mean={d.mean():.4f} #>1={int((d>1).sum())}")
print("done")
