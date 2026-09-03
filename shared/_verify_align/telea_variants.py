import numpy as np, cv2

H, W = 5, 5
# linear horizontal gray gradient; known everywhere; single hole at (2,2)
img = np.zeros((H, W, 3), np.uint8)
g = np.tile(np.arange(W, dtype=np.float32), (H, 1)) * 20.0 + 30.0  # 30..130
for c in range(3):
    img[:, :, c] = np.clip(g, 0, 255).astype(np.uint8)
mask = np.zeros((H, W), np.uint8)
mask[2, 2] = 255  # hole (cv2 mask: nonzero = hole)

cv2_res = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
cv2_val = cv2_res[2, 2].astype(np.float64)
print("cv2 TELEA (2,2) =", cv2_val)

def gray_grad(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    # central difference
    gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
    gy[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5
    return gx, gy

def dist_transform_edt(seed):
    # exact EDT via scipy-like; use cv2 mask5 as proxy for the binary seed (seed = hole)
    d = cv2.distanceTransform((seed == 0).astype(np.uint8), cv2.DIST_L2, 5)
    return d

def telea_variant(img, mask, variant):
    H, W = img.shape[:2]
    result = img.astype(np.float32).copy()
    known = (mask == 0)
    gx, gy = gray_grad(img)
    gn = np.sqrt(gx**2 + gy**2) + 1e-8
    nnx = gx / gn; nny = gy / gn  # normalized gradient at each pixel
    seed = mask.copy()  # hole=255
    T = dist_transform_edt(seed)  # distance from hole; T=0 at hole, >0 known
    # gradient of T (distance field) at each pixel
    tx = np.zeros_like(T); ty = np.zeros_like(T)
    tx[:, 1:-1] = (T[:, 2:] - T[:, :-2]) * 0.5
    ty[1:-1, :] = (T[2:, :] - T[:-2, :]) * 0.5
    tn = np.sqrt(tx**2 + ty**2) + 1e-8
    fnx = tx / tn; fny = ty / tn  # normalized T-gradient
    # FMM: process in order of increasing T (band loop)
    order = np.argsort(T.reshape(-1))
    for idx in order:
        y, x = divmod(idx, W)
        if known[y, x]:
            continue
        res = np.zeros(3, np.float64); wsum = 0.0
        for (dx, dy) in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]:
            nx_, ny_ = x+dx, y+dy
            if 0 <= nx_ < W and 0 <= ny_ < H and known[ny_, nx_]:
                qx = float(dx); qy = float(dy)
                qlen = np.sqrt(qx*qx + qy*qy)
                nxv = nnx[ny_, nx_]; nyv = nny[ny_, nx_]
                ndotq = nxv*qx + nyv*qy
                if variant == 0:
                    t = ndotq / qlen                       # (n·q)/|q|
                    third = 1.0 + (nxv*fnx[y,x] + nyv*fny[y,x])
                elif variant == 1:
                    t = ndotq / (qlen*qlen)               # (n·q)/|q|^2
                    third = 1.0 + (nxv*fnx[y,x] + nyv*fny[y,x])
                elif variant == 2:
                    t = ndotq / qlen
                    third = 1.0 + abs(nxv*fnx[y,x] + nyv*fny[y,x])
                elif variant == 3:
                    t = ndotq / (qlen*qlen)
                    third = 1.0 + abs(nxv*fnx[y,x] + nyv*fny[y,x])
                elif variant == 4:
                    # use n at p (hole) instead of neighbor: average known neighbors' n
                    t = ndotq / qlen
                    third = 1.0 + abs(nxv*fnx[y,x] + nyv*fny[y,x])
                elif variant == 5:
                    # no third factor
                    t = ndotq / qlen
                    third = 1.0
                elif variant == 6:
                    # pure distance weight
                    t = 0.0
                    third = 1.0
                w = float((1.0/qlen) * (1.0 + t) * third)
                if w < 0: w = 0.0
                res += w * result[ny_, nx_]
                wsum += w
        if wsum > 0:
            result[y, x] = res / wsum
    return result[2, 2]

for v in range(7):
    val = telea_variant(img, mask, v)
    print(f"variant {v}: {val}")
