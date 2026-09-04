import numpy as np

def compute_gradient(img, H, W):
    graf = np.zeros((H, W, 2), np.float32)
    for y in range(H):
        for x in range(W):
            ym = y-1 if y > 0 else 0
            yp = y+1 if y+1 < H else H-1
            xm = x-1 if x > 0 else 0
            xp = x+1 if x+1 < W else W-1
            gx = 0.0; gy = 0.0
            for c in range(3):
                gx += 0.5*(img[yp if False else y, xp, c] - img[y, xm, c])
                gy += 0.5*(img[yp, x, c] - img[ym, x, c])
            graf[y, x, 0] = gx
            graf[y, x, 1] = gy
    return graf

def telea_ref(img, mask, H, W, radius=3):
    out = img.astype(np.float32).copy()
    seed = (mask == 0).astype(np.uint8) * 255  # known=255
    # edt: distance from each pixel to nearest seed (known). Use scipy-like via cv2
    import cv2
    dist = cv2.distanceTransform(seed, cv2.DIST_L2, 5)
    graf = compute_gradient(img, H, W)
    holes = np.where(mask != 0)
    holes = list(zip(holes[0], holes[1]))
    holes.sort(key=lambda p: dist[p])
    dx8 = [1, 1, 0, -1, -1, -1, 0, 1]
    dy8 = [0, 1, 1, 1, 0, -1, -1, -1]
    for (py, px) in holes:
        pnx = graf[py, px, 0]; pny = graf[py, px, 1]
        pnlen = (pnx**2 + pny**2)**0.5
        if pnlen > 1e-5: pn_x, pn_y = pnx/pnlen, pny/pnlen
        else: pn_x, pn_y = 0.0, 0.0
        length = 0.0
        numer = [0.0, 0.0, 0.0]
        for m in range(8):
            nx = px + dx8[m]; ny = py + dy8[m]
            if nx < 0 or nx >= W or ny < 0 or ny >= H: continue
            ni = (ny, nx)
            if mask[ny, nx] != 0: continue
            qx = px - nx; qy = py - ny
            qlen = (qx**2 + qy**2)**0.5
            ngx = graf[ny, nx, 0]; ngy = graf[ny, nx, 1]
            nglen = (ngx**2 + ngy**2)**0.5
            if nglen > 1e-5: ngn_x, ngn_y = ngx/nglen, ngy/nglen
            else: ngn_x, ngn_y = 0.0, 0.0
            wgt = (1.0/qlen) * (1.0 + (ngn_x*qx + ngn_y*qy)/qlen) * (1.0 + (pn_x*ngn_x + pn_y*ngn_y))
            if wgt < 0: continue
            for c in range(3):
                nval = img[ny, nx, c] + (ngx*qx + ngy*qy)
                numer[c] += wgt * nval
            length += wgt
        if length > 0:
            for c in range(3):
                out[py, px, c] = numer[c]/length
    return out

# tiny case
H, W = 8, 8
g = np.linspace(30, 230, W, dtype=np.float32)
img = np.stack([g, g, g], axis=-1)[None].repeat(H, 0).copy()
mask = np.zeros((H, W), np.uint8); mask[3:5, 3:5] = 255
res = telea_ref(img, mask, H, W)
print("REF hole (3,3) ch0:", res[3,3,0])
print("REF hole (3,4) ch0:", res[3,4,0])
print("REF hole (4,3) ch0:", res[4,3,0])
print("REF hole (4,4) ch0:", res[4,4,0])
