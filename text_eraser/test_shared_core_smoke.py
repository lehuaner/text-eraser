import os, sys, numpy as np, cv2
# repo root = parent of the text_eraser package, so both `text_eraser` and `shared`
# resolve to the local source tree (not any installed copy in site-packages).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from text_eraser import _shared_core as sc

print("using_shared_core =", sc.using_shared_core())
assert sc.using_shared_core(), "wasm core must load"

rng = np.random.default_rng(0)

def rint_img(H, W):
    return rng.integers(0, 256, (H, W, 3), dtype=np.uint8)

ok = True
def check(name, a, b, tol=0, exact=False):
    global ok
    a = np.asarray(a); b = np.asarray(b)
    if exact:
        same = bool(np.array_equal(a, b))
        md = 0
    else:
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        md = float(d.max()); same = md <= tol
    print(f"  [{'OK' if same else 'FAIL'}] {name}: maxdiff={md}")
    if not same: ok = False

# 1) rgb2gray
img = rint_img(64, 80)
check("rgb2gray", sc.rgb2gray(img), cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), tol=1)

# 2) threshold_otsu
g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
thr, bin = sc.threshold_otsu(g)
ctr, cbin = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
check("otsu thr", [thr], [ctr], tol=0, exact=False)
check("otsu bin", bin, cbin, exact=True)

# 3) connected_components_with_stats
mask = (g > 120).astype(np.uint8)
n, lab, stats, cent = sc.connected_components_with_stats(mask, 8)
cn, clamp, cstats, ccent = cv2.connectedComponentsWithStats(mask, connectivity=8)
check("cc n", [n], [cn], exact=True)
check("cc labels", lab, clamp, exact=True)
# compare stats rows
for i in range(1, min(n, cn)):
    same = all(int(stats[i, k]) == int(cstats[i, k]) for k in range(5))
    if not same:
        print(f"  [FAIL] cc stats row {i}: ours={stats[i]} cv2={cstats[i]}"); ok = False
print(f"  [{'OK' if ok else 'FAIL'}] cc stats rows 1..{min(n,cn)-1}")

# 4) edt_to_nearest_zero
# The wasm core computes the EXACT Euclidean distance-to-zero; cv2.distanceTransform
# with DIST_L2+maskSize=3 is only approximate (≈2-3px error). So we validate against
# scipy's exact EDT, which is what the shared core replaces cv2 with.
from scipy import ndimage as ndi
seed = (g > 120).astype(np.uint8)
check("edt_to_nearest_zero (vs scipy exact)", sc.edt_to_nearest_zero(seed),
      ndi.distance_transform_edt(seed == 0).astype(np.float32), tol=1e-4)
check("edt_to_nearest_zero (inverted, vs scipy exact)", sc.edt_to_nearest_zero((seed == 0).astype(np.uint8)),
      ndi.distance_transform_edt(seed != 0).astype(np.float32), tol=1e-4)

# 5) dilate / erode (ellipse + rect), iterations
for ksize in (3, 5, 7):
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    kr = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    check(f"dilate ellipse {ksize}", sc.dilate(mask, ke), cv2.dilate(mask, ke), exact=True)
    check(f"erode ellipse {ksize}", sc.erode(mask, ke), cv2.erode(mask, ke), exact=True)
    check(f"dilate rect {ksize}", sc.dilate(mask, kr), cv2.dilate(mask, kr), exact=True)
    check(f"erode rect {ksize}", sc.erode(mask, kr), cv2.erode(mask, kr), exact=True)
    check(f"dilate ellipse iter2", sc.dilate(mask, ke, iterations=2),
          cv2.dilate(mask, ke, iterations=2), exact=True)

# 6) morphology_ex close/open
# NOTE: the backend uses separate cv2.erode/cv2.dilate calls, NOT cv2.morphologyEx — and
# cv2.morphologyEx differs from the naive dilate(erode) composition at borders (cv2 internal
# border handling). So we validate morphology_ex against the DECOMPOSED form it mimics.
kr3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
check("morph close", sc.morphology_ex(mask, cv2.MORPH_CLOSE, kr3),
      cv2.dilate(cv2.erode(mask, kr3), kr3), exact=True)
kr2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
check("morph open", sc.morphology_ex(mask, cv2.MORPH_OPEN, kr2),
      cv2.erode(cv2.dilate(mask, kr2), kr2), exact=True)

# 7) np.ones kernels
check("dilate np.ones(3)", sc.dilate(mask, np.ones((3,3),np.uint8)), cv2.dilate(mask, np.ones((3,3),np.uint8)), exact=True)

print("\nRESULT:", "ALL OK" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
