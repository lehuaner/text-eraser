"""Compute IoU between Python refmask.raw and JS mask_js.raw for each _cmp_work dir."""
import os

WORK = 'D:/Code/Project/Python/TextPatch/scripts/_cmp_work'

def load(name, fn):
    p = os.path.join(WORK, name, fn)
    if not os.path.exists(p):
        return None
    with open(p, 'rb') as f:
        return f.read()

for name in sorted(os.listdir(WORK)):
    d = os.path.join(WORK, name)
    dims = load(name, 'dims.txt')
    if dims is None:
        continue
    H, W = map(int, dims.decode().strip().split())
    ref = load(name, 'refmask.raw')
    js = load(name, 'mask_js.raw')
    if ref is None or js is None:
        print(f"{name}: MISSING (ref={'Y' if ref else 'N'} js={'Y' if js else 'N'})")
        continue
    ref = ref[:H*W]
    js = js[:H*W]
    inter = sum(1 for a, b in zip(ref, js) if a and b)
    union = sum(1 for a, b in zip(ref, js) if a or b)
    refn = sum(1 for a in ref if a)
    jsn = sum(1 for b in js if b)
    iou = inter / union if union else 1.0
    # Dice and per-set recall/precision
    prec = inter / jsn if jsn else 1.0
    rec = inter / refn if refn else 1.0
    print(f"{name}: H={H} W={W} ref={refn} js={jsn} inter={inter} union={union} "
          f"iou={iou:.4f} prec={prec:.4f} rec={rec:.4f}")
