"""Generate test masks for the shared-core proof.

Writes binary files with header (int32 h, int32 w) + H*W u8 mask bytes.
- mask_synth.bin : deterministic synthetic strokes (portable, no deps)
- mask_real.bin  : thresholded from text_eraser/assets/example.png (real-ish)
"""
import struct
import os
import numpy as np

try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

testdir = os.path.dirname(os.path.abspath(__file__))
repo = os.path.dirname(testdir)  # shared/


def save(path, m):
    h, w = m.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", h, w))
        f.write(m.astype(np.uint8).tobytes())
    print(f"  wrote {os.path.basename(path)} {h}x{w} ones={int(m.sum())}")


# --- synthetic ---
h, w = 120, 200
m = np.zeros((h, w), dtype=np.uint8)
if HAVE_CV2:
    cv2.rectangle(m, (20, 30), (60, 70), 1, -1)
    cv2.rectangle(m, (90, 40), (130, 90), 1, -1)
    cv2.line(m, (10, 100), (180, 100), 1, 3)
    cv2.circle(m, (150, 30), 15, 1, -1)
else:
    m[30:70, 20:60] = 1
    m[40:90, 90:130] = 1
    m[99:103, :] = 1
save(os.path.join(testdir, "mask_synth.bin"), m)

# --- real-ish ---
ex_path = os.path.join(repo, "..", "text_eraser", "assets", "example.png")
if HAVE_CV2 and os.path.exists(ex_path):
    img = cv2.imread(ex_path)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, 400.0 / max(g.shape))
    if scale < 1:
        g = cv2.resize(g, (int(g.shape[1] * scale), int(g.shape[0] * scale)))
    _, mb = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    save(os.path.join(testdir, "mask_real.bin"), mb)
else:
    print("  skip real mask (cv2/example.png unavailable)")
