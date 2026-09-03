"""诊断 DBNet 在小图上的输出"""
import sys
from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "core/models/det/ch_PP-OCRv4_det.onnx"

img = np.asarray(Image.open(ROOT / "data/needExtractAndPatch.png").convert("RGB"), dtype=np.uint8)
H, W = img.shape[:2]
print(f"input {W}x{H}")

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# DBNet 默认 max_side=960 → 缩小到 960 长边
max_side = 960
scale = min(max_side / max(H, W), 1.0)
nw, nh = int(round(W * scale)), int(round(H * scale))
nw -= nw % 32; nh -= nh % 32
nw = max(32, nw); nh = max(32, nh)
img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

# BGR, normalize
x = img_resized.astype(np.float32) / 255.0
x = (x - mean) / std
x = x[:, :, ::-1].copy()  # RGB->BGR
x = x.transpose(2, 0, 1)[None, ...].astype(np.float32)

sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0]  # (nh, nw)
print(f"output {out.shape}, min={out.min():.3f} max={out.max():.3f} mean={out.mean():.3f}")
print(f"  >0.15 比例 {float((out > 0.15).mean()):.4f}")
print(f"  >0.30 比例 {float((out > 0.30).mean()):.4f}")
print(f"  >0.50 比例 {float((out > 0.50).mean()):.4f}")

# 升采样回原图尺度, 截断到 [0,1]
prob_up = cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR)
prob_up = np.clip(prob_up, 0, 1)
Image.fromarray((prob_up * 255).astype(np.uint8)).save(ROOT / "data/dryrun_out/_dbnet_prob.png")

# 阈值二值化看看
for t in (0.10, 0.15, 0.20, 0.30, 0.50):
    m = (prob_up > t).astype(np.uint8) * 255
    Image.fromarray(m).save(ROOT / f"data/dryrun_out/_dbnet_thr{int(t*100):03d}.png")
    n_white = int((m > 0).sum())
    print(f"  thr={t:.2f}  white={n_white}px  ratio={n_white/(H*W):.4f}")
