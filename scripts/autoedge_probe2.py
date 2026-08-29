"""自动 edge 探针 v2：源图驱动的「文字色残留」逐环测量。

思路：edge 膨胀覆盖的是蒙版外围的抗锯齿/文字色边缘。若离蒙版 r 环处仍有明显
'文字色'像素(介于文字色与背景色之间的混色边)，则 edge<r 时该环不被填充 → 鬼影。
因此：needed_edge = 含显著文字色残留的最大环半径(>=1)。

本探针打印每张图 1..4 环的：
  - grad : 原图 Sobel 梯度均值(文字边处高)
  - blend: 介于 文字色↔背景色 之间(混色边)的像素数
  - tcol%: 该环像素中 '明显偏文字色' 的比例
用于标定 needed_edge 的阈值。
"""
import sys, json, time
import numpy as np
import cv2

sys.path.insert(0, ".")
from text_eraser.eraser import _ellipse
from text_eraser.text_select import detect_text_mask

DEFAULTS = dict(
    q_off=55.0, max_area_ratio=0.40, max_box_ratio=0.40,
    fill_white=True, fill_max_dist=12, tint_fill=True,
)

M = 1.0 / 255.0


def load_params(meta_path):
    p = dict(DEFAULTS)
    try:
        m = json.load(open(meta_path, encoding="utf-8"))
        for k in p:
            if k in m.get("params", {}):
                p[k] = m["params"][k]
    except Exception:
        pass
    return p


def ring_stats(rgb_lab, text_lab, bg_lab, mask):
    H, W = mask.shape
    out = []
    prev = mask > 0
    for r in range(1, 5):
        cur = cv2.dilate(mask, _ellipse(r)) > 0
        ring = cur & ~prev
        prev = cur
        n = int(ring.sum())
        if n == 0:
            out.append((0, 0, 0.0)); continue
        lab = rgb_lab[ring]
        # 到文字色、背景色的 LAB 距离
        d_text = np.linalg.norm(lab - text_lab, axis=2 if False else 1)  # (n,3)->(n,)
        d_text = np.sqrt(((lab - text_lab) ** 2).sum(1))
        d_bg = np.sqrt(((lab - bg_lab) ** 2).sum(1))
        # blend: 离文字色 <= 离背景色 且 离背景色 > 阈值(不是纯背景)
        blend = int(((d_text <= d_bg) & (d_bg > 18)).sum())
        # tcol: 明显偏文字色(离文字色近)
        tcol = int((d_text < 28).sum())
        out.append((n, blend, 100.0 * tcol / n))
    return out


def main():
    ids = ["1787765979188", "1787765716464", "1787766251689"]
    for tid in ids:
        bgr = cv2.imread(f"data/_hist3_now/{tid}.png")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        params = load_params(f"data/history/{tid}/meta.json")
        core, _ = detect_text_mask(rgb, method="ml", **params)
        if not core.any():
            print(tid, "no text"); continue
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        text_lab = lab[core > 0].mean(0)
        far = cv2.dilate(core, _ellipse(16)) == 0
        bg_lab = lab[far].mean(0)
        # 原图 Sobel 梯度
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx ** 2 + gy ** 2)
        stats = ring_stats(lab, text_lab, bg_lab, core)
        print(f"\n=== {tid}  core_pix={int((core>0).sum())}  text_LAB={text_lab.round(1)} bg_LAB={bg_lab.round(1)} ===")
        for r, (n, blend, tcol) in enumerate(stats, start=1):
            g = grad[(cv2.dilate(core, _ellipse(r)) > 0) & (cv2.dilate(core, _ellipse(r - 1)) == 0)] if r > 0 else np.array([0])
            gmean = float(g.mean()) if g.size else 0.0
            print(f"  ring{r}: n={n:>6}  blend(混色边)={blend:>6}  tcol%={tcol:5.1f}  grad_mean={gmean:6.2f}")


if __name__ == "__main__":
    main()
