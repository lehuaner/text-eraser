"""v4 管线冒烟测试：在历史图/发光样例上跑 deglow.pipeline.run，输出结果与报告。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

targets = [
    ROOT / "data" / "history" / "1787768178725" / "orig.bin",
    ROOT / "data" / "history" / "1787769941738" / "orig.bin",
    ROOT / "data" / "_batch_orig.png",
    ROOT / "data" / "_batch_new.png",
    ROOT / "data" / "final" / "needExtractAndPatch_orig.png",
]

for tp in targets:
    if not tp.is_file():
        continue
    raw = tp.read_bytes()
    if tp.suffix == ".bin":
        # .bin 是原图字节（PNG/JPG），用 PIL 解出
        from PIL import Image
        import io
        rgb = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), np.uint8)
    else:
        img = cv2.imread(str(tp), cv2.IMREAD_COLOR)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 缩到测试尺度（≤960 最长边）
    h, w = rgb.shape[:2]
    scale = min(1.0, 960 / max(h, w))
    if scale < 1.0:
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))
    print("=" * 60)
    print("target:", tp.name, "size:", rgb.shape[1], "x", rgb.shape[0])

    from deglow import pipeline

    res = pipeline.run(rgb, deglow_strength=1.0)
    rep = res.report
    print("has_glow:", rep["has_glow"], "| glow_pix:", rep.get("glow_pix"),
          "| tier:", rep.get("tier_pix"))
    print("sigma_bar:", rep.get("sigma_bar"), "| l_tex:", rep.get("l_tex"))
    for d in rep.get("domains", []):
        print("  dom", d["id"], d["mode"], "calib=", d["calibrated"],
              "a_max=", d["alpha_max"], "sg=", d["sigma_g"],
              "dye=", d["dye"], "pix=", d["pix"],
              "tier=", d["tier_pix"], "pass=", d["verify_pass_rate"])
    out = res.image
    diff = np.abs(out.astype(np.int16) - rgb.astype(np.int16)).mean()
    print("mean |Δ| vs orig:", round(float(diff), 2))
    outp = ROOT / "data" / f"_v4_smoke_{tp.stem}.png"
    cv2.imwrite(str(outp), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    provp = ROOT / "data" / f"_v4_smoke_{tp.stem}_prov.png"
    prov_img = np.repeat(res.prov[..., None] * (255 // 5), 3, axis=-1).astype(np.uint8)
    cv2.imwrite(str(provp), prov_img)
    print("saved:", outp.name)
print("DONE")