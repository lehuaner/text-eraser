"""API 端到端测试：同一张绿光图片 × 三种去发光方案。"""
import io
import sys
import time
import urllib.request
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
UP = "http://127.0.0.1:8766/api/erase"
raw = (ROOT / "data" / "history" / "1787768178725" / "orig.bin").read_bytes()

boundary = "----WebKitFormBoundary" + "x" * 16
for scheme in ("channel", "v4", "off"):
    fields = [("mask_pad", "2"), ("q_off", "55"), ("max_area_ratio", "0.40"),
              ("max_box_ratio", "0.40"), ("edge_aware", "false"),
              ("edge_extend", "1"), ("glow_mode", "auto"),
              ("deglow_strength", "1.0"), ("deglow_scheme", scheme),
              ("return_overlay", "true")]
    body = b""
    for k, v in fields:
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n"
                 f"{v}\r\n").encode()
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
             f"filename=\"glow.png\"\r\nContent-Type: image/png\r\n\r\n").encode()
    body += raw + b"\r\n" + f"--{boundary}--\r\n".encode()

    req = urllib.request.Request(UP, data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        j = json.loads(r.read().decode("utf-8"))
    d = j.get("data", {})
    print("=" * 50)
    print("scheme:", scheme, "| ok:", j.get("ok"), "| 客户端耗时 %.1fs" % (time.time() - t0))
    print("  elapsed:", d.get("elapsed"), "| mask_pix:", d.get("mask_pix"),
          "| boxes:", len(d.get("boxes", [])))
    if d.get("deglow_b64"):
        b64 = d["deglow_b64"]
        img = np.asarray(Image.open(io.BytesIO(__import__("base64").b64decode(b64))).convert("RGB"), np.uint8)
        g = img[..., 1].astype(np.int16)
        dom = (g - np.maximum(img[..., 0].astype(np.int16), img[..., 2].astype(np.int16))) > 6
        print(f"  deglow_b64 存在 ({len(b64)/1024:.0f}KB) 全图残留绿占比 {dom.mean():.2%}")
    rep = d.get("dglow_report")
    if rep:
        print("  dglow_report: has_glow=", rep.get("has_glow"),
              "| glow_pix=", rep.get("glow_pix"),
              "| tiers=", rep.get("tier_pix"),
              "| ε代理=", rep.get("epsilon_proxy"))
        for dm in rep.get("domains", [])[:2]:
            print("    dom", dm.get("id"), dm.get("mode"), "pix=", dm.get("pix"),
                  "dye=", dm.get("dye"), "σg=", dm.get("sigma_g"),
                  "pass=", dm.get("verify_pass_rate"))
print("ALL DONE")