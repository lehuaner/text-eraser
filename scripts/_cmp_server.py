"""Temp comparison server: receives backend+browser mask base64, computes IoU + saves a
side-by-side composite (orig with backend mask on left, browser mask on right)."""
import base64
import json
import numpy as np
import cv2
from http.server import BaseHTTPRequestHandler, HTTPServer

ORIG = r"D:\Code\Project\Python\TextPatch\data\_glowcheck\556_orig.png"
OUT = r"D:\Code\Project\Python\TextPatch\data\_glowcheck\_cmp_composite.png"


def b64_to_mask(b64):
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    png = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if png is None:
        return None
    if png.shape[2] == 4:
        a = png[:, :, 3]
    else:
        a = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
    return (a > 10).astype(np.uint8)


def red_over(orig, mask):
    out = orig.copy()
    out[mask > 0] = (40, 40, 255)  # BGR red
    return out


class H(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        orig = cv2.imread(ORIG)
        bm = b64_to_mask(body["backend"])
        wm = b64_to_mask(body["browser"])
        H, W = orig.shape[:2]
        bm = cv2.resize(bm, (W, H))
        wm = cv2.resize(wm, (W, H))
        left = red_over(orig, bm)
        right = red_over(orig, wm)
        composite = np.hstack([left, right])
        cv2.imwrite(OUT, composite)
        inter = int(((bm > 0) & (wm > 0)).sum())
        union = int(((bm > 0) | (wm > 0)).sum())
        iou = inter / union if union else 0.0
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "iou": round(iou, 4),
            "backend_px": int((bm > 0).sum()),
            "browser_px": int((wm > 0).sum()),
            "intersection": inter, "union": union,
        }).encode())

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 8799), H).serve_forever()
