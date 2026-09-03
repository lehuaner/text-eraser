"""Auto-bootstrap the heavy browser-side assets for the "本地浏览器计算" feature.

The pure-browser worker (``eraser.worker.js``) loads several large artifacts that
are intentionally NOT committed to the repo (see ``.gitignore``: ``browser/vendor/``,
``browser/dist/``):

* ``opencv.js``                 — OpenCV wasm UMD build (3rd-party)
* ``onnxruntime-web``           — ``ort.min.js`` + threaded wasm (3rd-party)
* ``ch_PP-OCRv4_det.onnx``      — DBNet text-detection model (shipped in the repo
                                  under ``text_eraser/models/det/``; auto-downloaded
                                  from HuggingFace by ``ml_text_select`` if absent)
* ``te-bundle.js``              — our ESM pipeline bundled to an IIFE (built locally
                                  with esbuild from ``browser/src``)

Previously these had to be fetched/placed by hand after a checkout or ``git clean``,
which left the worker 404-ing and the UI stuck on a 15s "浏览器引擎初始化超时".
This module makes the feature self-bootstrapping: on server start we download/copy
any missing asset so a fresh checkout (or a third party installing the package) just
works — no manual download step.

All downloads are best-effort and time-boxed; failures are logged, not fatal (the
"本地浏览器计算" toggle then surfaces a clear error instead of hanging).
"""
from __future__ import annotations

import logging
import os
import shutil
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger("text_eraser.browser_assets")

_PACKAGE_DIR = Path(__file__).resolve().parent          # .../text_eraser
_REPO_ROOT = _PACKAGE_DIR.parent                         # .../TextPatch
_BROWSER_DIR = _REPO_ROOT / "browser"
_VENDOR_DIR = _BROWSER_DIR / "vendor"
_ORT_DIR = _VENDOR_DIR / "ort"
_DIST_DIR = _BROWSER_DIR / "dist"

# Per-asset download manifest. Each entry: (dest_under_repo, [candidate sources],
# min_bytes). A source may be an ``https://``/``http://`` URL or a ``file://`` path
# (used for local fallbacks such as an already-``npm install``-ed node_modules).
# Candidates are tried in order; first success wins.
_ASSETS = [
    (
        "browser/vendor/opencv.js",
        [
            "https://docs.opencv.org/4.9.0/opencv.js",
            "https://docs.opencv.org/4.7.0/opencv.js",
        ],
        1_000_000,
    ),
    (
        "browser/vendor/ort/ort.min.js",
        [
            "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.min.js",
            "file://" + str(_BROWSER_DIR / "node_modules" / "onnxruntime-web" / "dist" / "ort.min.js"),
        ],
        100_000,
    ),
    (
        "browser/vendor/ort/ort-wasm-simd-threaded.wasm",
        [
            "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort-wasm-simd-threaded.wasm",
            "file://"
            + str(_BROWSER_DIR / "node_modules" / "onnxruntime-web" / "dist" / "ort-wasm-simd-threaded.wasm"),
        ],
        1_000_000,
    ),
]

# Download timeout per single source attempt (seconds). Short enough that an offline
# box doesn't stall server boot for minutes; the startup hook runs best-effort anyway.
_PER_SOURCE_TIMEOUT = 60


def _resolve_node() -> str | None:
    """Locate a ``node`` binary. Honour TEXTERASER_NODE_BIN, else PATH."""
    env = os.environ.get("TEXTERASER_NODE_BIN")
    if env and shutil.which(env):
        return env
    return shutil.which("node")


def _download_one(url: str, dest: Path, timeout: int) -> None:
    """Download ``url`` to ``dest`` (via ``.part`` then atomic rename). Supports
    http(s) and ``file://`` (local copy). Raises on failure."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(dest) + ".part"
    try:
        if url.startswith("file://"):
            src = url[len("file://"):]
            if not os.path.isfile(src):
                raise FileNotFoundError(src)
            with open(src, "rb") as f, open(tmp, "wb") as out:
                while True:
                    buf = f.read(1 << 20)
                    if not buf:
                        break
                    out.write(buf)
        else:
            req = urllib.request.Request(url, headers={"User-Agent": "text-eraser/browser-assets"})
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r, open(tmp, "wb") as out:
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    out.write(buf)
        sz = os.path.getsize(tmp)
        if sz < 1_000:
            raise RuntimeError(f"downloaded file too small ({sz} bytes)")
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _ensure_network_asset(rel_dest: str, sources: list[str], min_bytes: int) -> str:
    """Ensure ``<repo>/rel_dest`` exists and is large enough. Returns a status
    string: 'present' / 'downloaded' / 'error: ...'."""
    dest = _REPO_ROOT / rel_dest
    if dest.is_file() and dest.stat().st_size >= min_bytes:
        return "present"
    last_err: Exception | None = None
    for url in sources:
        try:
            logger.info("[browser-assets] downloading %s <- %s", rel_dest, url)
            _download_one(url, dest, _PER_SOURCE_TIMEOUT)
            logger.info("[browser-assets] ok: %s (%d bytes)", rel_dest, dest.stat().st_size)
            return "downloaded"
        except Exception as e:  # try next candidate
            last_err = e
            logger.warning("[browser-assets] source failed (%s): %s", url, e)
    return f"error: {'; '.join(str(x) for x in (last_err,))}"


def _ensure_model() -> str:
    """Copy the DBNet model into browser/vendor. Reuses ml_text_select, which
    auto-downloads it from HuggingFace/hf-mirror on first need."""
    dest = _VENDOR_DIR / "ch_PP-OCRv4_det.onnx"
    try:
        from text_eraser import ml_text_select

        src = ml_text_select.get_model_path()  # ensures download if missing
        if not dest.is_file() or dest.stat().st_size != os.path.getsize(src):
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            logger.info("[browser-assets] model copied %s -> %s", src, dest)
        return "present" if dest.stat().st_size >= 1_000_000 else "error: model too small"
    except Exception as e:
        return f"error: {e}"


def _ensure_bundle() -> str:
    """Build browser/dist/te-bundle.js via `node browser/build.mjs` (esbuild)."""
    dest = _DIST_DIR / "te-bundle.js"
    if dest.is_file() and dest.stat().st_size >= 10_000:
        return "present"
    node = _resolve_node()
    if not node:
        return ("error: node not found — run `npm install` in browser/ then "
                "`node browser/build.mjs` to build te-bundle.js")
    # esbuild must be resolvable from browser/ (npm install esbuild there).
    try:
        subprocess.run(
            [node, "browser/build.mjs"],
            cwd=str(_REPO_ROOT),
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if dest.is_file() and dest.stat().st_size >= 10_000:
            logger.info("[browser-assets] bundle built: %s", dest)
            return "downloaded"
        return "error: build produced no bundle"
    except subprocess.TimeoutExpired:
        return "error: esbuild build timed out"
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or str(e))[:300]
        return f"error: esbuild build failed: {err}"


def ensure_browser_assets(force: bool = False) -> dict[str, str]:
    """Ensure all browser-compute assets exist. Returns {asset: status}.

    Best-effort: never raises. Network failures are reported per-asset so the
    caller can surface a precise message instead of letting the worker hang.
    """
    if force:
        logger.info("[browser-assets] force re-fetch requested")
    results: dict[str, str] = {}

    for rel_dest, sources, min_bytes in _ASSETS:
        results[rel_dest] = _ensure_network_asset(rel_dest, sources, min_bytes)

    results["browser/vendor/ch_PP-OCRv4_det.onnx"] = _ensure_model()
    results["browser/dist/te-bundle.js"] = _ensure_bundle()

    missing = [k for k, v in results.items() if v.startswith("error")]
    if missing:
        logger.warning("[browser-assets] %d asset(s) missing/unavailable: %s", len(missing), missing)
    else:
        logger.info("[browser-assets] all assets ready")
    return results


def browser_assets_ready() -> bool:
    """Quick check: are all required assets present on disk right now?"""
    checks = [
        (_VENDOR_DIR / "opencv.js", 1_000_000),
        (_ORT_DIR / "ort.min.js", 100_000),
        (_ORT_DIR / "ort-wasm-simd-threaded.wasm", 1_000_000),
        (_VENDOR_DIR / "ch_PP-OCRv4_det.onnx", 1_000_000),
        (_DIST_DIR / "te-bundle.js", 10_000),
    ]
    return all(p.is_file() and p.stat().st_size >= mb for p, mb in checks)


def browser_assets_status() -> dict:
    """Detailed per-asset status for /api/browser-assets (UI/health)."""
    items = [
        ("opencv.js", _VENDOR_DIR / "opencv.js", 1_000_000),
        ("ort.min.js", _ORT_DIR / "ort.min.js", 100_000),
        ("ort-wasm-simd-threaded.wasm", _ORT_DIR / "ort-wasm-simd-threaded.wasm", 1_000_000),
        ("ch_PP-OCRv4_det.onnx", _VENDOR_DIR / "ch_PP-OCRv4_det.onnx", 1_000_000),
        ("te-bundle.js", _DIST_DIR / "te-bundle.js", 10_000),
    ]
    out = {}
    for name, p, mb in items:
        out[name] = {
            "present": p.is_file() and p.stat().st_size >= mb,
            "path": str(p),
            "bytes": p.stat().st_size if p.is_file() else 0,
        }
    out["ready"] = all(v["present"] for v in out.values())
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = ensure_browser_assets()
    print(res)
    print("ready:", browser_assets_ready())
