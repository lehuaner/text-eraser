"""Drop-in `cv2` shim that routes the shared-core operators through textcore.wasm while
falling through to the real cv2 for everything else.

Usage in a backend algorithm module::

    from text_eraser import _cv as cv2

After that, ``cv2.dilate(...)``, ``cv2.connectedComponents(...)``, etc. transparently
use the SAME wasm operators the browser Worker runs (``textcore.wasm``) — that is the
"前后端共用一套算法" guarantee. Operators that are not part of the shared core
(``cvtColor`` with non-RGB2GRAY codes, ``resize``, ``GaussianBlur``, ``boundingRect``,
``Sobel``, ``boxFilter``, ``distanceTransform``, ...) are served by the real cv2 via the
module-level ``__getattr__`` below.

Every routed op keeps cv2's value-range and border semantics and falls back to cv2 on any
error, so importing this module never changes observable behavior unless the wasm core is
available — and when it is, the routed ops are byte-identical to cv2 (validated by
``test_shared_core_smoke.py``). This makes the backend and the browser compute the exact
same morphological / connected-component / grayscale operators.
"""
import cv2

from ._shared_core import (
    dilate as _dilate,
    erode as _erode,
    morphology_ex as _morphology_ex,
    connected_components as _connected_components,
    connected_components_with_stats as _connected_components_with_stats,
    rgb2gray as _rgb2gray,
)


def dilate(mask, kernel, iterations=1):
    """cv2.dilate-compatible; delegates to the shared wasm core."""
    return _dilate(mask, kernel, iterations)


def erode(mask, kernel, iterations=1):
    """cv2.erode-compatible; delegates to the shared wasm core."""
    return _erode(mask, kernel, iterations)


def morphologyEx(mask, op, kernel):
    """cv2.morphologyEx-compatible.

    Uses the *separated* dilate/erode composition (``dilate(erode)`` for CLOSE,
    ``erode(dilate)`` for OPEN) — the same composition the browser's cv-bridge uses —
    so the backend matches the browser for close/open instead of cv2's internal
    ``morphologyEx`` (which differs from the separated form by up to 1 at borders).
    """
    return _morphology_ex(mask, op, kernel)


def connectedComponents(mask, connectivity=8):
    """cv2.connectedComponents-compatible; returns ``(n, labels)``."""
    return _connected_components(mask, connectivity)


def connectedComponentsWithStats(mask, connectivity=8):
    """cv2.connectedComponentsWithStats-compatible; returns
    ``(n, labels, stats[n,5], centroids[n,2])`` with cv2 column ordering."""
    return _connected_components_with_stats(mask, connectivity)


def cvtColor(src, code, dst=None, dstCn=None):
    """cv2.cvtColor-compatible. ``COLOR_RGB2GRAY`` is delegated to the shared wasm core
    (byte-identical to cv2's fixed-point formula); every other code falls through to cv2."""
    if code == cv2.COLOR_RGB2GRAY:
        return _rgb2gray(src)
    if dst is not None:
        return cv2.cvtColor(src, code, dst, dstCn)
    return cv2.cvtColor(src, code)


def __getattr__(name):
    # Any name not defined above (Sobel, resize, GaussianBlur, boundingRect,
    # distanceTransform, COLOR_*, MORPH_*, DIST_*, INTER_*, CV_32F, Mat, ...) is
    # served by the real cv2 module.
    return getattr(cv2, name)
