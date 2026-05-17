import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class FPSCounter:
    smoothing: float = 0.9
    _last_t: Optional[float] = None
    _fps: float = 0.0

    def tick(self) -> float:
        t = time.time()
        if self._last_t is None:
            self._last_t = t
            return 0.0
        dt = max(t - self._last_t, 1e-6)
        inst_fps = 1.0 / dt
        self._fps = self.smoothing * self._fps + (1.0 - self.smoothing) * inst_fps
        self._last_t = t
        return self._fps


def draw_text(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.6,
    thickness: int = 2,
) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def alpha_blend(base_bgr: np.ndarray, overlay_bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Alpha blend: base*(1-a) + overlay*a.

    alpha can be float [0..1] array (H,W) or uint8 [0..255].
    """
    if alpha.dtype == np.uint8:
        a = alpha.astype(np.float32) / 255.0
    else:
        a = alpha.astype(np.float32)
        if a.max() > 1.0:
            a = a / 255.0

    a = a[..., None]  # H,W,1
    out = base_bgr.astype(np.float32) * (1.0 - a) + overlay_bgr.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def blur_region(img: np.ndarray, mask: np.ndarray, k: int = 25) -> np.ndarray:
    """Selective blur: blur entire img then composite using mask."""
    blurred = cv2.GaussianBlur(img, (k | 1, k | 1), 0)
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)
    mask01 = mask.astype(np.float32) / 255.0
    a = mask01[..., None]
    out = img.astype(np.float32) * (1.0 - a) + blurred.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def circle_falloff(radius: float, dist: float) -> float:
    if dist >= radius:
        return 0.0
    t = dist / radius
    return (1.0 - t) ** 2

