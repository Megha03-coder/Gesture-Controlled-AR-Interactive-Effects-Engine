from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from utils import alpha_blend, clamp


class OverlayManager:
    def __init__(self):
        pass

    def rainbow_overlay(self, w: int, h: int, t: float) -> np.ndarray:
        """Create a dynamic rainbow RG overlay (BGR)."""
        # create horizontal rainbow bands with time shift
        x = np.linspace(0, 1, w, dtype=np.float32)
        phase = (x + (t * 0.2)) % 1.0
        # Map phase to colors using sine waves
        r = (np.sin(2 * np.pi * (phase + 0.0)) * 127 + 128).astype(np.uint8)
        g = (np.sin(2 * np.pi * (phase + 0.33)) * 127 + 128).astype(np.uint8)
        b = (np.sin(2 * np.pi * (phase + 0.66)) * 127 + 128).astype(np.uint8)
        # tile to height
        R = np.tile(r[None, :], (h, 1))
        G = np.tile(g[None, :], (h, 1))
        B = np.tile(b[None, :], (h, 1))
        return cv2.merge([B, G, R])

    def apply_rgb_filter(self, frame: np.ndarray, strength: float, t: float) -> np.ndarray:
        """Dynamic channel shifting (RGB) + slight color cycling."""
        h, w = frame.shape[:2]
        # shift channels
        shift = int(10 * strength * np.sin(t * 1.5))
        Mx = np.float32([[1, 0, shift], [0, 1, 0]])
        My = np.float32([[1, 0, 0], [0, 1, shift]])

        b, g, r = cv2.split(frame)
        r2 = cv2.warpAffine(r, Mx, (w, h), borderMode=cv2.BORDER_REFLECT)
        g2 = cv2.warpAffine(g, My, (w, h), borderMode=cv2.BORDER_REFLECT)
        b2 = b

        out = cv2.merge([b2, g2, r2])
        # blend with original
        a = clamp(strength, 0.0, 1.0)
        return cv2.addWeighted(frame, 1.0 - a, out, a, 0)

    def rainbow_alpha(self, frame_shape: Tuple[int, int, int], t: float) -> np.ndarray:
        h, w = frame_shape[:2]
        x = np.linspace(0, 2 * np.pi, w, dtype=np.float32)
        y = np.linspace(0, 2 * np.pi, h, dtype=np.float32)
        X, Y = np.meshgrid(x, y)
        a = (np.sin(X * 1.2 + Y * 0.4 + t * 2.0) * 0.5 + 0.5)
        # alpha in [0..1]
        return a.astype(np.float32)

    def apply_rainbow_overlay(self, frame: np.ndarray, t: float, strength: float = 0.35) -> np.ndarray:
        h, w = frame.shape[:2]
        rainbow = self.rainbow_overlay(w, h, t)
        alpha = self.rainbow_alpha(frame.shape, t)
        alpha = np.clip(alpha * strength, 0.0, 1.0)
        return alpha_blend(frame, rainbow, alpha)

    def apply_sepia_filter(self, frame: np.ndarray) -> np.ndarray:
        """Instagram-style Vintage Sepia filter."""
        # Matrix weights for Sepia in OpenCV's BGR format
        kernel = np.array([
            [0.131, 0.534, 0.272],  # Blue channel
            [0.168, 0.686, 0.349],  # Green channel
            [0.189, 0.769, 0.393]   # Red channel
        ], dtype=np.float32)
        
        # Apply the color transformation
        sepia = cv2.transform(frame, kernel)
        
        # Clip values to valid 0-255 range
        return np.clip(sepia, 0, 255).astype(np.uint8)
