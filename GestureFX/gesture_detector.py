from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class GestureResult:
    gesture_name: str
    finger_bits: List[int]
    gesture_id: str


class GestureDetector:
    """Map 5-bit finger vector -> gesture id/name."""

    def __init__(self):
        self.map: Dict[Tuple[int, int, int, int, int], Tuple[str, str]] = {
            (0, 1, 0, 0, 0): ("One Finger Up", "fire"),
            (0, 1, 1, 0, 0): ("Two Fingers Up", "spark"),
            (1, 1, 1, 1, 1): ("Open Palm", "rgb_palm"),
            (0, 0, 0, 0, 0): ("Closed Fist", "freeze"),
            (1, 0, 0, 0, 0): ("Thumbs Up", "lightning"),
            (0, 1, 1, 0, 0): ("Victory Sign", "neon_particles"),  # same bits per spec; will resolve below
            (1, 1, 0, 0, 0): ("L-Sign", "sepia_filter"),
            (0, 1, 0, 0, 1): ("Rock On", "portal"),
            (0, 1, 1, 1, 1): ("Four Fingers", "magic_circle"),
            (0, 0, 0, 0, 1): ("Pinky Up", "smoke"),
        }

        # Resolve spec ambiguity: Victory and Two Fingers both [0,1,1,0,0].
        # We'll disambiguate by hand orientation in effect_manager using palm bbox width/height.
        self.two_fingers_key = (0, 1, 1, 0, 0)

    def detect(self, finger_bits: List[int]) -> GestureResult:
        key = tuple(finger_bits)

        if key == self.two_fingers_key:
            # default to Two Fingers Up (spark); effect_manager may override to neon.
            name, gid = "Two Fingers Up", "spark"
            return GestureResult(gesture_name=name, finger_bits=finger_bits, gesture_id=gid)

        if key in self.map:
            name, gid = self.map[key]
            return GestureResult(gesture_name=name, finger_bits=finger_bits, gesture_id=gid)

        return GestureResult(gesture_name="Unknown", finger_bits=finger_bits, gesture_id="none")
