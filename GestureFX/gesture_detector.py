from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class GestureResult:
    gesture_name: str
    finger_bits: List[int]
    gesture_id: str


class GestureDetector:
    """Map 5-bit finger vector -> gesture id/name."""

    def __init__(self):
        self.map: Dict[Tuple[int, int, int, int, int], Tuple[str, str]] = {
            (0, 1, 0, 0, 0): ("One Finger Up", "black_white"),
            (1, 1, 0, 0, 0): ("One Finger Up", "black_white"),  # Forgiving thumb
            (0, 1, 1, 0, 0): ("Two Fingers Up", "neon_glow"),
            (1, 1, 1, 0, 0): ("Two Fingers Up", "neon_glow"),   # Forgiving thumb
            (0, 1, 1, 1, 0): ("Three Fingers Up", "cartoon"),
            (1, 1, 1, 1, 0): ("Three Fingers Up", "cartoon"),   # Forgiving thumb
            (1, 1, 1, 1, 1): ("Open Palm", "reset"),
            (0, 1, 1, 1, 1): ("Open Palm", "reset"),            # Forgiving thumb
            (0, 0, 0, 0, 0): ("Closed Fist", "toggle_pause"),
            (1, 0, 0, 0, 0): ("Closed Fist", "toggle_pause"),   # Forgiving thumb
        }

    def detect(self, finger_bits: List[int]) -> GestureResult:
        key = tuple(finger_bits)

        if key in self.map:
            name, gid = self.map[key]
            return GestureResult(gesture_name=name, finger_bits=finger_bits, gesture_id=gid)

        return GestureResult(gesture_name="Unknown", finger_bits=finger_bits, gesture_id="none")
