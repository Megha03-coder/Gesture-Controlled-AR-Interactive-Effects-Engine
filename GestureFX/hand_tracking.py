from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class HandState:
    handedness: str  # 'Left'/'Right'
    lm_list: List[Tuple[int, int]]  # 21 points in image pixels
    lm_world_list: Optional[List] = None
    finger_bits: List[int] = None  # [thumb,index,middle,ring,pinky]
    bbox: Optional[Tuple[int, int, int, int]] = None  # x,y,w,h


class HandTracker:
    """MediaPipe Hands wrapper.

    Note: some MediaPipe wheels expose `mp.solutions.*`, others expose Tasks API.
    This code supports both.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
    ):
        self._hands_api = "solutions" if hasattr(mp, "solutions") else "tasks"

        if self._hands_api == "solutions":
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_num_hands,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.mp_draw = mp.solutions.drawing_utils
            self.mp_styles = mp.solutions.drawing_styles
        else:
            # Tasks API fallback (MediaPipe wheel without mp.solutions)
            self.hands = None


        # Indices for fingertips
        self.tip_ids = [4, 8, 12, 16, 20]

    def _ensure_tasks(self):
        from mediapipe.tasks.python import vision
        from mediapipe.tasks import python

        # Try to locate hand landmarker model inside mediapipe package.
        # If not found, user must install correct model asset.
        model_candidates = [
            "hand_landmarker.task",
            "hand_landmarker_short_range.task",
            "hand_landmarker_full.task",
        ]
        import os
        import mediapipe

        mp_root = os.path.dirname(mediapipe.__file__)
        found = None

        # 1. Check current directory first
        for c in model_candidates:
            if os.path.exists(c):
                found = c
                break

        # 2. Check mediapipe package directory
        if not found:
            for root, _, files in os.walk(mp_root):
                for c in model_candidates:
                    if c in files:
                        found = os.path.join(root, c)
                        break
                if found:
                    break

        # 3. Auto-download if not found
        if not found:
            import urllib.request
            print("Downloading MediaPipe HandLandmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            found = "hand_landmarker.task"
            try:
                urllib.request.urlretrieve(url, found)
                print("Download complete.")
            except Exception as e:
                raise RuntimeError(f"Failed to download model: {e}")

        base_options = python.BaseOptions(model_asset_path=found)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.6,
        )
        self.hands = vision.HandLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray, draw: bool = False) -> List[HandState]:
        h, w = frame_bgr.shape[:2]
        out: List[HandState] = []

        if self._hands_api == "solutions":
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False
            res = self.hands.process(img_rgb)
            img_rgb.flags.writeable = True

            if res.multi_hand_landmarks:
                for idx, hand_lms in enumerate(res.multi_hand_landmarks):
                    handedness = "Left"
                    if res.multi_handedness and idx < len(res.multi_handedness):
                        handedness = res.multi_handedness[idx].classification[0].label

                    lm_list: List[Tuple[int, int]] = []
                    xs = []
                    ys = []
                    for lm in hand_lms.landmark:
                        x_px = int(lm.x * w)
                        y_px = int(lm.y * h)
                        lm_list.append((x_px, y_px))
                        xs.append(x_px)
                        ys.append(y_px)

                    x0, x1 = min(xs), max(xs)
                    y0, y1 = min(ys), max(ys)
                    bbox = (x0, y0, x1 - x0, y1 - y0)

                    finger_bits = self._count_fingers(lm_list, handedness)
                    out.append(
                        HandState(
                            handedness=handedness,
                            lm_list=lm_list,
                            finger_bits=finger_bits,
                            bbox=bbox,
                        )
                    )

                    if draw:
                        self.mp_draw.draw_landmarks(
                            frame_bgr,
                            hand_lms,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_styles.get_default_hand_landmarks_style(),
                            self.mp_styles.get_default_hand_connections_style(),
                        )

            return out

        # Tasks API fallback
        self._ensure_tasks()

        # tasks expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        res = self.hands.detect(mp_image)

        if not res or not res.hand_landmarks:
            return []

        for hand_landmarks in res.hand_landmarks:
            lm_list: List[Tuple[int, int]] = []
            xs = []
            ys = []
            for lm in hand_landmarks:
                x_px = int(lm.x * w)
                y_px = int(lm.y * h)
                lm_list.append((x_px, y_px))
                xs.append(x_px)
                ys.append(y_px)

            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            bbox = (x0, y0, x1 - x0, y1 - y0)

            handedness = "Right"  # tasks output may include classification; keep default
            finger_bits = self._count_fingers(lm_list, handedness)
            out.append(
                HandState(
                    handedness=handedness,
                    lm_list=lm_list,
                    finger_bits=finger_bits,
                    bbox=bbox,
                )
            )

        return out

    def _count_fingers(self, lm_list: List[Tuple[int, int]], handedness: str) -> List[int]:
        """Return finger up bits [thumb,index,middle,ring,pinky].

        Heuristic:
        - Index..Pinky: finger up if tip y < PIP y.
        - Thumb: use x comparison depending on handedness.
        """
        bits = [0, 0, 0, 0, 0]

        thumb_tip_x = lm_list[4][0]
        thumb_ip_x = lm_list[3][0]

        if handedness == "Right":
            bits[0] = 1 if thumb_tip_x > thumb_ip_x else 0
        else:
            bits[0] = 1 if thumb_tip_x < thumb_ip_x else 0

        for i, tip_id in enumerate([8, 12, 16, 20], start=1):
            pip_id = tip_id - 2
            bits[i] = 1 if lm_list[tip_id][1] < lm_list[pip_id][1] else 0

        return bits
