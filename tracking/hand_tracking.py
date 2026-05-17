import cv2
import mediapipe as mp

from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class HandData:
    handedness: str
    lm_list: List[Tuple[int, int]]
    finger_bits: List[int]
    bbox: Tuple[int, int, int, int]

class AdvancedHandTracker:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        self._hands_api = "solutions" if hasattr(mp, "solutions") else "tasks"
        
        if self._hands_api == "solutions":
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
            self.mp_styles = mp.solutions.drawing_styles
            
            self.hands = self.mp_hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
        else:
            self.hands = None
            self.max_num_hands = max_num_hands
            self.min_detection_confidence = min_detection_confidence
            self.min_tracking_confidence = min_tracking_confidence
            self._ensure_tasks()

    def _ensure_tasks(self):
        from mediapipe.tasks.python import vision
        from mediapipe.tasks import python
        import os
        import mediapipe

        model_candidates = ["hand_landmarker.task"]
        mp_root = os.path.dirname(mediapipe.__file__)
        found = None

        for c in model_candidates:
            if os.path.exists(c):
                found = c
                break
        
        if not found:
            for root, _, files in os.walk(mp_root):
                for c in model_candidates:
                    if c in files:
                        found = os.path.join(root, c)
                        break
                if found:
                    break
        
        if not found:
            import urllib.request
            print("Downloading MediaPipe HandLandmarker model for Tasks API...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            found = "hand_landmarker.task"
            urllib.request.urlretrieve(url, found)
            print("Download complete.")

        base_options = python.BaseOptions(model_asset_path=found)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.hands = vision.HandLandmarker.create_from_options(options)

    def process(self, frame, draw=True) -> List[HandData]:
        hands_data = []
        h, w, c = frame.shape
        
        if self._hands_api == "solutions":
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                    handedness = results.multi_handedness[idx].classification[0].label
                    
                    lm_list = []
                    x_list, y_list = [] , []
                    for id, lm in enumerate(hand_lms.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lm_list.append((cx, cy))
                        x_list.append(cx)
                        y_list.append(cy)
                        
                    xmin, xmax = min(x_list), max(x_list)
                    ymin, ymax = min(y_list), max(y_list)
                    bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
                    
                    finger_bits = self._get_finger_bits(lm_list, handedness)
                    
                    hands_data.append(HandData(
                        handedness=handedness,
                        lm_list=lm_list,
                        finger_bits=finger_bits,
                        bbox=bbox
                    ))
                    
                    if draw:
                        self.mp_draw.draw_landmarks(
                            frame, hand_lms, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_styles.get_default_hand_landmarks_style(),
                            self.mp_styles.get_default_hand_connections_style()
                        )
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            res = self.hands.detect(mp_image)

            if res and res.hand_landmarks:
                for idx, hand_landmarks in enumerate(res.hand_landmarks):
                    handedness = "Right"
                    if res.handedness and idx < len(res.handedness):
                        handedness = res.handedness[idx][0].category_name
                        
                    lm_list = []
                    x_list, y_list = [] , []
                    for lm in hand_landmarks:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lm_list.append((cx, cy))
                        x_list.append(cx)
                        y_list.append(cy)
                        
                    xmin, xmax = min(x_list), max(x_list)
                    ymin, ymax = min(y_list), max(y_list)
                    bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
                    
                    finger_bits = self._get_finger_bits(lm_list, handedness)
                    
                    hands_data.append(HandData(
                        handedness=handedness,
                        lm_list=lm_list,
                        finger_bits=finger_bits,
                        bbox=bbox
                    ))
                    
                    if draw:
                        for cx, cy in lm_list:
                            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
                            
        return hands_data
        
    def _get_finger_bits(self, lm_list, handedness) -> List[int]:
        bits = [0, 0, 0, 0, 0]
        
        # Thumb check (comparing X coordinates, inverted for mirrored cameras)
        if handedness == "Right":
            bits[0] = 1 if lm_list[4][0] > lm_list[3][0] else 0
        else:
            bits[0] = 1 if lm_list[4][0] < lm_list[3][0] else 0
            
        # Other 4 fingers (comparing Y tip coordinates vs Y joint coordinates)
        for i, tip_id in enumerate([8, 12, 16, 20], start=1):
            bits[i] = 1 if lm_list[tip_id][1] < lm_list[tip_id - 2][1] else 0
            
        return bits