import cv2
import mediapipe as mp
import math
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class FaceData:
    bbox: Tuple[int, int, int, int]
    emotion: str
    lm_list: List[Tuple[int, int]]

class FaceEmotionTracker:
    def __init__(self, max_faces=1):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process(self, frame, draw=False) -> List[FaceData]:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        
        faces_data = []
        h, w, c = frame.shape
        
        if results.multi_face_landmarks:
            for face_lms in results.multi_face_landmarks:
                lm_list = []
                x_list, y_list = [], []
                for lm in face_lms.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append((cx, cy))
                    x_list.append(cx)
                    y_list.append(cy)
                    
                xmin, xmax = min(x_list), max(x_list)
                ymin, ymax = min(y_list), max(y_list)
                bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
                
                # Heuristic Emotion Detection using Landmarks
                emotion = self._detect_emotion(lm_list)
                
                faces_data.append(FaceData(bbox=bbox, emotion=emotion, lm_list=lm_list))
                
                if draw:
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 1)
                    
        return faces_data

    def _detect_emotion(self, lm_list) -> str:
        try:
            mouth_width = math.dist(lm_list[61], lm_list[291]) # Left and Right mouth corners
            mouth_height = math.dist(lm_list[13], lm_list[14]) # Top and Bottom inner lips
            face_width = math.dist(lm_list[234], lm_list[454]) # Left and Right cheek edges
            
            if face_width == 0: return "NEUTRAL"
            
            smile_ratio = mouth_width / face_width
            open_ratio = mouth_height / mouth_width
            
            if open_ratio > 0.4: return "SURPRISED"
            elif smile_ratio > 0.42: return "HAPPY"
            else: return "NEUTRAL"
        except IndexError:
            return "NEUTRAL"