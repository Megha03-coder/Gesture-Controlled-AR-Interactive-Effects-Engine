
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
        self._face_api = "solutions" if hasattr(mp, "solutions") else "tasks"
        
        if self._face_api == "solutions":
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=max_faces,
                refine_landmarks=True,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6
            )
            self.mp_draw = mp.solutions.drawing_utils
        else:
            self.max_faces = max_faces
            self._ensure_tasks()

    def _ensure_tasks(self):
        from mediapipe.tasks.python import vision
        from mediapipe.tasks import python
        import os
        import mediapipe
        
        model_candidates = ["face_landmarker.task"]
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
            print("Downloading MediaPipe FaceLandmarker model for Tasks API...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            found = "face_landmarker.task"
            urllib.request.urlretrieve(url, found)
            print("Download complete.")

        base_options = python.BaseOptions(model_asset_path=found)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=self.max_faces,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.6,
            min_tracking_confidence=0.6,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.face_mesh = vision.FaceLandmarker.create_from_options(options)

    def process(self, frame, draw=False) -> List[FaceData]:
        faces_data = []
        h, w, c = frame.shape
        
        if self._face_api == "solutions":
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(img_rgb)
            
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
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            res = self.face_mesh.detect(mp_image)

            if res and res.face_landmarks:
                for face_landmarks in res.face_landmarks:
                    lm_list = []
                    x_list, y_list = [], []
                    for lm in face_landmarks:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lm_list.append((cx, cy))
                        x_list.append(cx)
                        y_list.append(cy)
                        
                    xmin, xmax = min(x_list), max(x_list)
                    ymin, ymax = min(y_list), max(y_list)
                    bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
                    
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