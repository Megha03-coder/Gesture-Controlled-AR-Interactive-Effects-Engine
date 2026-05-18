import cv2
import mediapipe as mp
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class PoseData:
    lm_list: List[Tuple[int, int]]
    bbox: Tuple[int, int, int, int]

class BodyPoseTracker:
    def __init__(self, static_image_mode=False, min_detection_confidence=0.6, min_tracking_confidence=0.6):
        self._pose_api = "solutions" if hasattr(mp, "solutions") else "tasks"
        
        if self._pose_api == "solutions":
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=static_image_mode,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.mp_draw = mp.solutions.drawing_utils
            self.mp_styles = mp.solutions.drawing_styles
        else:
            self.min_detection_confidence = min_detection_confidence
            self.min_tracking_confidence = min_tracking_confidence
            self._ensure_tasks()

    def _ensure_tasks(self):
        from mediapipe.tasks.python import vision
        from mediapipe.tasks import python
        import os
        import mediapipe

        model_candidates = ["pose_landmarker.task"]
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
            print("Downloading MediaPipe PoseLandmarker model for Tasks API...")
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
            found = "pose_landmarker.task"
            urllib.request.urlretrieve(url, found)
            print("Download complete.")

        base_options = python.BaseOptions(model_asset_path=found)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.pose = vision.PoseLandmarker.create_from_options(options)

    def process(self, frame, draw=False) -> List[PoseData]:
        poses_data = []
        h, w, c = frame.shape
        
        if self._pose_api == "solutions":
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(img_rgb)
            
            if results.pose_landmarks:
                lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in results.pose_landmarks.landmark]
                
                x_list, y_list = zip(*lm_list)
                bbox = (min(x_list), min(y_list), max(x_list) - min(x_list), max(y_list) - min(y_list))
                poses_data.append(PoseData(lm_list=lm_list, bbox=bbox))
                
                if draw:
                    self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=self.mp_styles.get_default_pose_landmarks_style())
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            res = self.pose.detect(mp_image)
            
            if res and res.pose_landmarks:
                for pose_landmarks in res.pose_landmarks:
                    lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in pose_landmarks]
                    
                    x_list, y_list = zip(*lm_list)
                    bbox = (min(x_list), min(y_list), max(x_list) - min(x_list), max(y_list) - min(y_list))
                    poses_data.append(PoseData(lm_list=lm_list, bbox=bbox))
                    
                    if draw:
                        for cx, cy in lm_list:
                            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                
        return poses_data