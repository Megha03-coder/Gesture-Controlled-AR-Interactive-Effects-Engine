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
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

    def process(self, frame, draw=False) -> List[PoseData]:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        
        poses_data = []
        h, w, c = frame.shape
        
        if results.pose_landmarks:
            lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in results.pose_landmarks.landmark]
            
            x_list, y_list = zip(*lm_list)
            bbox = (min(x_list), min(y_list), max(x_list) - min(x_list), max(y_list) - min(y_list))
            poses_data.append(PoseData(lm_list=lm_list, bbox=bbox))
            
            if draw:
                self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=self.mp_styles.get_default_pose_landmarks_style())
                
        return poses_data