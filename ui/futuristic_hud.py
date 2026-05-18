import cv2
import numpy as np
import time
import math
import random
from config import Config

class IronManHUD:
    def __init__(self):
        self.main_color = Config.HUD_COLOR_MAIN
        self.alert_color = Config.HUD_COLOR_ALERT
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Simulated neural-network data stream
        self.data_stream = [f"0x{random.randint(1000, 9999):04X}" for _ in range(10)]

    def draw(self, frame: np.ndarray, fps: int, active_gestures: str, active_mode: str, faces: list = None) -> np.ndarray:
        h, w = frame.shape[:2]
        t = time.time()
        
        # Create a blank overlay to draw the HUD on (allows for transparency blending)
        overlay = frame.copy()
        
        # 1. Draw corner viewfinder brackets
        self._draw_brackets(overlay, w, h)
        
        # 2. Draw rotating center reticle
        self._draw_reticle(overlay, w, h, t)
        
        # 3. Draw side informational panels & data streams
        self._draw_info_panel(overlay, fps, active_gestures, active_mode, w, h, t, faces)
        
        # 4. Global tracking crosshair grid
        self._apply_fx(overlay, w, h)
        
        # Blend overlay over the original frame to give it a glowing/glass effect
        return cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
        
    def _draw_brackets(self, frame, w, h):
        length = 60
        thick = 2
        margin = 30
        # Top-Left
        cv2.line(frame, (margin, margin), (margin + length, margin), self.main_color, thick, cv2.LINE_AA)
        cv2.line(frame, (margin, margin), (margin, margin + length), self.main_color, thick, cv2.LINE_AA)
        # Top-Right
        cv2.line(frame, (w - margin, margin), (w - margin - length, margin), self.main_color, thick, cv2.LINE_AA)
        cv2.line(frame, (w - margin, margin), (w - margin, margin + length), self.main_color, thick, cv2.LINE_AA)
        # Bottom-Left
        cv2.line(frame, (margin, h - margin), (margin + length, h - margin), self.main_color, thick, cv2.LINE_AA)
        cv2.line(frame, (margin, h - margin), (margin, h - margin - length), self.main_color, thick, cv2.LINE_AA)
        # Bottom-Right
        cv2.line(frame, (w - margin, h - margin), (w - margin - length, h - margin), self.main_color, thick, cv2.LINE_AA)
        cv2.line(frame, (w - margin, h - margin), (w - margin, h - margin - length), self.main_color, thick, cv2.LINE_AA)
        
    def _draw_reticle(self, frame, w, h, t):
        cx, cy = w // 2, h // 2
        
        # Outer ring 1 (Clockwise)
        angle1 = int(t * 45) % 360
        cv2.ellipse(frame, (cx, cy), (140, 140), angle1, 0, 80, self.main_color, 2, cv2.LINE_AA)
        cv2.ellipse(frame, (cx, cy), (140, 140), angle1, 180, 260, self.main_color, 2, cv2.LINE_AA)
        
        # Outer ring 2 (Counter-clockwise, Red/Alert colored)
        angle2 = int(-t * 70) % 360
        cv2.ellipse(frame, (cx, cy), (125, 125), angle2, 0, 100, self.alert_color, 1, cv2.LINE_AA)
        cv2.ellipse(frame, (cx, cy), (125, 125), angle2, 180, 280, self.alert_color, 1, cv2.LINE_AA)
        
        # Inner details & ticks
        cv2.circle(frame, (cx, cy), 70, self.main_color, 1, cv2.LINE_AA)
        for i in range(0, 360, 45):
            rad_i = math.radians(i + t * 15)
            x1, y1 = int(cx + math.cos(rad_i) * 70), int(cy + math.sin(rad_i) * 70)
            x2, y2 = int(cx + math.cos(rad_i) * 80), int(cy + math.sin(rad_i) * 80)
            cv2.line(frame, (x1, y1), (x2, y2), self.main_color, 2, cv2.LINE_AA)
            
        # Dead center target
        cv2.circle(frame, (cx, cy), 3, self.alert_color, -1)
        cv2.line(frame, (cx - 15, cy), (cx + 15, cy), self.main_color, 1)
        cv2.line(frame, (cx, cy - 15), (cx, cy + 15), self.main_color, 1)
        
    def _draw_info_panel(self, frame, fps, active_gestures, active_mode, w, h, t, faces):
        # Continuously update the fake neural data stream
        if int(t * 15) % 5 == 0:
            self.data_stream.pop(0)
            self.data_stream.append(f"0x{random.randint(1000, 9999):04X} : {random.randint(10, 99)}")
            
        # --- Left Panel ---
        left_x = 40
        cv2.putText(frame, "VISIONFX SYSTEM OP", (left_x, h // 2 - 80), self.font, 0.45, self.main_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"MODE:    {active_mode.upper()}", (left_x, h // 2 - 50), self.font, 0.55, self.main_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"GESTURE: {active_gestures}", (left_x, h // 2 - 20), self.font, 0.55, self.alert_color, 2, cv2.LINE_AA)
        
        fps_color = self.main_color if fps >= 30 else self.alert_color
        cv2.putText(frame, f"FPS:     {fps}", (left_x, h // 2 + 10), self.font, 0.55, fps_color, 2, cv2.LINE_AA)
        
        # Draw data feed
        for i, data in enumerate(self.data_stream[:5]):
            cv2.putText(frame, data, (left_x, h // 2 + 50 + i * 18), self.font, 0.35, self.main_color, 1, cv2.LINE_AA)
            
        # --- Right Panel ---
        right_x = w - 240
        cv2.putText(frame, "AI ENGINE STATUS", (right_x, h // 2 - 80), self.font, 0.45, self.main_color, 1, cv2.LINE_AA)
        cv2.putText(frame, "TARGETING:  ACTIVE", (right_x, h // 2 - 50), self.font, 0.45, self.main_color, 1, cv2.LINE_AA)
        cv2.putText(frame, "NEURAL NET: ALIGNED", (right_x, h // 2 - 30), self.font, 0.45, self.main_color, 1, cv2.LINE_AA)
        
        emotion = "SCANNING..."
        if faces:
            emotion = faces[0].emotion
        cv2.putText(frame, f"EMOTION:    {emotion}", (right_x, h // 2 - 10), self.font, 0.45, self.main_color, 1, cv2.LINE_AA)
        
        # Dynamic Sine Wave Visualizer
        wave_y = h // 2 + 20
        pts = [(right_x + i, int(wave_y + math.sin((i + t * 80) * 0.1) * 15)) for i in range(150)]
        cv2.polylines(frame, [np.array(pts, np.int32)], False, self.main_color, 1, cv2.LINE_AA)
        
    def _apply_fx(self, frame, w, h):
        # Subtle crosshair grid overlay across the entire screen
        step = 80
        for y in range(step, h, step):
            for x in range(step, w, step):
                cv2.line(frame, (x - 3, y), (x + 3, y), self.main_color, 1)
                cv2.line(frame, (x, y - 3), (x, y + 3), self.main_color, 1)