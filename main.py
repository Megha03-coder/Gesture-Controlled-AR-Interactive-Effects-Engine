import cv2
import time
import sys
import os
from config import Config

# --- FUTURE VISIONFX MODULES (To be implemented) ---
from tracking.hand_tracking import AdvancedHandTracker

# Add GestureFX to path to import effects modules
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "GestureFX"))
from filters import Filters
from particle_system import ParticleSystem

class VisionFXEngine:
    def __init__(self):
        print(f"Booting {Config.PROJECT_NAME} v{Config.VERSION}...")
        
        self.cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.RESOLUTION[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.RESOLUTION[1])
        
        # Initialize sub-systems (Placeholders for Phase 2)
        self.hand_tracker = AdvancedHandTracker()
        self.particle_sys = ParticleSystem(max_particles=1600)
        self.filter_engine = Filters(self.particle_sys)
        
        self.is_running = True
        self.modes = ["Normal", "Black & White", "Neon Glow", "Rainbow Wave", "Prism Split", "Glitch Mode", "Cartoon", "Fire Trail", "Hologram"]
        self.active_mode = "Normal"
        self.menu_items = []
        self.hover_start_time = 0
        self.hovered_item = None
        self.hover_duration = 1.0  # Seconds required to hover and select
        self._init_menu()

    def _init_menu(self):
        # Define bounding boxes for each mode button
        start_y = 60
        for i, mode in enumerate(self.modes):
            rect = {
                "name": mode,
                "x1": 20,
                "y1": start_y + (i * 70),
                "x2": 220,
                "y2": start_y + (i * 70) + 50
            }
            self.menu_items.append(rect)

    def _draw_menu(self, frame):
        # Draw semi-transparent background to make text visible
        overlay = frame.copy()
        for item in self.menu_items:
            cv2.rectangle(overlay, (item["x1"], item["y1"]), (item["x2"], item["y2"]), (0, 0, 0), -1)
           
        # Safely force the blended overlay back into the original frame array
        frame[:] = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
        
        for item in self.menu_items:
            color = (0, 255, 0) if item["name"] == self.active_mode else (255, 255, 255)
            cv2.rectangle(frame, (item["x1"], item["y1"]), (item["x2"], item["y2"]), color, 2)
            cv2.putText(frame, item["name"].upper(), (item["x1"] + 10, item["y1"] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
    def _check_menu_interaction(self, x, y, frame):
        hovering_any = False
        for item in self.menu_items:
            if item["x1"] < x < item["x2"] and item["y1"] < y < item["y2"]:
                hovering_any = True
                if self.hovered_item != item["name"]:
                    self.hovered_item = item["name"]
                    self.hover_start_time = time.time()
                else:
                    elapsed = time.time() - self.hover_start_time
                    progress = min(1.0, elapsed / self.hover_duration)
                    
                    # Draw loading bar at the bottom of the hovered button
                    bar_w = int((item["x2"] - item["x1"]) * progress)
                    cv2.rectangle(frame, (item["x1"], item["y2"] - 10), (item["x1"] + bar_w, item["y2"]), (0, 255, 255), -1)
                    
                    if elapsed >= self.hover_duration:
                        self.active_mode = item["name"]
                break
                
        if not hovering_any:
            self.hovered_item = None

    def start(self):
        prev_time = time.time()

        while self.cap.isOpened() and self.is_running:
            success, frame = self.cap.read()
            if not success:
                break
                
            frame = cv2.flip(frame, 1) # Mirror display
            
            # --- 1. TRACKING PHASE ---
            # draw=True to see basic skeleton lines
            hands = self.hand_tracker.process(frame, draw=True)
            
            # Extract standard landmarks for filters & menu
            compat_hand_lm = None
            if hands:
                try:
                    hand = hands[0]
                    lm_list = getattr(hand, 'lm_list', hand)
                    if len(lm_list) > 8:
                        compat_hand_lm = [lm[1:3] if len(lm) == 3 else lm[0:2] for lm in lm_list]
                except Exception:
                    pass

            # --- 1.5 APPLY VISUAL EFFECTS ---
            if self.active_mode == "Black & White":
                frame = self.filter_engine.black_and_white(frame)
            elif self.active_mode == "Neon Glow":
                frame = self.filter_engine.neon_glow(frame)
            elif self.active_mode == "Rainbow Wave":
                frame = self.filter_engine.rainbow_wave(frame)
            elif self.active_mode == "Prism Split":
                frame = self.filter_engine.prism_split(frame)
            elif self.active_mode == "Glitch Mode":
                frame = self.filter_engine.glitch_mode(frame)
            elif self.active_mode == "Cartoon":
                frame = self.filter_engine.cartoon(frame)
            elif self.active_mode == "Fire Trail":
                frame = self.filter_engine.fire_trail(frame, compat_hand_lm)
            elif self.active_mode == "Hologram":
                frame = self.filter_engine.hologram(frame)

            # Draw menu base first so selection animations draw on top
            self._draw_menu(frame)

            # --- 2. MENU INTERACTION ---
            if compat_hand_lm:
                try:
                    cx, cy = compat_hand_lm[8]
                    cv2.circle(frame, (int(cx), int(cy)), 10, (255, 0, 255), cv2.FILLED)
                    self._check_menu_interaction(int(cx), int(cy), frame)
                except Exception:
                    pass

            curr_time = time.time()
            fps = int(1 / (curr_time - prev_time)) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # --- 3. ON-SCREEN DISPLAY (OSD) ---
            cv2.putText(frame, f"ACTIVE MODE: {self.active_mode.upper()}", (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"FPS: {fps}", (250, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow(f"{Config.PROJECT_NAME} Interface", frame)

            if cv2.waitKey(1) & 0xFF == 27: # ESC to quit
                self.is_running = False

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    engine = VisionFXEngine()
    engine.start()