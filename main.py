import cv2
import time
from config import Config

# --- FUTURE VISIONFX MODULES (To be implemented) ---
from tracking.hand_tracking import AdvancedHandTracker

class VisionFXEngine:
    def __init__(self):
        print(f"Booting {Config.PROJECT_NAME} v{Config.VERSION}...")
        
        self.cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.RESOLUTION[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.RESOLUTION[1])
        
        # Initialize sub-systems (Placeholders for Phase 2)
        self.hand_tracker = AdvancedHandTracker()
        
        self.is_running = True
        self.active_mode = "idle"  # idle, draw, pc_control, presentation, magic

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
            
            curr_time = time.time()
            fps = int(1 / (curr_time - prev_time)) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            cv2.imshow(f"{Config.PROJECT_NAME} Interface", frame)

            if cv2.waitKey(1) & 0xFF == 27: # ESC to quit
                self.is_running = False

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    engine = VisionFXEngine()
    engine.start()