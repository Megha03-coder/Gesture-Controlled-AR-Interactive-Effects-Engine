import cv2
import time
import threading
from config import Config

# --- FUTURE VISIONFX MODULES (To be implemented) ---
from tracking.hand_tracking import AdvancedHandTracker
from audio.voice_commands import VoiceAssistant

class VisionFXEngine:
    def __init__(self):
        print(f"Booting {Config.PROJECT_NAME} v{Config.VERSION}...")
        
        self.cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.RESOLUTION[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.RESOLUTION[1])
        
        # Initialize sub-systems (Placeholders for Phase 2)
        self.hand_tracker = AdvancedHandTracker()
        self.voice = VoiceAssistant()
        
        self.is_running = True
        self.active_mode = "idle"  # idle, draw, pc_control, presentation, magic

    def _voice_listener_loop(self):
        """Runs on a separate thread to listen for voice commands without lagging the video."""
        print("Voice Assistant Thread Started. Listening for commands...")
        while self.is_running:
            command = self.voice.listen()
            if command:
                self._handle_voice_command(command)

    def _handle_voice_command(self, command: str):
        if "fire" in command:
            self.voice.speak("Activating Fire Mode")
            self.active_mode = "magic"
        elif "draw" in command:
            self.voice.speak("Virtual Painter initialized")
            self.active_mode = "draw"
        elif "screenshot" in command:
            self.voice.speak("Capture sequence initiated")
        elif "control" in command or "mouse" in command:
            self.voice.speak("System control mode activated")
            self.active_mode = "pc_control"
        elif "idle" in command or "stop" in command or "deactivate" in command:
            self.voice.speak("Deactivating systems")
            self.active_mode = "idle"

    def start(self):
        if Config.ENABLE_VOICE_ASSISTANT:
            voice_thread = threading.Thread(target=self._voice_listener_loop, daemon=True)
            voice_thread.start()

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