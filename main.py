import cv2
import time
import threading
from config import Config

# --- FUTURE VISIONFX MODULES (To be implemented) ---
from tracking.hand_tracking import AdvancedHandTracker
from tracking.face_tracking import FaceEmotionTracker
from gestures.gesture_detector import DeepGestureDetector
from audio.voice_commands import VoiceAssistant
from ui.futuristic_hud import IronManHUD
from controls.system_controls import SystemController
from effects.particle_engine import ParticleEngine

class VisionFXEngine:
    def __init__(self):
        print(f"Booting {Config.PROJECT_NAME} v{Config.VERSION}...")
        
        self.cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.RESOLUTION[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.RESOLUTION[1])
        
        # Initialize sub-systems (Placeholders for Phase 2)
        self.hand_tracker = AdvancedHandTracker()
        self.face_tracker = FaceEmotionTracker()
        self.hud = IronManHUD()
        self.voice = VoiceAssistant()
        self.gesture_detector = DeepGestureDetector()
        self.particles = ParticleEngine()
        self.system_controls = SystemController()
        
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
        elif "music" in command:
            self.voice.speak("Playing background music")
            self.active_mode = "music"
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
            # draw=False removes the yellow landmark dots and skeleton lines
            hands = self.hand_tracker.process(frame, draw=False)
            faces = self.face_tracker.process(frame, draw=False)
            
            # --- 2. GESTURE & LOGIC PHASE ---
            active_gestures = self.gesture_detector.analyze(hands)
            
            if self.active_mode == "pc_control" and Config.ENABLE_SYSTEM_CONTROL:
                self.system_controls.process(hands, active_gestures)
            
            # --- 3. EFFECTS & RENDER PHASE ---
            # Darken the camera feed (turning bright white into grey) so glowing effects pop beautifully
            frame = cv2.addWeighted(frame, 0.6, frame, 0, 0)
            
            frame = self.particles.render(frame, active_gestures, hands, self.active_mode)
            
            curr_time = time.time()
            fps = int(1 / (curr_time - prev_time)) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # Draw futuristic HUD
            frame = self.hud.draw(frame, fps, active_gestures, self.active_mode, faces)

            cv2.imshow(f"{Config.PROJECT_NAME} Interface", frame)

            if cv2.waitKey(1) & 0xFF == 27: # ESC to quit
                self.is_running = False

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    engine = VisionFXEngine()
    engine.start()