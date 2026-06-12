from __future__ import annotations

import cv2

from gesture_detector import GestureDetector
from filters import Filters
from hand_tracking import HandTracker
from particle_system import ParticleSystem
from utils import FPSCounter, now_ms


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check permissions/device index.")

    # Slight performance tweaks
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    tracker = HandTracker(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
    detector = GestureDetector()
    particle_sys = ParticleSystem(max_particles=1600)
    filter_engine = Filters(particle_sys)
    fps = FPSCounter()

    active_filter_name = "Normal"
    filter_paused = False

    last_action_ms = 0
    debounce_ms = 1000  # 1 second debounce for gesture actions

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Mirror for natural interaction
        frame = cv2.flip(frame, 1)

        # draw=True will show the skeleton lines to confirm tracking works
        hands = tracker.process(frame, draw=True)
        active_gesture_name = "None"
        hand_lm = None
        finger_status = ""

        t_ms = now_ms()

        if hands:
            hand = hands[0]
            hand_lm = hand.lm_list
            finger_status = str(hand.finger_bits)
            detected = detector.detect(hand.finger_bits)
            active_gesture_name = detected.gesture_name
            gesture_id = detected.gesture_id

            if gesture_id != "none" and (t_ms - last_action_ms) > debounce_ms:
                if gesture_id == "toggle_pause":
                    filter_paused = not filter_paused
                    last_action_ms = t_ms
                elif not filter_paused:
                    if gesture_id == "black_white":
                        active_filter_name = "Black & White"
                        last_action_ms = t_ms
                    elif gesture_id == "neon_glow":
                        active_filter_name = "Neon Glow"
                        last_action_ms = t_ms
                    elif gesture_id == "cartoon":
                        active_filter_name = "Cartoon"
                        last_action_ms = t_ms
                    elif gesture_id == "reset":
                        active_filter_name = "Normal"
                        last_action_ms = t_ms

        # Apply the active filter
        if active_filter_name == "Black & White":
            frame = filter_engine.black_and_white(frame)
        elif active_filter_name == "Cartoon":
            frame = filter_engine.cartoon(frame)
        elif active_filter_name == "Neon Glow":
            frame = filter_engine.neon_glow(frame)
        elif active_filter_name == "Rainbow Wave":
            frame = filter_engine.rainbow_wave(frame)
        elif active_filter_name == "Prism Split":
            frame = filter_engine.prism_split(frame)
        elif active_filter_name == "Thermal Vision":
            frame = filter_engine.thermal_vision(frame)
        elif active_filter_name == "Glitch Mode":
            frame = filter_engine.glitch_mode(frame)
        elif active_filter_name == "Kaleidoscope":
            frame = filter_engine.kaleidoscope(frame)
        elif active_filter_name == "Fire Trail":
            frame = filter_engine.fire_trail(frame, hand_lm)
        elif active_filter_name == "Hologram":
            frame = filter_engine.hologram(frame)

        # Update UI Overlays
        current_fps = fps.tick()
        
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Gesture: {active_gesture_name} {finger_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        status = " (PAUSED)" if filter_paused else ""
        cv2.putText(frame, f"Filter: {active_filter_name}{status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("GestureFX AI", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
