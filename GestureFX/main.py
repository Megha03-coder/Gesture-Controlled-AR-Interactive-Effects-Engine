from __future__ import annotations

import cv2
import time

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

    # --- MENU SETUP ---
    modes = ["Normal", "Black & White", "Cartoon", "Neon Glow", "Rainbow Wave", "Prism Split", "Glitch Mode", "Fire Trail", "Hologram"]
    menu_items = []
    start_y = 15
    for i, mode in enumerate(modes):
        menu_items.append({
            "name": mode,
            "x1": 460, # Right side of the 640px wide frame
            "y1": start_y + (i * 50),
            "x2": 620,
            "y2": start_y + (i * 50) + 40
        })
    hover_start_time = 0
    hovered_item = None
    hover_duration = 1.0

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
                if not filter_paused:
                    if gesture_id == "black_white":
                        active_filter_name = "Black & White"
                        last_action_ms = t_ms
                    elif gesture_id == "neon_glow":
                        active_filter_name = "Neon Glow"
                        last_action_ms = t_ms
                    elif gesture_id == "cartoon":
                        active_filter_name = "Cartoon"
                        last_action_ms = t_ms
                    elif gesture_id == "rainbow_wave":
                        active_filter_name = "Rainbow Wave"
                        last_action_ms = t_ms
                    elif gesture_id == "hologram":
                        active_filter_name = "Hologram"
                        last_action_ms = t_ms
                    elif gesture_id == "glitch_mode":
                        active_filter_name = "Glitch Mode"
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

        # --- MENU INTERACTION & DRAWING ---
        overlay = frame.copy()
        for item in menu_items:
            cv2.rectangle(overlay, (item["x1"], item["y1"]), (item["x2"], item["y2"]), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        if hand_lm and len(hand_lm) > 8:
            try:
                lm8 = hand_lm[8]
                cx, cy = lm8[1:3] if len(lm8) == 3 else lm8[0:2]
                cx, cy = int(cx), int(cy)
                
                cv2.circle(frame, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                
                hovering_any = False
                for item in menu_items:
                    if item["x1"] < cx < item["x2"] and item["y1"] < cy < item["y2"]:
                        hovering_any = True
                        if hovered_item != item["name"]:
                            hovered_item = item["name"]
                            hover_start_time = time.time()
                        else:
                            elapsed = time.time() - hover_start_time
                            progress = min(1.0, elapsed / hover_duration)
                            bar_w = int((item["x2"] - item["x1"]) * progress)
                            cv2.rectangle(frame, (item["x1"], item["y2"] - 5), (item["x1"] + bar_w, item["y2"]), (0, 255, 255), -1)
                            
                            if elapsed >= hover_duration:
                                active_filter_name = item["name"]
                        break
                if not hovering_any:
                    hovered_item = None
            except Exception:
                pass
                
        for item in menu_items:
            color = (0, 255, 0) if item["name"] == active_filter_name else (255, 255, 255)
            cv2.rectangle(frame, (item["x1"], item["y1"]), (item["x2"], item["y2"]), color, 2)
            cv2.putText(frame, item["name"], (item["x1"] + 10, item["y1"] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow("GestureFX AI", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
