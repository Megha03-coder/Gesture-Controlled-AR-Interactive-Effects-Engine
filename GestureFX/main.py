from __future__ import annotations

import time

import cv2

from effect_manager import EffectManager
from gesture_detector import GestureDetector
from hand_tracking import HandTracker
from utils import FPSCounter, draw_text


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check permissions/device index.")

    # Slight performance tweaks
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    tracker = HandTracker(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
    detector = GestureDetector()
    effects = EffectManager()
    fps = FPSCounter()

    active_gesture = "none"

    # Debounce gesture switching
    last_effect_id = "none"
    effect_change_cooldown_ms = 150
    last_change_ms = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Mirror for natural interaction
        frame = cv2.flip(frame, 1)

        hands = tracker.process(frame, draw=False)

        # UI info
        current_fingers = [0, 0, 0, 0, 0]

        if hands:
            hand = hands[0]
            current_fingers = hand.finger_bits
            detected = detector.detect(current_fingers)

            # override victory vs two-fingers
            effect_id = effects.maybe_override_victory(detected.gesture_id, hand)

            # Switch effects with debounce
            t_ms = int(time.time() * 1000)
            if effect_id != last_effect_id and (t_ms - last_change_ms) > effect_change_cooldown_ms:
                effects.trigger(effect_id, hand_lm=hand.lm_list, bbox=hand.bbox, frame=frame)
                last_effect_id = effect_id
                last_change_ms = t_ms
            active_gesture = detected.gesture_name

            # Render effects
            frame = effects.render(frame, current_fingers, hand_state=hand)

            # Draw landmarks for clarity (optional):
            # tracker.process(frame, draw=True)  # would process twice; keep off for performance

        else:
            active_gesture = "No hand"

        fps.tick()
        h_frame, w_frame = frame.shape[:2]

        # Display gesture and effect in the bottom left, only when a hand is detected
        if active_gesture != "No hand" and effects.active_effect != "none":
            # Draw a subtle translucent dark background for readability
            text_overlay = frame.copy()
            cv2.rectangle(text_overlay, (10, h_frame - 80), (350, h_frame - 10), (0, 0, 0), -1)
            cv2.addWeighted(text_overlay, 0.5, frame, 0.5, 0, frame)

            draw_text(frame, f"Gesture: {active_gesture}", (20, h_frame - 50), color=(255, 255, 255))
            draw_text(frame, f"Effect: {effects.active_effect}", (20, h_frame - 20), color=(0, 255, 255))

        cv2.imshow("GestureFX AI", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord(' '):
            filename = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved as {filename}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
