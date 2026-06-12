from __future__ import annotations

import cv2

from hand_tracking import HandTracker
from utils import FPSCounter


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check permissions/device index.")

    # Slight performance tweaks
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    tracker = HandTracker(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
    fps = FPSCounter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Mirror for natural interaction
        frame = cv2.flip(frame, 1)

        # Draw landmarks for clarity
        hands = tracker.process(frame, draw=True)

        fps.tick()

        cv2.imshow("GestureFX AI", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
