import math

class DeepGestureDetector:
    """
    Analyzes hand landmarks to detect both simple finger counts and
    more complex gestures like the Vulcan Salute.
    """
    def _get_finger_status(self, hand):
        """Determines which of the 5 fingers are extended."""
        lm = hand.lm_list
        tip_ids = [4, 8, 12, 16, 20]
        fingers = []

        # Thumb (special case: compare x-coordinates relative to wrist)
        if hand.handedness == "Right":
            if lm[tip_ids[0]][0] < lm[tip_ids[0] - 1][0]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:  # Left hand
            if lm[tip_ids[0]][0] > lm[tip_ids[0] - 1][0]:
                fingers.append(1)
            else:
                fingers.append(0)

        # 4 Fingers (compare y-coordinates of tip and joint)
        for id in range(1, 5):
            # A more robust check is to see if the fingertip is higher than the
            # joint two segments down (MCP joint). This prevents bent fingers
            # from being accidentally counted as "up".
            if lm[tip_ids[id]][1] < lm[tip_ids[id] - 3][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def analyze(self, hands):
        """Analyzes the hand data and returns the name of the detected gesture."""
        if not hands:
            return "NO TARGET DETECTED"

        hand = hands[0]

        # Fallback to simple finger counting for other gestures.
        fingers = self._get_finger_status(hand)
        
        if fingers == [0, 1, 0, 0, 0]: return "FIRE MODE"
        if fingers == [0, 1, 1, 0, 0]: return "LIGHTNING / SPARK"
        if fingers == [1, 1, 1, 1, 1]: return "OPEN PALM (RGB)"
        if fingers == [0, 0, 0, 0, 0]: return "CLOSED FIST (FREEZE)"
        if fingers == [0, 0, 0, 0, 1]: return "PINKY UP (SMOKE)"
        if fingers == [1, 0, 0, 0, 0]: return "THUMBS UP"
        if fingers == [0, 1, 0, 0, 1]: return "ROCK SIGN"
        if fingers == [1, 1, 1, 1, 0]: return "MAGIC CIRCLE"
        if fingers == [0, 1, 1, 1, 0]: return "DAZZLING"
        if fingers == [0, 1, 1, 1, 1]: return "PRISM"

        return "UNKNOWN PATTERN"