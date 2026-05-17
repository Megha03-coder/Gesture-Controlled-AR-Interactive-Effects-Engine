class DeepGestureDetector:
    def __init__(self):
        # Dictionary mapping finger states [Thumb, Index, Middle, Ring, Pinky] to specific actions
        self.gesture_map = {
            (0, 1, 0, 0, 0): "FIRE MODE",
            (0, 1, 1, 0, 0): "LIGHTNING / SPARK",
            (1, 1, 1, 1, 1): "OPEN PALM (RGB)",
            (0, 0, 0, 0, 0): "CLOSED FIST (FREEZE)",
            (1, 0, 0, 0, 0): "THUMBS UP",
            (1, 0, 0, 0, 1): "ROCK SIGN",
            (0, 0, 0, 0, 1): "PINKY UP (SMOKE)",
            (0, 1, 1, 1, 1): "MAGIC CIRCLE"
        }
        
        # History buffer for temporal smoothing (prevents flickering)
        self.history = []
        self.smoothing_frames = 5

    def analyze(self, hands, faces=None) -> str:
        if not hands:
            self.history.clear()
            return "NO TARGET DETECTED"

        hand = hands[0]
        if hasattr(hand, 'finger_bits'):
            bits = tuple(hand.finger_bits)
            self.history.append(bits)
            if len(self.history) > self.smoothing_frames:
                self.history.pop(0)
                
            smoothed_bits = max(set(self.history), key=self.history.count)
            return self.gesture_map.get(smoothed_bits, "UNKNOWN PATTERN")
            
        return "ANALYZING..."