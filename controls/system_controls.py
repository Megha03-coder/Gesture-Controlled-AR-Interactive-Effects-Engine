import pyautogui
import time
import numpy as np
from config import Config

# Disable failsafe to prevent crashes when cursor hits the corners
pyautogui.FAILSAFE = False

class SystemController:
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()
        self.prev_x, self.prev_y = 0, 0
        
        # Smoothing factor (higher = smoother but more delay)
        self.smoothing = 5.0
        
        # Cooldowns to prevent spamming actions
        self.last_click_time = 0
        self.last_vol_time = 0

    def process(self, hands, active_gesture):
        if not hands:
            return

        hand = hands[0]
        lm = hand.lm_list
        
        # Use Index Finger Tip (Node 8) for mouse position
        x, y = lm[8][0], lm[8][1]
        
        cam_w, cam_h = Config.RESOLUTION
        
        # Define an active tracking area (margin) so the user doesn't 
        # have to stretch their arm to the very edges of the camera view
        margin_x, margin_y = 150, 100
        
        # Interpolate camera coordinates to screen coordinates
        screen_x = np.interp(x, (margin_x, cam_w - margin_x), (0, self.screen_w))
        screen_y = np.interp(y, (margin_y, cam_h - margin_y), (0, self.screen_h))
        
        # Apply low-pass filter for smooth cursor movement
        curr_x = self.prev_x + (screen_x - self.prev_x) / self.smoothing
        curr_y = self.prev_y + (screen_y - self.prev_y) / self.smoothing
        
        self.prev_x, self.prev_y = curr_x, curr_y
        
        now = time.time()

        # --- GESTURE TO ACTION MAPPING ---
        if active_gesture == "FIRE MODE":
            pyautogui.moveTo(curr_x, curr_y) # 1 Finger: Move Mouse
        elif active_gesture == "LIGHTNING / SPARK":
            pyautogui.moveTo(curr_x, curr_y) # 2 Fingers: Move and Click
            if now - self.last_click_time > 0.5:  # 500ms click cooldown
                pyautogui.click()
                self.last_click_time = now
        elif active_gesture == "THUMBS UP":
            if now - self.last_vol_time > 0.3:  # 300ms volume up cooldown
                pyautogui.press('volumeup')
                self.last_vol_time = now
        elif active_gesture == "PINKY UP (SMOKE)":
            if now - self.last_vol_time > 0.3:  # 300ms volume down cooldown
                pyautogui.press('volumedown')
                self.last_vol_time = now