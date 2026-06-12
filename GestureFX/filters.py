import cv2
import numpy as np
import time
import random
import math

from utils import now_ms

class Filters:
    def __init__(self, particle_system=None):
        self.particle_system = particle_system

    def black_and_white(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def cartoon(self, frame):
        # Bilateral filter for color smoothing
        color = cv2.bilateralFilter(frame, 9, 75, 75)
        
        # Edge detection using adaptive threshold
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        
        # Combine edges and color
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return cv2.bitwise_and(color, edges_color)

    def neon_glow(self, frame):
        # Edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Colorize edges (Cyan-magenta-ish tones)
        neon_color = np.zeros_like(frame)
        neon_color[:, :] = [255, 0, 255] # Magenta
        
        cyan = np.zeros_like(frame)
        cyan[:, :] = [255, 255, 0] # Cyan in BGR
        
        t = time.time()
        blend_factor = (math.sin(t * 3) + 1) / 2
        
        glow = cv2.addWeighted(neon_color, blend_factor, cyan, 1 - blend_factor, 0)
        colored_edges = cv2.bitwise_and(glow, edges_bgr)
        
        # Blur for neon glow effect
        blurred = cv2.GaussianBlur(colored_edges, (15, 15), 0)
        return cv2.add(frame, blurred)

    def rainbow_wave(self, frame):
        h, w = frame.shape[:2]
        t = time.time()
        
        # Create moving rainbow gradients
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xv, yv = np.meshgrid(x, y)
        
        hue = ((xv + yv + t) % 1.0 * 180).astype(np.uint8)
        sat = np.full((h, w), 255, dtype=np.uint8)
        val = np.full((h, w), 255, dtype=np.uint8)
        
        hsv_wave = cv2.merge([hue, sat, val])
        bgr_wave = cv2.cvtColor(hsv_wave, cv2.COLOR_HSV2BGR)
        
        # Overlay wave across entire frame smoothly
        return cv2.addWeighted(frame, 0.7, bgr_wave, 0.3, 0)

    def prism_split(self, frame):
        h, w = frame.shape[:2]
        t = time.time()
        
        # Shifting red and blue channels independently
        shift_x = int(math.sin(t * 5) * 10)
        shift_y = int(math.cos(t * 5) * 10)
        
        b, g, r = cv2.split(frame)
        
        M_r = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        M_b = np.float32([[1, 0, -shift_x], [0, 1, -shift_y]])
        
        r_shifted = cv2.warpAffine(r, M_r, (w, h))
        b_shifted = cv2.warpAffine(b, M_b, (w, h))
        
        return cv2.merge([b_shifted, g, r_shifted])

    def thermal_vision(self, frame):
        # Convert brightness into thermal heat map
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    def glitch_mode(self, frame):
        h, w = frame.shape[:2]
        out = frame.copy()
        
        # Horizontal scanline distortions
        for _ in range(5):
            y1 = random.randint(0, h - 20)
            y2 = y1 + random.randint(5, 20)
            shift = random.randint(-20, 20)
            
            if shift > 0:
                out[y1:y2, shift:] = out[y1:y2, :-shift]
            elif shift < 0:
                out[y1:y2, :shift] = out[y1:y2, -shift:]
                
        # Random RGB channel offsets
        if random.random() > 0.8:
            b, g, r = cv2.split(out)
            r_shift = np.roll(r, random.randint(-10, 10), axis=0)
            b_shift = np.roll(b, random.randint(-10, 10), axis=1)
            out = cv2.merge([b_shift, g, r_shift])
            
        return out

    def kaleidoscope(self, frame):
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        
        # Grab top-left quadrant
        q1 = frame[0:cy, 0:cx]
        
        # Create symmetric quadrants
        q2 = cv2.flip(q1, 1)  # mirror horizontally
        q3 = cv2.flip(q1, 0)  # mirror vertically
        q4 = cv2.flip(q1, -1) # mirror both
        
        out = np.zeros_like(frame)
        out[0:cy, 0:cx] = q1
        out[0:cy, cx:cx+q1.shape[1]] = q2
        out[cy:cy+q1.shape[0], 0:cx] = q3
        out[cy:cy+q1.shape[0], cx:cx+q1.shape[1]] = q4
        
        return out

    def fire_trail(self, frame, hand_lm):
        out = frame.copy()
        if hand_lm and self.particle_system:
            x, y = hand_lm[8] # index fingertip
            
            # Spawn animated fire-like particles
            self.particle_system.spawn(
                x=x, y=y, n=8,
                base_color=(0, 140, 255), # Orange
                speed=3.0, life_ms=400, size=8.0,
                spread=math.pi * 2, gravity=-0.5, drag=0.9
            )
            
        if self.particle_system:
            t = now_ms()
            self.particle_system.update(t)
            self.particle_system.render(out, t)
            
        return out

    def hologram(self, frame):
        h, w = frame.shape[:2]
        
        # Cyan-blue holographic tint
        tint = np.full_like(frame, (255, 150, 0)) # BGR
        hologram = cv2.addWeighted(frame, 0.6, tint, 0.4, 0)
        
        # Scanlines
        for y in range(0, h, 4):
            hologram[y:y+2, :] = (hologram[y:y+2, :] * 0.7).astype(np.uint8)
            
        # Flickering
        if random.random() > 0.9:
            hologram = cv2.addWeighted(hologram, 0.8, np.zeros_like(frame), 0.2, 0)
            
        # Futuristic HUD-style visual effects
        cv2.rectangle(hologram, (20, 20), (w - 20, h - 20), (255, 200, 0), 2)
        cv2.circle(hologram, (w // 2, h // 2), 50, (255, 200, 0), 1)
        cv2.line(hologram, (0, h // 2), (w, h // 2), (255, 200, 0), 1)
        cv2.line(hologram, (w // 2, 0), (w // 2, h), (255, 200, 0), 1)
        
        return hologram