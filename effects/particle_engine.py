import cv2
import numpy as np
import random
import math

class Particle:
    def __init__(self, x, y, vx, vy, life_ms, color, size, gravity=0.0, drag=1.0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life_ms = life_ms
        self.max_life_ms = life_ms
        self.color = color
        self.size = size
        self.gravity = gravity
        self.drag = drag
        self.spawn_time = cv2.getTickCount() / cv2.getTickFrequency() * 1000

    def is_alive(self, current_time_ms):
        return (current_time_ms - self.spawn_time) < self.max_life_ms

class ParticleEngine:
    def __init__(self, max_particles=1000):
        self.particles = []
        self.max_particles = max_particles
        self.freeze_buffer = None
        self.freeze_start_time = 0
        self.freeze_duration = 1500
        
        # Virtual Painter State
        self.canvas = None
        self.prev_x, self.prev_y = 0, 0
        self.palette = [
            (255, 0, 255),  # Magenta
            (0, 255, 0),    # Green
            (0, 255, 255),  # Yellow
            (0, 165, 255),  # Orange
            (255, 0, 0),    # Blue
            (255, 255, 255) # White
        ]
        self.color_idx = 0
        self.brush_color = self.palette[self.color_idx]
        self.brush_thickness = 8
        self.last_color_change_time = 0

    def spawn(self, x, y, n, base_color, speed, life_ms, size, spread=math.pi*2, gravity=0.0, drag=1.0):
        for _ in range(n):
            if len(self.particles) >= self.max_particles:
                self.particles.pop(0)
                
            angle = random.uniform(0, spread)
            s = random.uniform(speed * 0.5, speed * 1.5)
            vx = math.cos(angle) * s
            vy = math.sin(angle) * s
            
            # Add subtle color variance to make it look natural
            color = (
                max(0, min(255, base_color[0] + random.randint(-20, 20))),
                max(0, min(255, base_color[1] + random.randint(-20, 20))),
                max(0, min(255, base_color[2] + random.randint(-20, 20)))
            )
            
            self.particles.append(Particle(x, y, vx, vy, life_ms, color, size, gravity, drag))

    def update(self):
        current_time = cv2.getTickCount() / cv2.getTickFrequency() * 1000
        alive_particles = []
        
        for p in self.particles:
            if p.is_alive(current_time):
                p.vx *= p.drag
                p.vy *= p.drag
                p.vy += p.gravity
                p.x += p.vx
                p.y += p.vy
                alive_particles.append(p)
                
        self.particles = alive_particles
        return current_time

    def render(self, frame: np.ndarray, active_gesture: str, hands: list, active_mode: str = "idle") -> np.ndarray:
        current_time_render = cv2.getTickCount() / cv2.getTickFrequency() * 1000
        t_sec = current_time_render / 1000.0
        
        # Initialize or resize the persistent drawing canvas
        h_frame, w_frame = frame.shape[:2]
        if self.canvas is None or self.canvas.shape[:2] != (h_frame, w_frame):
            self.canvas = np.zeros((h_frame, w_frame, 3), dtype=np.uint8)

        # --- VIRTUAL PAINTER MODE ---
        if active_mode == "draw":
            if hands and active_gesture not in ["UNKNOWN PATTERN", "NO TARGET DETECTED", "SCANNING..."]:
                lm = hands[0].lm_list
                x, y = lm[8][0], lm[8][1]  # Index finger tip
                
                # 1 Finger Up = Draw (Mapped to FIRE MODE gesture)
                if active_gesture == "FIRE MODE":
                    if self.prev_x == 0 and self.prev_y == 0:
                        self.prev_x, self.prev_y = x, y
                    
                    cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y), self.brush_color, self.brush_thickness, cv2.LINE_AA)
                    self.prev_x, self.prev_y = x, y
                    
                    # Draw hover cursor
                    cv2.circle(frame, (x, y), self.brush_thickness, (0, 255, 255), -1)
                    cv2.circle(frame, (x, y), int(self.brush_thickness/2), self.brush_color, -1)
                
                # Thumbs Up = Change Color
                elif active_gesture == "THUMBS UP":
                    self.prev_x, self.prev_y = 0, 0
                    if current_time_render - self.last_color_change_time > 1000:  # 1 second cooldown
                        self.color_idx = (self.color_idx + 1) % len(self.palette)
                        self.brush_color = self.palette[self.color_idx]
                        self.last_color_change_time = current_time_render
                        
                    cv2.putText(frame, "COLOR SWAPPED", (w_frame//2 - 140, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, self.brush_color, 2, cv2.LINE_AA)
                    cv2.circle(frame, (x, y), self.brush_thickness, (255, 255, 255), -1)
                    cv2.circle(frame, (x, y), int(self.brush_thickness/2), self.brush_color, -1)
                else:
                    self.prev_x, self.prev_y = 0, 0
                    
                    # Open Palm = Clear Canvas
                    if active_gesture == "OPEN PALM (RGB)":
                        self.canvas = np.zeros((h_frame, w_frame, 3), dtype=np.uint8)
                        cv2.putText(frame, "CANVAS CLEARED", (w_frame//2 - 120, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                self.prev_x, self.prev_y = 0, 0
                
            # Blend canvas over the frame completely seamlessly
            gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            frame_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
            return cv2.add(frame_bg, frame_fg)
        
        # Handle Freeze Frame (Slow Motion) Logic
        if active_gesture == "CLOSED FIST (FREEZE)":
            if self.freeze_buffer is None:
                self.freeze_buffer = frame.copy()
                self.freeze_start_time = current_time_render
                
            age_ms = current_time_render - self.freeze_start_time
            a = max(0.0, min(1.0, age_ms / self.freeze_duration))
            scale = 1.0 + 0.05 * (1.0 - a)  # Zoom out slowly
            h, w = self.freeze_buffer.shape[:2]
            
            M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
            frame = cv2.warpAffine(self.freeze_buffer, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            
            # Apply cinematic scanlines
            for yy in range(0, h, 4):
                alpha = 0.15 if (yy // 4) % 2 == 0 else 0.05
                frame[yy:yy+1, :] = (frame[yy:yy+1, :] * (1 - alpha)).astype(np.uint8)
                
            # Apply cold blue tint
            tint = np.full_like(frame, (255, 150, 0))  # BGR for light blue
            frame = cv2.addWeighted(frame, 0.8, tint, 0.2, 0)
            
            # Spawn ambient snow/ice particles from the top
            self.spawn(random.randint(0, w), 0, n=2, base_color=(255, 255, 255), 
                       speed=2.0, life_ms=1500, size=3.0, spread=math.pi, gravity=0.1, drag=0.98)
        else:
            self.freeze_buffer = None

        # 1. Spawn new particles based on active gesture
        if hands and active_gesture not in ["UNKNOWN PATTERN", "NO TARGET DETECTED", "SCANNING...", "CLOSED FIST (FREEZE)"]:
            lm = hands[0].lm_list
            
            if active_gesture == "FIRE MODE":
                # Orange/Red fire at index tip (Node 8)
                x, y = lm[8][0], lm[8][1]

                # Add a glowing aura around the finger tip
                glow_overlay = np.zeros_like(frame, dtype=np.uint8)
                pulse = 0.8 + 0.2 * math.sin(t_sec * 15)
                cv2.circle(glow_overlay, (x, y), int(40 * pulse), (0, 100, 255), -1) # Orange glow
                cv2.circle(glow_overlay, (x, y), int(20 * pulse), (150, 200, 255), -1) # Yellowish core
                frame = cv2.addWeighted(frame, 1.0, glow_overlay, 0.4, 0) # Blend the glow onto the frame

                # Spawn more intense orange/red particles
                self.spawn(x, y, n=20, base_color=(0, 100, 255), # More particles
                           speed=4.5, life_ms=500, size=8.0, gravity=-0.6, drag=0.9) # Faster, larger, longer life
                # Add yellow core particles for more fiery look
                self.spawn(x, y, n=15, base_color=(0, 200, 255), # Yellow, more particles
                           speed=4.0, life_ms=400, size=6.0, gravity=-0.7, drag=0.9) # Faster, larger, longer life
                # Add some red embers
                self.spawn(x, y, n=8, base_color=(0, 0, 255), # Red
                           speed=3.0, life_ms=600, size=4.0, gravity=-0.4, drag=0.95) # Slower, smaller, longer life
            elif active_gesture == "LIGHTNING / SPARK":
                # Blue sparks at index and middle fingers (Nodes 8 & 12)
                for tip in [8, 12]:
                    self.spawn(lm[tip][0], lm[tip][1], n=3, base_color=(255, 200, 50), 
                               speed=6.0, life_ms=300, size=3.0, gravity=0.1, drag=0.9)
            elif active_gesture == "PINKY UP (SMOKE)":
                # Gray smoke rising slowly from pinky tip (Node 20)
                self.spawn(lm[20][0], lm[20][1], n=2, base_color=(150, 150, 150), 
                           speed=1.5, life_ms=800, size=8.0, gravity=-0.2, drag=0.95)
            elif active_gesture == "OPEN PALM (RGB)":
                # Full frame RGB distortion (chromatic aberration)
                t = cv2.getTickCount() / cv2.getTickFrequency()
                shift_x = int(math.sin(t * 15) * 15)
                shift_y = int(math.cos(t * 15) * 15)
                
                b, g, r = cv2.split(frame)
                rows, cols = frame.shape[:2]
                
                M_r = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                M_b = np.float32([[1, 0, -shift_x], [0, 1, -shift_y]])
                
                r_shifted = cv2.warpAffine(r, M_r, (cols, rows))
                b_shifted = cv2.warpAffine(b, M_b, (cols, rows))
                
                frame[:] = cv2.merge([b_shifted, g, r_shifted])
                
                # Ambient rainbow particles from the palm (Node 9)
                self.spawn(lm[9][0], lm[9][1], n=2, 
                           base_color=(random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)), 
                           speed=3.0, life_ms=500, size=6.0, gravity=0.1, drag=0.9)
            elif active_gesture == "MAGIC CIRCLE":
                # Doctor Strange style Magic Circle
                t = cv2.getTickCount() / cv2.getTickFrequency()
                
                # Calculate the center of the palm using nodes 0, 5, 17
                cx = int((lm[0][0] + lm[5][0] + lm[17][0]) / 3)
                cy = int((lm[0][1] + lm[5][1] + lm[17][1]) / 3)
                
                radius = 90
                angle = t * 90  # Degrees per second
                
                color = (50, 170, 255)  # Glowing orange in BGR
                mandala_overlay = frame.copy()
                
                # Draw rotating outer circles and squares
                cv2.circle(mandala_overlay, (cx, cy), radius, color, 2, cv2.LINE_AA)
                cv2.circle(mandala_overlay, (cx, cy), radius - 15, color, 1, cv2.LINE_AA)
                for offset in [0, 45]:
                    rect = ((cx, cy), (radius * 1.4, radius * 1.4), angle + offset)
                    box = cv2.boxPoints(rect)
                    cv2.polylines(mandala_overlay, [np.int32(box)], True, color, 2, cv2.LINE_AA)
                    
                # Blend the magic circle (glow effect) directly into the frame buffer
                frame[:] = cv2.addWeighted(mandala_overlay, 0.8, frame, 0.4, 0)
                
                # Ambient sparks falling from the center
                self.spawn(cx, cy, n=4, base_color=(0, 200, 255),
                           speed=3.0, life_ms=300, size=2.0, spread=math.pi*2, drag=0.9, gravity=0.1)
            elif active_gesture == "THUMBS UP":
                # Success animation: Green rings and particles at thumb tip (Node 4)
                x, y = lm[4][0], lm[4][1]
                t = cv2.getTickCount() / cv2.getTickFrequency()
                
                # Expanding green rings
                for i in range(3):
                    r = int(18 + i * 14 + 10 * (0.5 + 0.5 * math.sin(t * 2.5 - i)))
                    cv2.circle(frame, (x, y), max(1, r), (0, 255, 0), 2, cv2.LINE_AA)
                    
                # Spawn green particles
                self.spawn(x, y, n=4, base_color=(0, 255, 0), 
                           speed=3.8, life_ms=520, size=5.0, spread=math.pi*2, drag=0.95, gravity=0.06)
                
                # Slight green overlay tint
                tint = np.full_like(frame, (0, 80, 0))
                frame[:] = cv2.addWeighted(frame, 0.88, tint, 0.12, 0)
            elif active_gesture == "ROCK SIGN":
                # Rock music fire explosion from the palm
                cx = int((lm[0][0] + lm[5][0] + lm[17][0]) / 3)
                cy = int((lm[0][1] + lm[5][1] + lm[17][1]) / 3)
                t = cv2.getTickCount() / cv2.getTickFrequency()
                
                # Explosive fire particles
                self.spawn(cx, cy, n=8, base_color=(0, 140, 255), # Orange
                           speed=8.0, life_ms=350, size=8.0, spread=math.pi*2, drag=0.85, gravity=0)
                self.spawn(cx, cy, n=5, base_color=(150, 255, 255), # Yellow
                           speed=12.0, life_ms=250, size=5.0, spread=math.pi*2, drag=0.8, gravity=0)
                
                # Explosive shockwave ring
                r = int((t * 400) % 150)
                alpha_ring = max(0.0, 1.0 - (r / 150.0))
                cv2.circle(frame, (cx, cy), max(1, r), (0, 100, 255), max(1, int(15 * alpha_ring)), cv2.LINE_AA)
                
                # Intense screen flash tint
                flash_alpha = 0.3 * alpha_ring
                tint = np.full_like(frame, (0, 50, 255)) # BGR for Orange
                frame[:] = cv2.addWeighted(frame, 1.0 - flash_alpha, tint, flash_alpha, 0)
            elif active_gesture == "DAZZLING":
                # Dazzling star-like particles from 3 finger tips
                for tip in [8, 12, 16]: # Index, Middle, Ring
                    x, y = lm[tip][0], lm[tip][1]
                    
                    # Spawn bright white/yellow "star" particles
                    self.spawn(x, y, n=5, base_color=(220, 255, 255), # Bright yellow
                               speed=5.0, life_ms=400, size=5.0, spread=math.pi*2, drag=0.9, gravity=0.05)
                    self.spawn(x, y, n=5, base_color=(255, 255, 255), # Bright white
                               speed=4.0, life_ms=350, size=3.0, spread=math.pi*2, drag=0.9, gravity=0.05)
                
                # Add a random lens flare effect for extra "dazzle"
                if random.random() < 0.08: # 8% chance per frame
                    lx, ly = random.randint(0, w_frame), random.randint(0, h_frame)
                    l_radius = random.randint(50, 150)
                    l_color = (200, 255, 255) # Yellowish-white
                    
                    flare_overlay = frame.copy()
                    cv2.circle(flare_overlay, (lx, ly), l_radius, l_color, -1)
                    cv2.line(flare_overlay, (lx - l_radius*2, ly), (lx + l_radius*2, ly), l_color, 2)
                    cv2.line(flare_overlay, (lx, ly - l_radius*2), (lx, ly + l_radius*2), l_color, 2)
                    
                    frame[:] = cv2.addWeighted(flare_overlay, 0.2, frame, 0.8, 0)
            elif active_gesture == "PRISM":
                # Prism effect emanating from the palm
                cx = int((lm[0][0] + lm[5][0] + lm[17][0]) / 3)
                cy = int((lm[0][1] + lm[5][1] + lm[17][1]) / 3)
                t = cv2.getTickCount() / cv2.getTickFrequency()

                # Draw a rotating prism shape (triangle)
                prism_overlay = frame.copy()
                radius = 50
                angle = t * 120
                
                pts = []
                for i in range(3):
                    px = cx + int(radius * math.cos(math.radians(angle + i * 120)))
                    py = cy + int(radius * math.sin(math.radians(angle + i * 120)))
                    pts.append([px, py])
                
                cv2.polylines(prism_overlay, [np.array(pts)], True, (255, 255, 255), 2, cv2.LINE_AA)
                frame[:] = cv2.addWeighted(prism_overlay, 0.6, frame, 0.4, 0)

                # Spawn rainbow particles by cycling through hue
                hue = int(t * 150) % 180 # OpenCV HSV hue is 0-179
                color_hsv = np.uint8([[[hue, 255, 255]]])
                color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
                
                self.spawn(cx, cy, n=10, 
                           base_color=(int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])),
                           speed=6.0, life_ms=700, size=5.0, spread=math.pi, gravity=0, drag=0.95)

        # 2. Update physics
        current_time = self.update()
        if not self.particles:
            return frame

        # 3. Draw particles to an overlay and blend for a glowing effect
        overlay = frame.copy()
        for p in self.particles:
            alpha = max(0.0, 1.0 - ((current_time - p.spawn_time) / p.max_life_ms))
            current_size = max(1, int(p.size * alpha))
            cv2.circle(overlay, (int(p.x), int(p.y)), current_size, p.color, -1)

        return cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)