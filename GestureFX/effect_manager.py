from __future__ import annotations

import math
import random
from typing import Optional, Tuple

import cv2
import numpy as np

from overlay_manager import OverlayManager
from particle_system import ParticleSystem
from utils import blur_region, circle_falloff, clamp, draw_text, now_ms


class EffectManager:
    def __init__(self):
        self.overlay = OverlayManager()
        self.particles = ParticleSystem(max_particles=1600)

        self.active_effect = "none"
        self._effect_since_ms = now_ms()

        self._freeze_buffer: Optional[np.ndarray] = None
        self._freeze_start_ms: Optional[int] = None
        self._freeze_duration_ms = 1200

        # For smooth transitions
        self._last_finger_bits = None

        # Victory-vs-two-fingers heuristic
        self._victory_threshold = 0.30  # bbox aspect ratio

    def trigger(self, effect_id: str, hand_lm: Optional[list] = None, bbox=None, frame=None) -> None:
        if effect_id == self.active_effect:
            return
        self.active_effect = effect_id
        self._effect_since_ms = now_ms()
        self.particles.clear()

        if effect_id == "freeze" and frame is not None:
            self._freeze_buffer = frame.copy()
            self._freeze_start_ms = now_ms()

    def render(self, frame_bgr: np.ndarray, hand_bits, hand_state=None) -> np.ndarray:
        t_ms = now_ms()
        t = t_ms / 1000.0

        if self.active_effect == "none" or hand_state is None:
            return frame_bgr

        # Closed fist freeze
        if self.active_effect == "freeze":
            if self._freeze_buffer is not None and self._freeze_start_ms is not None:
                age = t_ms - self._freeze_start_ms
                if age < self._freeze_duration_ms:
                    return self._apply_slow_motion_style(self._freeze_buffer, age)
            # end freeze
            self.active_effect = "none"
            self._freeze_buffer = None
            self._freeze_start_ms = None
            return frame_bgr

        out = frame_bgr.copy()
        x1, y1 = hand_state.lm_list[8]  # index finger tip
        # For thumb tip
        x2, y2 = hand_state.lm_list[4]

        # One finger -> fire
        if self.active_effect == "fire":
            self._render_fire(out, (x1, y1), t)

        elif self.active_effect == "lightning":
            self._render_lightning(out, hand_state, t)

        elif self.active_effect == "spark":
            self._render_spark(out, hand_state, t)

        elif self.active_effect == "rgb_palm":
            strength = 0.65
            out = self.overlay.apply_rgb_filter(out, strength=strength, t=t)
            out = self.overlay.apply_rainbow_overlay(out, t=t, strength=0.38)
            out = self._apply_background_blur(out, hand_state)

        elif self.active_effect == "success":
            out = self._render_success(out, hand_state, t)

        elif self.active_effect == "neon_particles":
            self._render_neon_particles(out, hand_state, t)

        elif self.active_effect == "water":
            self._render_water(out, hand_state, t)

        elif self.active_effect == "sepia_filter":
            out = self.overlay.apply_sepia_filter(out)

        elif self.active_effect == "portal":
            self._render_portal(out, hand_state, t)

        elif self.active_effect == "magic_circle":
            self._render_magic_circle(out, hand_state, t)

        elif self.active_effect == "smoke":
            self._render_smoke(out, hand_state, t)

        return out

    def maybe_override_victory(self, detected_effect_id: str, hand_state) -> str:
        """Disambiguate [0,1,1,0,0] between spark (two fingers) and victory (neon).

        Heuristic: if hand bbox is tall-ish vs wide-ish, pick spark; otherwise neon.
        """
        if detected_effect_id != "spark":
            return detected_effect_id
        if not hand_state or not hand_state.bbox:
            return detected_effect_id
        x, y, w, h = hand_state.bbox
        aspect = h / max(1, w)
        # If aspect ratio is high (hand taller), interpret as spark.
        if aspect >= (1.0 + self._victory_threshold):
            return "spark"
        return "neon_particles"

    def _apply_slow_motion_style(self, frozen: np.ndarray, age_ms: int) -> np.ndarray:
        # Add a subtle scanline + zoom pulse.
        a = clamp(age_ms / self._freeze_duration_ms, 0.0, 1.0)
        scale = 1.0 + 0.02 * (1.0 - a)
        h, w = frozen.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
        out = cv2.warpAffine(frozen, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        # scanlines
        lines = out.copy()
        for yy in range(0, h, 4):
            alpha = 0.12 if (yy // 4) % 2 == 0 else 0.05
            lines[yy : yy + 1, :] = (lines[yy : yy + 1, :] * (1 - alpha)).astype(np.uint8)
        return lines

    def _render_fire(self, out: np.ndarray, tip: Tuple[int, int], t: float) -> None:
        x, y = tip

        # Orange glow aura with finger-following pulse
        overlay = out.copy()
        pulse = 0.8 + 0.2 * math.sin(t * 15)
        cv2.circle(overlay, (x, y), int(40 * pulse), (0, 100, 255), -1)
        cv2.circle(overlay, (x, y), int(20 * pulse), (150, 200, 255), -1)
        out[:] = cv2.addWeighted(overlay, 0.4, out, 0.6, 0)

        # Spawn dense particles that fly upwards to look like animated fire
        self.particles.spawn(
            x=x, y=y, n=15,
            base_color=(150, 255, 255),  # Pale yellow core
            speed=2.0, life_ms=300, size=5.0,
            spread=math.pi * 2, gravity=-0.4, drag=0.9
        )
        self.particles.spawn(
            x=x, y=y, n=20,
            base_color=(0, 140, 255),    # Orange mid-flame
            speed=3.0, life_ms=500, size=7.0,
            spread=math.pi * 2, gravity=-0.6, drag=0.92
        )
        self.particles.spawn(
            x=x, y=y, n=15,
            base_color=(0, 0, 255),      # Red outer flame
            speed=4.0, life_ms=700, size=9.0,
            spread=math.pi * 2, gravity=-0.8, drag=0.95
        )

        self.particles.update(now_ms())
        self.particles.render(out, now_ms())

    def _render_lightning(self, out: np.ndarray, hand_state, t: float) -> None:
        # Draw electric polyline from wrist(0) to thumb tip(4)
        pts = hand_state.lm_list
        start = pts[0]
        end = pts[4]
        # intermediate jitter points
        steps = 6
        xs = np.linspace(start[0], end[0], steps)
        ys = np.linspace(start[1], end[1], steps)
        poly = []
        for i in range(steps):
            dx = random.uniform(-15, 15) * (1 - i / steps)
            dy = random.uniform(-15, 15) * (1 - i / steps)
            poly.append((int(xs[i] + dx), int(ys[i] + dy)))

        # Animated electric lines with a glow pulse
        color1 = (255, 255, 255)  # White core
        color2 = (255, 200, 0)    # Blue/cyan edge glow (BGR format)
        pulse = math.sin(t * 20) * 0.5 + 0.5
        glow_thickness = int(3 + 5 * pulse)

        for _ in range(2):
            cv2.polylines(out, [np.array(poly, np.int32)], False, color1, 2, lineType=cv2.LINE_AA)
            cv2.polylines(out, [np.array(poly, np.int32)], False, color2, glow_thickness, lineType=cv2.LINE_AA)

        # Blue sparks
        self.particles.spawn(
            x=end[0],
            y=end[1],
            n=40,
            base_color=(255, 200, 50),
            speed=6.0,
            life_ms=420,
            size=4.0,
            spread=math.pi * 2,
            drag=0.92,
            gravity=0.05,
        )
        self.particles.update(now_ms())
        self.particles.render(out, now_ms())

    def _render_spark(self, out: np.ndarray, hand_state, t: float) -> None:
        # Spawns bright white/yellow sparks from index and middle fingers
        for tip in [8, 12]:
            if tip < len(hand_state.lm_list):
                x, y = hand_state.lm_list[tip]
                self.particles.spawn(
                    x=x, y=y, n=10,
                    base_color=(255, 255, 200),
                    speed=6.0,
                    life_ms=350,
                    size=3.0,
                    spread=math.pi * 2,
                    gravity=0.2,
                    drag=0.9
                )
        self.particles.update(now_ms())
        self.particles.render(out, now_ms())

    def _render_neon_particles(self, out: np.ndarray, hand_state, t: float) -> None:
        # Neon edge glow from fingertips
        tips = [8, 12]
        for tip_id in tips:
            x, y = hand_state.lm_list[tip_id]
            # outline glow effect by drawing multiple circles
            hue_shift = 0.5 + 0.5 * math.sin(t * 3.0)
            c1 = (int(255 * hue_shift), 255, int(255 * (1 - hue_shift)))
            c2 = (255, int(255 * (1 - hue_shift)), int(255 * hue_shift))
            cv2.circle(out, (x, y), 18, c1, 2, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 30, c2, 2, lineType=cv2.LINE_AA)

            self.particles.spawn(
                x=x,
                y=y,
                n=22,
                base_color=(255, 0, 255),
                speed=4.5,
                life_ms=580,
                size=5.5,
                spread=math.pi * 2,
                drag=0.96,
                gravity=0.0,
            )

        self.particles.update(now_ms())
        self.particles.render(out, now_ms())

        # Blur glow layer
        glow = cv2.GaussianBlur(out, (0, 0), 6)
        out = cv2.addWeighted(out, 0.85, glow, 0.25, 0)

    def _apply_background_blur(self, out: np.ndarray, hand_state) -> np.ndarray:
        # Create a soft mask roughly around hand bbox
        if not hand_state.bbox:
            return out
        x, y, w, h = hand_state.bbox
        h_img, w_img = out.shape[:2]
        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        pad = int(max(w, h) * 0.7)
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w_img - 1, x + w + pad)
        y1 = min(h_img - 1, y + h + pad)
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)
        mask = cv2.GaussianBlur(mask, (0, 0), 15)
        # blur where hand is (so background stays sharper)
        # We'll instead blur outside by inverting: user asked background blur.
        inv = cv2.bitwise_not(mask)
        blurred = blur_region(out, inv, k=35)
        return blurred

    def _render_success(self, out: np.ndarray, hand_state, t: float) -> np.ndarray:
        # Big expanding ring at thumb tip
        x, y = hand_state.lm_list[4]
        for i in range(3):
            r = int(18 + i * 14 + 10 * (0.5 + 0.5 * math.sin(t * 2.5 - i)))
            cv2.circle(out, (x, y), r, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        # spawn green particles
        self.particles.spawn(
            x=x,
            y=y,
            n=18,
            base_color=(0, 255, 0),
            speed=3.8,
            life_ms=520,
            size=5.0,
            spread=math.pi * 2,
            drag=0.95,
            gravity=0.06,
        )
        self.particles.update(now_ms())
        self.particles.render(out, now_ms())

        # slight overlay tint
        tint = np.full_like(out, (0, 80, 0))
        out = cv2.addWeighted(out, 0.88, tint, 0.12, 0)
        return out

    def _render_water(self, out: np.ndarray, hand_state, t: float) -> None:
        # Spawn falling blue water particles from the middle finger tip
        x, y = hand_state.lm_list[12]

        self.particles.spawn(
            x=x,
            y=y,
            n=18,
            base_color=(255, 200, 50),  # Light blue in BGR format
            speed=3.5,
            life_ms=650,
            size=5.5,
            spread=math.pi * 2,
            drag=0.96,
            gravity=0.3,  # Makes particles fall downwards like water
        )
        self.particles.update(now_ms())
        self.particles.render(out, now_ms())

        # Ripple rings simulating water splashes
        r1 = int((t * 60) % 60)
        r2 = int((t * 60 + 30) % 60)
        cv2.circle(out, (x, y), max(1, r1), (255, 180, 50), 2, lineType=cv2.LINE_AA)
        cv2.circle(out, (x, y), max(1, r2), (255, 180, 50), 2, lineType=cv2.LINE_AA)

    def _render_magic_circle(self, out: np.ndarray, hand_state, t: float) -> None:
        pts = hand_state.lm_list
        # Calculate the center of the palm
        cx = (pts[0][0] + pts[5][0] + pts[17][0]) // 3
        cy = (pts[0][1] + pts[5][1] + pts[17][1]) // 3

        radius = 90
        angle = t * 90  # Degrees per second
        
        overlay = out.copy()
        color = (50, 170, 255)  # Glowing orange in BGR
        
        # Draw rotating outer circles
        cv2.circle(overlay, (cx, cy), radius, color, 2, cv2.LINE_AA)
        cv2.circle(overlay, (cx, cy), radius - 15, color, 1, cv2.LINE_AA)
        
        # Draw inner rotating squares to simulate a geometric spell mandala
        for offset in [0, 45]:
            rect = ((cx, cy), (radius * 1.4, radius * 1.4), angle + offset)
            box = cv2.boxPoints(rect)
            cv2.polylines(overlay, [np.int32(box)], True, color, 2, cv2.LINE_AA)
            
        # Blend the magic circle (glow effect)
        out[:] = cv2.addWeighted(overlay, 0.8, out, 0.4, 0)
        
        # Add ambient sparks falling from the circle
        self.particles.spawn(
            x=cx, y=cy, n=4, base_color=(0, 200, 255),
            speed=3.0, life_ms=300, size=2.0, spread=math.pi*2, drag=0.9, gravity=0.1
        )
        self.particles.update(now_ms())
        self.particles.render(out, now_ms())

    def _render_portal(self, out: np.ndarray, hand_state, t: float) -> None:
        pts = hand_state.lm_list
        cx, cy = pts[0]
        cy -= 150  # Floating portal above the wrist
        radius = 110
        
        # Inner black hole for the void effect
        overlay = out.copy()
        cv2.circle(overlay, (cx, cy), radius - 10, (0, 0, 0), -1)
        out[:] = cv2.addWeighted(overlay, 0.7, out, 0.3, 0)

        # Circular spark emitters traveling around the perimeter
        for offset in [0, math.pi / 2, math.pi, 3 * math.pi / 2]:
            spawn_angle = t * 8 + offset
            px = cx + int(math.cos(spawn_angle) * radius)
            py = cy + int(math.sin(spawn_angle) * radius)
            self.particles.spawn(
                x=px, y=py, n=6, base_color=(0, 120, 255), # Orange sparks
                speed=1.5, life_ms=400, size=4.0, spread=math.pi*2, drag=0.9, gravity=0.05
            )
        self.particles.update(now_ms())
        self.particles.render(out, now_ms())

    def _render_smoke(self, out: np.ndarray, hand_state, t: float) -> None:
        # Emit rising gray smoke from all fingertips
        for tip in [4, 8, 12, 16, 20]:
            if tip < len(hand_state.lm_list):
                x, y = hand_state.lm_list[tip]
                color = random.choice([(150, 150, 150), (100, 100, 100), (200, 200, 200)])
                self.particles.spawn(
                    x=x, y=y, n=3, base_color=color, speed=1.5, life_ms=1200, 
                    size=random.uniform(8.0, 15.0), spread=math.pi, drag=0.92, gravity=-0.2
                )
        self.particles.update(now_ms())
        self.particles.render(out, now_ms())
