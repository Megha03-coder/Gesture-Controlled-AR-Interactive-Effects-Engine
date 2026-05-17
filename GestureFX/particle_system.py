from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from utils import clamp, now_ms


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    color: Tuple[int, int, int]
    life_ms: int
    born_ms: int
    size: float
    drag: float = 0.98
    gravity: float = 0.0

    def age_ms(self, t_ms: int) -> int:
        return max(0, t_ms - self.born_ms)


class ParticleSystem:
    def __init__(self, max_particles: int = 1500):
        self.max_particles = max_particles
        self.particles: List[Particle] = []

    def clear(self) -> None:
        self.particles.clear()

    def spawn(
        self,
        x: float,
        y: float,
        n: int,
        base_color: Tuple[int, int, int],
        speed: float,
        life_ms: int,
        size: float,
        spread: float = np.pi * 2,
        drag: float = 0.98,
        gravity: float = 0.0,
    ) -> None:
        import random

        if len(self.particles) >= self.max_particles:
            return

        for _ in range(n):
            ang = random.uniform(0, spread)
            vel = speed * random.uniform(0.5, 1.0)
            vx = vel * np.cos(ang)
            vy = vel * np.sin(ang)
            c = (
                int(clamp(base_color[0] + random.randint(-10, 10), 0, 255)),
                int(clamp(base_color[1] + random.randint(-10, 10), 0, 255)),
                int(clamp(base_color[2] + random.randint(-10, 10), 0, 255)),
            )
            self.particles.append(
                Particle(
                    x=x,
                    y=y,
                    vx=vx,
                    vy=vy,
                    color=c,
                    life_ms=life_ms,
                    born_ms=now_ms(),
                    size=size * random.uniform(0.6, 1.3),
                    drag=drag,
                    gravity=gravity,
                )
            )
            if len(self.particles) >= self.max_particles:
                break

    def update(self, t_ms: int) -> None:
        alive: List[Particle] = []
        for p in self.particles:
            age = p.age_ms(t_ms)
            if age >= p.life_ms:
                continue
            # integrate
            p.vx *= p.drag
            p.vy = p.vy * p.drag + p.gravity
            p.x += p.vx
            p.y += p.vy
            alive.append(p)
        self.particles = alive

    def render(self, img: np.ndarray, t_ms: int, alpha_curve: str = "fade") -> None:
        for p in self.particles:
            age = p.age_ms(t_ms)
            if alpha_curve == "fade":
                a = 1.0 - age / max(1, p.life_ms)
            else:
                a = 1.0
            if a <= 0:
                continue
            # Draw as circles
            r = max(1, int(p.size * (0.5 + a)))
            color = (
                int(p.color[0]),
                int(p.color[1]),
                int(p.color[2]),
            )
            # alpha via thickness trick: multiple draws
            cv2.circle(img, (int(p.x), int(p.y)), r, color, -1, lineType=cv2.LINE_AA)


