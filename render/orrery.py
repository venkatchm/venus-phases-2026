# NOTE: no `from __future__ import annotations` -- PEP 563 would turn the Taichi
# kernel argument annotations below into strings, which Taichi rejects.
"""Two schematic panels that explain the geometry behind the sky view.

Left: the inner solar system from ecliptic north, with the Sun-Earth-Venus
triangle that fixes Venus' elongation and phase. Right: the Earth-Moon system
to scale along the line of sight to Venus, which is the actual reason an
occultation is a local event -- the Moon is only 0.5 deg wide and 60 Earth radii
away, so which part of the Earth you stand on decides whether it covers Venus.
"""
import math

import numpy as np
import taichi as ti

DEG = math.pi / 180.0


@ti.data_oriented
class OrreryRenderer:
    MAX_ORBIT = 2048

    def __init__(self, width=1280, height=720):
        self.width, self.height = width, height
        self.pixels = ti.Vector.field(3, ti.f32, shape=(width, height))

        self.orbit_pts = ti.Vector.field(2, ti.f32, shape=self.MAX_ORBIT)
        self.orbit_col = ti.Vector.field(3, ti.f32, shape=self.MAX_ORBIT)
        self.n_orbit = ti.field(ti.i32, shape=())

        self.body_pos = ti.Vector.field(2, ti.f32, shape=16)     # AU, ecliptic x/y
        self.body_col = ti.Vector.field(3, ti.f32, shape=16)
        self.body_rad = ti.field(ti.f32, shape=16)               # pixels
        self.n_body = ti.field(ti.i32, shape=())

        self.seg_a = ti.Vector.field(2, ti.f32, shape=16)
        self.seg_b = ti.Vector.field(2, ti.f32, shape=16)
        self.seg_col = ti.Vector.field(3, ti.f32, shape=16)
        self.n_seg = ti.field(ti.i32, shape=())

        self.scale = ti.field(ti.f32, shape=())                  # pixels per AU
        self.centre = ti.Vector.field(2, ti.f32, shape=())       # pixels

    @ti.func
    def to_screen(self, p):
        return ti.Vector([self.centre[None][0] + p[0] * self.scale[None],
                          self.centre[None][1] + p[1] * self.scale[None]])

    @ti.kernel
    def clear(self):
        for i, j in self.pixels:
            v = float(j) / float(self.height)
            self.pixels[i, j] = ti.Vector([0.020 + 0.020 * v,
                                           0.024 + 0.026 * v,
                                           0.040 + 0.045 * v])

    @ti.kernel
    def draw_segments(self):
        """Anti-aliased line segments, evaluated per pixel against every segment."""
        for i, j in self.pixels:
            p = ti.Vector([float(i) + 0.5, float(j) + 0.5])
            acc = ti.Vector([0.0, 0.0, 0.0])
            for k in range(self.n_seg[None]):
                a = self.to_screen(self.seg_a[k])
                b = self.to_screen(self.seg_b[k])
                ab = b - a
                denom = ti.max(1e-6, ab.dot(ab))
                t = ti.math.clamp((p - a).dot(ab) / denom, 0.0, 1.0)
                d = (p - (a + ab * t)).norm()
                acc += self.seg_col[k] * ti.exp(-d * d / 1.4)
            self.pixels[i, j] += acc

    @ti.kernel
    def draw_orbits(self):
        for k in range(self.n_orbit[None]):
            s = self.to_screen(self.orbit_pts[k])
            x = ti.cast(s[0], ti.i32)
            y = ti.cast(s[1], ti.i32)
            if 1 <= x < self.width - 1 and 1 <= y < self.height - 1:
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        w = 0.55 if (dx == 0 and dy == 0) else 0.14
                        self.pixels[x + dx, y + dy] += self.orbit_col[k] * w

    @ti.kernel
    def draw_bodies(self):
        for i, j in self.pixels:
            p = ti.Vector([float(i) + 0.5, float(j) + 0.5])
            for k in range(self.n_body[None]):
                c = self.to_screen(self.body_pos[k])
                d = (p - c).norm()
                r = self.body_rad[k]
                if d < r + 2.0:
                    a = ti.math.clamp(r + 0.8 - d, 0.0, 1.0)
                    self.pixels[i, j] = (self.pixels[i, j] * (1.0 - a)
                                         + self.body_col[k] * a)
                if d < r * 9.0:
                    self.pixels[i, j] += self.body_col[k] * (0.10 * ti.exp(-d / (r * 2.2)))

    @ti.kernel
    def blit(self, src: ti.template(), x0: ti.i32, y0: ti.i32, border: ti.f32):
        """Composite a smaller panel in, with a one-pixel frame around it."""
        for i, j in src:
            self.pixels[x0 + i, y0 + j] = src[i, j]
        n = src.shape[0]
        m = src.shape[1]
        for i in range(-1, n + 1):
            self.pixels[x0 + i, y0 - 1] = ti.Vector([border, border, border])
            self.pixels[x0 + i, y0 + m] = ti.Vector([border, border, border])
        for j in range(-1, m + 1):
            self.pixels[x0 - 1, y0 + j] = ti.Vector([border, border, border])
            self.pixels[x0 + n, y0 + j] = ti.Vector([border, border, border])

    # ------------------------------------------------------------------
    def set_view(self, scale_px_per_au, centre_px):
        self.scale[None] = scale_px_per_au
        self.centre[None] = centre_px

    def load_orbits(self, orbits):
        """`orbits` is a list of (points Nx2 in AU, colour)."""
        k = 0
        for pts, col in orbits:
            for p in pts:
                if k >= self.MAX_ORBIT:
                    break
                self.orbit_pts[k] = [p[0], p[1]]
                self.orbit_col[k] = col
                k += 1
        self.n_orbit[None] = k

    def load_bodies(self, bodies):
        for k, (pos, col, rad) in enumerate(bodies[:16]):
            self.body_pos[k] = [pos[0], pos[1]]
            self.body_col[k] = col
            self.body_rad[k] = rad
        self.n_body[None] = min(len(bodies), 16)

    def load_segments(self, segments):
        for k, (a, b, col) in enumerate(segments[:16]):
            self.seg_a[k] = [a[0], a[1]]
            self.seg_b[k] = [b[0], b[1]]
            self.seg_col[k] = col
        self.n_seg[None] = min(len(segments), 16)

    def image(self):
        return self.pixels.to_numpy()
