"""Python-side mirror of the sky kernel's camera, for overlays.

The kernel builds its rays from (azimuth, altitude, field of view); anything
drawn on top of the rendered frame -- motion trails, markers, labels pinned to a
body -- has to project with exactly the same convention or it will drift away
from what was rendered. Keeping that one projection in one place is the whole
point of this module.
"""
from __future__ import annotations

import math

import numpy as np

DEG = math.pi / 180.0


def basis(az_deg: float, alt_deg: float):
    """Forward / right / up unit vectors in the (north, east, up) horizon frame."""
    a, h = az_deg * DEG, alt_deg * DEG
    f = np.array([math.cos(h) * math.cos(a), math.cos(h) * math.sin(a), math.sin(h)])
    r = np.array([-math.sin(a), math.cos(a), 0.0])
    return f, r, np.cross(f, r)


def project(direction, az_deg: float, alt_deg: float, fov_deg: float,
            width: int, height: int):
    """Horizon-frame direction -> (column, row) in the final image, or None.

    Rows are counted from the top, matching the PNG/video frame after the
    transpose-and-flip that turns a Taichi field into an image.
    """
    f, r, u = basis(az_deg, alt_deg)
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    fz = float(np.dot(d, f))
    if fz <= 1e-6:                       # behind the camera
        return None
    t = math.tan(0.5 * fov_deg * DEG)
    x_ndc = float(np.dot(d, r)) / fz / t
    y_ndc = float(np.dot(d, u)) / fz / (t * height / width)
    col = (x_ndc + 1.0) * 0.5 * width
    row = height - 1.0 - (y_ndc + 1.0) * 0.5 * height
    return col, row


def splat(img: np.ndarray, col: float, row: float, colour, radius: float = 1.6,
          alpha: float = 1.0) -> None:
    """Additively draw a soft dot into a float RGB image."""
    h, w = img.shape[0], img.shape[1]
    c0, c1 = int(col - radius - 1), int(col + radius + 2)
    r0, r1 = int(row - radius - 1), int(row + radius + 2)
    if c1 <= 0 or r1 <= 0 or c0 >= w or r0 >= h:
        return
    c0, c1 = max(c0, 0), min(c1, w)
    r0, r1 = max(r0, 0), min(r1, h)
    ys, xs = np.mgrid[r0:r1, c0:c1]
    d2 = (xs + 0.5 - col) ** 2 + (ys + 0.5 - row) ** 2
    weight = np.exp(-d2 / (radius * radius)) * alpha
    img[r0:r1, c0:c1] += weight[..., None] * np.asarray(colour, dtype=np.float32)
