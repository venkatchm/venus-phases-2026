"""Python-side mirror of the space kernel's camera, for overlays.

Same role `render.camera` plays for the sky kernel, and the same reason for
existing: anything drawn on top of a rendered frame -- the Sun-Venus-Earth
triangle, an elongation arc, a body label -- has to project with exactly the
convention the kernel used, or it drifts away from what was rendered. An
annotation that disagrees with the trace under it would undermine the one thing
this film is claiming.

`basis` is a line-for-line match of `SpaceRenderer.set_camera`. Keep them so.
"""
from __future__ import annotations

import math

import numpy as np

from .camera import splat

DEG = math.pi / 180.0


def basis(pos, target, up=(0.0, 0.0, 1.0), roll_deg: float = 0.0):
    """Forward / right / up unit vectors for a camera at `pos` facing `target`."""
    pos = np.asarray(pos, dtype=np.float64)
    f = np.asarray(target, dtype=np.float64) - pos
    f = f / np.linalg.norm(f)
    up = np.asarray(up, dtype=np.float64)
    if abs(float(np.dot(f, up))) > 1.0 - 1e-9:
        # looking exactly down the reference axis: nothing defines a roll, so
        # pick another. The threshold is this tight on purpose -- a looser one
        # fires for any near-polar view, and then `cross(f, up)` snaps to a
        # fixed ecliptic axis and the camera's azimuth silently stops doing
        # anything. `cross` is still well conditioned in f64 a millidegree from
        # the pole; it is only degenerate *at* it.
        up = np.array([0.0, 1.0, 0.0])
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    if roll_deg:
        a = roll_deg * DEG
        r, u = r * math.cos(a) + u * math.sin(a), u * math.cos(a) - r * math.sin(a)
    return f, r, u


def project(point_au, pos, f, r, u, fov_deg: float, width: int, height: int):
    """Heliocentric AU point -> (column, row), or None if behind the camera.

    Rows count from the top, matching the image after the transpose-and-flip
    that turns a Taichi field into a picture.
    """
    v = np.asarray(point_au, dtype=np.float64) - np.asarray(pos, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return None
    d = v / n
    fz = float(np.dot(d, f))
    if fz <= 1e-6:
        return None
    t = math.tan(0.5 * fov_deg * DEG)
    x = float(np.dot(d, r)) / fz / t
    y = float(np.dot(d, u)) / fz / (t * height / width)
    return (x + 1.0) * 0.5 * width, height - 1.0 - (y + 1.0) * 0.5 * height


def _stamp(img, x0, y0, x1, y1, colour, radius, alpha, dash, gap, s0, total):
    """Rasterise one sub-segment over its own tight bounding box."""
    h, w = img.shape[0], img.shape[1]
    pad = radius + 2.0
    c0 = max(int(min(x0, x1) - pad), 0)
    c1 = min(int(max(x0, x1) + pad) + 1, w)
    r0 = max(int(min(y0, y1) - pad), 0)
    r1 = min(int(max(y0, y1) + pad) + 1, h)
    if c1 <= c0 or r1 <= r0:
        return
    ax, ay = x1 - x0, y1 - y0
    den = ax * ax + ay * ay
    if den < 1e-9:
        return
    px = (np.arange(c0, c1, dtype=np.float32) + np.float32(0.5 - x0))[None, :]
    py = (np.arange(r0, r1, dtype=np.float32) + np.float32(0.5 - y0))[:, None]
    t = np.clip((px * ax + py * ay) / den, 0.0, 1.0)
    d2 = (px - t * ax) ** 2 + (py - t * ay) ** 2
    weight = np.exp(-d2 / (radius * radius), dtype=np.float32) * np.float32(alpha)
    if dash > 0:
        along = s0 + t * math.hypot(ax, ay)
        weight *= (along % (dash + gap) <= dash)
    img[r0:r1, c0:c1] += weight[..., None] * np.asarray(colour, dtype=np.float32)


def draw_line(img: np.ndarray, a, b, colour, radius: float = 1.1,
              alpha: float = 1.0, dash: int = 0, gap: int = 0) -> None:
    """Soft screen-space line between two (col, row) points.

    Rasterised per-pixel over the segment's bounding box rather than by walking
    the line and splatting a dot at every step -- walking it cost four times the
    ray trace it was being drawn on top of. The segment is then chunked, because
    a long diagonal's bounding box is almost entirely empty: splitting it into
    roughly square pieces cuts the pixels touched by an order of magnitude.

    `dash`/`gap` are in pixels; dash=0 draws solid. Dashes are how a construction
    line (a sight line, an orbit tangent) is kept visually distinct from a thing
    that is actually there.
    """
    if a is None or b is None:
        return
    (x0, y0), (x1, y1) = a, b
    if not all(np.isfinite([x0, y0, x1, y1])):
        return
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 0.5:
        return
    chunks = max(1, int(length / 32.0))
    for k in range(chunks):
        s0, s1 = k / chunks, (k + 1) / chunks
        _stamp(img, x0 + (x1 - x0) * s0, y0 + (y1 - y0) * s0,
               x0 + (x1 - x0) * s1, y0 + (y1 - y0) * s1,
               colour, radius, alpha, dash, gap, s0 * length, length)


def draw_arc(img: np.ndarray, centre, a, b, radius_px: float, colour,
             alpha: float = 1.0, dash: int = 0, gap: int = 0,
             thickness: float = 1.1) -> None:
    """Arc at `centre` sweeping the short way from direction `a` to direction `b`.

    Used to mark the elongation at the Earth and the phase angle at Venus, so it
    has to take the *minor* arc -- the angle the geometry actually means, never
    its reflex.
    """
    if centre is None or a is None or b is None or radius_px < 1.0:
        return
    h, w = img.shape[0], img.shape[1]
    cx, cy = centre
    if not all(np.isfinite([cx, cy])):
        return
    t0 = math.atan2(a[1] - cy, a[0] - cx)
    t1 = math.atan2(b[1] - cy, b[0] - cx)
    sweep = (t1 - t0 + math.pi) % (2.0 * math.pi) - math.pi

    pad = radius_px + thickness + 2.0
    c0, c1 = max(int(cx - pad), 0), min(int(cx + pad) + 1, w)
    r0, r1 = max(int(cy - pad), 0), min(int(cy + pad) + 1, h)
    if c1 <= c0 or r1 <= r0:
        return

    dx = (np.arange(c0, c1, dtype=np.float32) + np.float32(0.5 - cx))[None, :]
    dy = (np.arange(r0, r1, dtype=np.float32) + np.float32(0.5 - cy))[:, None]
    rr = np.hypot(dx, dy)
    # offset from the arc's own angular start, unwrapped into the sweep
    off = (np.arctan2(dy, dx) - t0 + math.pi) % (2.0 * math.pi) - math.pi
    inside = (off / sweep >= 0.0) & (np.abs(off) <= abs(sweep)) if sweep else None
    if inside is None:
        return
    weight = np.exp(-((rr - radius_px) ** 2) / (thickness * thickness)) * alpha
    weight = np.where(inside, weight, 0.0)
    if dash > 0:
        weight = np.where((np.abs(off) * radius_px) % (dash + gap) <= dash, weight, 0.0)
    img[r0:r1, c0:c1] += weight[..., None] * np.asarray(colour, dtype=np.float32)


def arc_label_point(centre, a, b, radius_px: float):
    """Where to put the number that goes with `draw_arc`: on its bisector."""
    if centre is None or a is None or b is None:
        return None
    cx, cy = centre
    t0 = math.atan2(a[1] - cy, a[0] - cx)
    t1 = math.atan2(b[1] - cy, b[0] - cx)
    d = (t1 - t0 + math.pi) % (2.0 * math.pi) - math.pi
    t = t0 + 0.5 * d
    return cx + radius_px * math.cos(t), cy + radius_px * math.sin(t)
