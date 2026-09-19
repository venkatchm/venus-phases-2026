"""Solve and render the September 14 2026 occultation visibility footprint."""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import taichi as ti

ap = argparse.ArgumentParser()
ap.add_argument("--width", type=int, default=1600)
ap.add_argument("--height", type=int, default=800)
ap.add_argument("--grid", type=int, default=900)
ap.add_argument("--steps", type=int, default=481)
ap.add_argument("--arch", default="cpu")
ap.add_argument("--out", default="out/footprint.png")
ap.add_argument("--fp32", action="store_true", help="run with a 32-bit default float")
args = ap.parse_args()

ti.init(arch=getattr(ti, args.arch),
        default_fp=ti.f32 if args.fp32 else ti.f64, offline_cache=True)

from render.footprint import FootprintRenderer, FootprintSolver   # noqa: E402
from render.png import write_png                                  # noqa: E402
from render.text import draw_marker, draw_text                    # noqa: E402
from vmsim.observer import SITES                                  # noqa: E402
from vmsim.timescale import Instant, jd_to_utc_string             # noqa: E402

START = Instant.from_utc(2026, 9, 14, 6, 0)
HOURS = 12.0

solver = FootprintSolver(n_lon=args.grid * 2, n_lat=args.grid, n_steps=args.steps)
t0 = time.time()
solver.load_times(START, hours=HOURS)
t_eph = time.time() - t0
t0 = time.time()
solver.solve()
ti.sync()
t_solve = time.time() - t0
n = solver.n_lon * solver.n_lat * solver.n_steps
print(f"ephemeris: {solver.n_steps} epochs in {t_eph:.2f}s")
print(f"footprint: {n / 1e6:.0f}M site-instants in {t_solve:.2f}s "
      f"({n / max(t_solve, 1e-9) / 1e6:.0f}M/s)")
summary = solver.summary()
for k, v in summary.items():
    print(f"  {k:<24s} {v * 100:6.2f}% of the globe")

renderer = FootprintRenderer(solver, args.width, args.height)
renderer.draw(0)
img = np.ascontiguousarray(np.transpose(renderer.image(), (1, 0, 2))[::-1]).astype(np.float32)
h, w = img.shape[0], img.shape[1]


def to_pixel(lat, lon):
    return int((lon + 180.0) / 360.0 * w), int((90.0 - lat) / 180.0 * h)


occ = solver.occults.to_numpy()
sun = solver.sun_alt.to_numpy()
for s in SITES:
    ix = int((s.longitude + 180.0) / 360.0 * solver.n_lon) % solver.n_lon
    iy = int((90.0 - s.latitude) / 180.0 * solver.n_lat)
    seen = bool(occ[ix, iy])
    x, y = to_pixel(s.latitude, s.longitude)
    col = (1.0, 1.0, 1.0) if seen else (0.62, 0.62, 0.68)
    draw_marker(img, x, y, col, size=4)
    label = s.name.upper() + ("" if seen else " (NO)")
    draw_text(img, x + 7, y - 3, label, color=col, scale=1)

title = "LUNAR OCCULTATION OF VENUS - 2026 SEPTEMBER 14 - VISIBILITY"
draw_text(img, 14, 12, title, color=(1.0, 1.0, 1.0), scale=2)
draw_text(img, 14, 32, f"SEARCH WINDOW {jd_to_utc_string(START.jd_ut, False)} "
                       f"+ {HOURS:.0f}H   GRID {solver.n_lon}X{solver.n_lat}",
          color=(0.70, 0.74, 0.82), scale=1)

legend = [((0.30, 0.72, 0.95), "DARK SKY (SUN BELOW -12)"),
          ((0.55, 0.80, 0.45), "NAUTICAL TWILIGHT"),
          ((0.85, 0.62, 0.22), "CIVIL TWILIGHT"),
          ((0.42, 0.33, 0.14), "DAYLIGHT - NEEDS OPTICS"),
          ((0.10, 0.12, 0.17), "MOON UP, NO OCCULTATION"),
          ((0.055, 0.065, 0.095), "MOON BELOW HORIZON")]
ly = h - 14 - 13 * len(legend)
for col, text in legend:
    img[ly:ly + 9, 14:32] = np.asarray(col, dtype=np.float32)
    draw_text(img, 38, ly + 1, text, color=(0.92, 0.94, 0.98), scale=1)
    ly += 13

draw_text(img, 14, h - 12,
          f"GLOBE FRACTION {summary['fraction_of_globe'] * 100:.1f}%   "
          f"IN DARKNESS {summary['fraction_in_darkness'] * 100:.1f}%   "
          f"GRATICULE EVERY 30 DEG   NO COASTLINES - CITIES MARK THE GEOGRAPHY",
          color=(0.66, 0.70, 0.78), scale=1)

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
write_png(args.out, (np.clip(img, 0, 1) * 255).astype(np.uint8))
print("wrote", args.out)
