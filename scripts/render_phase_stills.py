"""A contact sheet of Venus through one apparition, at a single fixed scale.

Eight dates evenly spaced from superior to inferior conjunction, each rendered
at the same arcseconds per pixel, so the panels can be compared with a ruler.
The full cycle is in one image: Venus starts small and full on the far side of
the Sun, and ends enormous and thin on the near side.

    python scripts/render_phase_stills.py --arch gpu

This is also the quick check on the renderer. It takes seconds, where the film
takes minutes, so it is the thing to look at first after changing any shading.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import taichi as ti

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--cols", type=int, default=4)
ap.add_argument("--rows", type=int, default=2)
ap.add_argument("--tile", type=int, default=480)
# 0.15 rather than the film's 0.13: the same tile has to hold the 60-arcsec
# crescent near inferior conjunction without clipping it
ap.add_argument("--arcsec-per-px", type=float, default=0.15)
ap.add_argument("--samples", type=int, default=3)
ap.add_argument("--arch", default="cpu")
ap.add_argument("--year", type=int, default=2026)
ap.add_argument("--out", default="out/venus_phases_sheet.png")
args = ap.parse_args()

ti.init(arch=getattr(ti, args.arch), offline_cache=True, log_level=ti.WARN)

from render.bridge import radius_au, spin_phase                       # noqa: E402
from render.png import write_png                                      # noqa: E402
from render.space import VENUS, SpaceRenderer                         # noqa: E402
from render.text import draw_text, text_width                         # noqa: E402
from vmsim.events import apparition_beats                             # noqa: E402
from vmsim.planets import geocentric_of_date, heliocentric_of_date    # noqa: E402
from vmsim.scene import venus_geometry                                # noqa: E402
from vmsim.timescale import Instant, jd_to_utc_string                 # noqa: E402

print("solving the apparition...")
beats = dict(apparition_beats(Instant.from_utc(args.year - 1, 12, 1), span_days=400.0))
try:
    t0 = beats["superior conjunction"].jd_ut
    t1 = beats["inferior conjunction"].jd_ut
except KeyError:
    raise SystemExit(f"no complete apparition found starting {args.year - 1}-12")
print(f"  superior conjunction {jd_to_utc_string(t0, seconds=False)}")
print(f"  inferior conjunction {jd_to_utc_string(t1, seconds=False)}")

N = args.cols * args.rows
TILE = args.tile
HEAD = 44
W = args.cols * TILE
H = HEAD + args.rows * TILE
UI = max(1, TILE // 240)

renderer = SpaceRenderer(TILE, TILE)
renderer.samples[None] = args.samples
renderer.corona[None] = 0.0
renderer.exposure[None] = 1.9
fov = TILE * args.arcsec_per_px / 3600.0

sheet = np.zeros((H, W, 3), np.float32)
draw_text(sheet, 8 * UI, 5 * UI,
          f"THE PHASES OF VENUS - {args.year} APPARITION", (1.0, 1.0, 1.0), 2 * UI)
note = f"ALL PANELS AT {args.arcsec_per_px}\"/PIXEL - THE DISC REALLY DOES GROW"
draw_text(sheet, W - text_width(note, UI) - 8 * UI, 8 * UI, note, (0.62, 0.68, 0.82), UI)
sheet[HEAD - 1:HEAD] = (0.30, 0.34, 0.42)

# The last panel stops a few days short of inferior conjunction: at conjunction
# itself Venus is under one percent lit and its crescent is a thread, which
# reproduces correctly but reads as an empty panel.
for k in range(N):
    jd = t0 + (t1 - 5.0 - t0) * k / (N - 1)
    inst = Instant(jd)
    earth = np.array(heliocentric_of_date("earth", inst.jde))
    _geo, _lt, venus = geocentric_of_date("venus", inst.jde)
    venus = np.array(venus)
    renderer.set_camera(earth, venus, fov)
    renderer.clear_bodies()
    renderer.add_body(venus, radius_au("venus"), VENUS, tint=(1.00, 0.978, 0.912),
                      axis=(0.0, 0.0, -1.0), spin=spin_phase("venus", jd))
    panel = renderer.image()

    g = venus_geometry(inst)
    cx, cy = k % args.cols, k // args.cols
    x0, y0 = cx * TILE, HEAD + cy * TILE
    sheet[y0:y0 + TILE, x0:x0 + TILE] = panel
    draw_text(sheet, x0 + 8 * UI, y0 + TILE - 40 * UI,
              jd_to_utc_string(jd, seconds=False)[:10], (0.88, 0.92, 1.0), UI)
    draw_text(sheet, x0 + 8 * UI, y0 + TILE - 27 * UI,
              f"{g.angular_diameter_arcsec:.1f}\"   {g.illuminated_fraction * 100:.0f}% LIT",
              (1.00, 0.84, 0.52), UI)
    draw_text(sheet, x0 + 8 * UI, y0 + TILE - 14 * UI,
              f"{g.delta_au:.2f} AU   ELONG {g.elongation_deg:.0f} DEG",
              (0.50, 0.56, 0.68), UI)
    if cx:
        sheet[y0:y0 + TILE, x0:x0 + 1] = (0.16, 0.18, 0.24)
    if cy:
        sheet[y0:y0 + 1, x0:x0 + TILE] = (0.16, 0.18, 0.24)
    print(f"  {jd_to_utc_string(jd, seconds=False)[:10]}  "
          f"{g.angular_diameter_arcsec:5.1f}\"  {g.illuminated_fraction * 100:5.1f}% lit")

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
write_png(args.out, (np.clip(sheet, 0.0, 1.0) * 255).astype(np.uint8))
print(f"wrote {args.out}  ({W}x{H})")
