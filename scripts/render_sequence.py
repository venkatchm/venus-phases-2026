"""Render a labelled contact sheet of the occultation as seen from one site."""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import taichi as ti

ap = argparse.ArgumentParser()
ap.add_argument("--site", default="Chennai")
ap.add_argument("--cols", type=int, default=3)
ap.add_argument("--tile", type=int, default=520)
ap.add_argument("--fov", type=float, default=1.35)
ap.add_argument("--arch", default="cpu")
ap.add_argument("--centre", default="moon", choices=["moon", "venus"],
                help="which body stays fixed in frame")
ap.add_argument("--out", default="out/sequence.png")
args = ap.parse_args()

ti.init(arch=getattr(ti, args.arch), offline_cache=True, log_level=ti.WARN)

from render.bridge import apply_state, load_star_field    # noqa: E402
from render.png import write_png                          # noqa: E402
from render.sky import SkyRenderer                        # noqa: E402
from render.text import draw_text                         # noqa: E402
from vmsim.events import find_occultation                 # noqa: E402
from vmsim.observer import find_site                      # noqa: E402
from vmsim.scene import sky_state                         # noqa: E402
from vmsim.timescale import Instant, jd_to_utc_string     # noqa: E402

site = find_site(args.site)
occ = find_occultation(site, Instant.from_utc(2026, 9, 14, 4, 0), hours=14.0)
if not occ.occurs:
    raise SystemExit(f"no occultation visible from {site.name}")

d = occ.disappearance.instant.jd_ut
r = occ.reappearance.instant.jd_ut
if args.centre == "venus":
    # tight on Venus, where the interesting thing is the lunar limb arriving
    frames = [
        (d - 150.0 / 86400.0, "VENUS CLEAR OF THE LIMB"),
        (d - 20.0 / 86400.0, "THE DARK LIMB ARRIVES"),
        (d + 55.0 / 86400.0, "BEING CUT IN HALF"),
        (r - 40.0 / 86400.0, "THE BRIGHT LIMB LETS GO"),
        (r + 35.0 / 86400.0, "MOSTLY BACK"),
        (r + 150.0 / 86400.0, "FULLY OUT"),
    ]
else:
    frames = [
        (d - 22.0 / 1440.0, "APPROACH"),
        (d - 45.0 / 86400.0, "SECONDS TO CONTACT"),
        (d + 90.0 / 86400.0, "DISAPPEARANCE"),
        (0.5 * (d + r), "HIDDEN BEHIND THE MOON"),
        (r + 45.0 / 86400.0, "REAPPEARANCE"),
        (r + 22.0 / 1440.0, "CLEAR OF THE LIMB"),
    ]

tile = args.tile
th = int(tile * 0.72)
cols = args.cols
rows = (len(frames) + cols - 1) // cols
pad, header = 8, 46
W = cols * tile + (cols + 1) * pad
H = header + rows * (th + 34) + pad
sheet = np.zeros((H, W, 3), np.float32)
sheet[:] = (0.035, 0.038, 0.048)

renderer = SkyRenderer(tile, th)
renderer.samples[None] = 3

for k, (jd, label) in enumerate(frames):
    st = sky_state(Instant(jd), site)
    load_star_field(renderer, st)
    # keep one body fixed in frame so the eye tracks the relative motion
    anchor = st.venus if args.centre == "venus" else st.moon
    renderer.set_camera(anchor.azimuth_deg, anchor.apparent_altitude_deg, args.fov)
    apply_state(renderer, st, show_ground=False)
    img = renderer.frame()
    rgb = np.transpose(img, (1, 0, 2))[::-1]

    cx, cy = k % cols, k // cols
    x0 = pad + cx * (tile + pad)
    y0 = header + cy * (th + 34)
    sheet[y0:y0 + th, x0:x0 + tile] = np.clip(rgb, 0, 1)

    gap = st.limb_distance_arcsec
    status = (f"VENUS HIDDEN, {abs(gap):.0f}\" INSIDE THE LIMB" if st.is_occulted
              else f"SEPARATION {st.separation_arcsec:.0f}\"  ({gap:+.0f}\" FROM LIMB)")
    draw_text(sheet, x0 + 2, y0 + th + 4,
              f"{jd_to_utc_string(jd)[11:19]} UT  {label}", (1.0, 1.0, 1.0), 1)
    draw_text(sheet, x0 + 2, y0 + th + 16, status, (0.62, 0.68, 0.80), 1)

draw_text(sheet, pad + 2, 8,
          f"LUNAR OCCULTATION OF VENUS - 2026 SEPTEMBER 14 - FROM {site.name.upper()}, "
          f"{site.region.upper()}", (1.0, 1.0, 1.0), 2)
draw_text(sheet, pad + 2, 28,
          f"FIELD {args.fov:.2f} DEG, {args.centre.upper()} CENTRED   DISAPPEARANCE "
          f"{jd_to_utc_string(d)[11:19]} UT   REAPPEARANCE {jd_to_utc_string(r)[11:19]} UT"
          f"   DURATION {occ.duration_seconds / 60:.0f} MIN",
          (0.66, 0.72, 0.84), 1)

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
write_png(args.out, (np.clip(sheet, 0, 1) * 255).astype(np.uint8))
print("wrote", args.out)
