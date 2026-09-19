"""Headless renderer: write PNG stills of the event to out/."""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import taichi as ti

from render.bridge import apply_state, load_star_field
from render.png import write_png
from render.sky import SkyRenderer
from vmsim.observer import find_site
from vmsim.scene import sky_state
from vmsim.timescale import Instant


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="Chennai")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fov", type=float, default=2.0)
    ap.add_argument("--out", default="out")
    ap.add_argument("--arch", default="cpu")
    ap.add_argument("--times", nargs="*", default=[])
    ap.add_argument("--exposure", type=float, default=1.0)
    args = ap.parse_args()

    ti.init(arch=getattr(ti, args.arch), offline_cache=True)
    site = find_site(args.site)
    renderer = SkyRenderer(args.width, args.height)

    times = args.times or ["12:00", "12:20", "12:40", "13:00", "13:16", "13:40"]
    os.makedirs(args.out, exist_ok=True)
    for label in times:
        hh, mm = (int(x) for x in label.split(":"))
        st = sky_state(Instant.from_utc(2026, 9, 14, hh, mm), site)
        load_star_field(renderer, st)
        renderer.set_camera(st.moon.azimuth_deg, st.moon.apparent_altitude_deg, args.fov)
        apply_state(renderer, st, exposure_bias=args.exposure)
        img = renderer.frame()
        rgb = (np.clip(np.transpose(img, (1, 0, 2))[::-1], 0, 1) * 255).astype(np.uint8)
        path = os.path.join(args.out, f"{args.site.lower()}_{hh:02d}{mm:02d}.png")
        write_png(path, rgb)
        print(f"{path}  sun {st.sun.altitude_deg:+6.1f}  sep {st.separation_arcsec:7.1f}\"  "
              f"occulted={st.is_occulted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
