#!/usr/bin/env python3
"""Interactive simulation of the September 2026 Moon-Venus encounter.

Three views over one shared clock:

  1  SKY        ray-traced view from a chosen site; the occultation happens
                because the Moon's sphere is nearer than Venus' sphere
  2  GEOMETRY   the Sun-Venus-Earth triangle that sets the phase and elongation,
                plus the parallax diagram that explains why the occultation is
                visible from some places and not others
  3  FOOTPRINT  where on Earth Venus is hidden, solved on a global grid

Run `python main.py --help` for options, or press H in the window for keys.
"""
import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import taichi as ti

DEG = math.pi / 180.0

# The events this simulation is about, as UT instants.
KEY_MOMENTS = [
    ("Moon approaches Venus", (2026, 9, 13, 14, 0)),
    ("Occultation - Europe/Africa", (2026, 9, 14, 9, 25)),
    ("Disappearance over India", (2026, 9, 14, 11, 56)),
    ("Reappearance over Chennai", (2026, 9, 14, 13, 12)),
    ("Moon has moved on", (2026, 9, 15, 14, 0)),
    ("Greatest brilliancy", (2026, 9, 24, 13, 0)),
    ("Late September - sinking", (2026, 9, 30, 13, 0)),
]

HELP_LINES = [
    "SPACE play/pause      LEFT/RIGHT step time     UP/DOWN speed (x real time)",
    "1 sky   2 geometry   3 footprint     N/P next/prev site",
    "[ ] zoom    , . step size    G ground    F follow    T next key moment",
    "A airless (default on; toggles the atmosphere back in)",
    "R reset to occultation      H toggle help      Q quit",
]


class App:
    def __init__(self, args):
        self.args = args
        # f32 default: the footprint solver holds its geocentric vectors in
        # explicit f64 fields, so its arithmetic stays f64 regardless, while
        # the display kernels run at the width they actually need
        ti.init(arch=getattr(ti, args.arch), offline_cache=True, log_level=ti.WARN)

        from render.footprint import FootprintRenderer, FootprintSolver
        from render.orrery import OrreryRenderer
        from render.sky import SkyRenderer
        from vmsim.observer import SITES
        from vmsim.timescale import Instant

        self.SITES = SITES
        self.Instant = Instant
        self.site_index = next((i for i, s in enumerate(SITES)
                                if s.name.lower().startswith(args.site.lower())), 0)

        self.w, self.h = args.width, args.height
        self.sky = SkyRenderer(self.w, self.h)
        self.orrery = OrreryRenderer(self.w, self.h)
        self.inset = OrreryRenderer(360, 360)
        self.solver = FootprintSolver(n_lon=args.grid * 2, n_lat=args.grid,
                                      n_steps=args.steps)
        self.map = FootprintRenderer(self.solver, self.w, self.h)
        self.map_ready = False
        self.occults_cache = None
        self._orbit_epoch = None
        self._inset_limb = -1.0

        self.moment = 2                      # index into KEY_MOMENTS
        self.jd = Instant.from_utc(*KEY_MOMENTS[self.moment][1]).jd_ut
        self.rate = 120.0                    # simulated seconds per real second
        self.last_tick = None
        self.step_seconds = 60.0              # one arrow press
        self.playing = True
        self.view = 1
        self.fov = args.fov
        self.follow = True
        self.ground = True
        # Airless by default: the atmosphere is physically right but it washes
        # the Moon out, and the first thing anyone wants to look at is the
        # surface. Press A (or pass --atmosphere) for the view from the ground.
        self.airless = not args.atmosphere
        self.show_help = True
        self.frame_index = 0
        self.state = None
        self._star_site = None

    # ------------------------------------------------------------------
    def current_site(self):
        return self.SITES[self.site_index]

    def update_state(self):
        from vmsim.scene import sky_state
        self.state = sky_state(self.Instant(self.jd), self.current_site())

    # ------------------------------------------------------------------
    def draw_sky(self):
        from render.bridge import apply_state, load_star_field
        st = self.state
        if self._star_site != (self.site_index, int(self.jd * 24)):
            load_star_field(self.sky, st)
            self._star_site = (self.site_index, int(self.jd * 24))
        if self.follow:
            mid_az = st.moon.azimuth_deg
            mid_alt = st.moon.apparent_altitude_deg
        else:
            mid_az, mid_alt = st.sun.azimuth_deg, 15.0
        self.sky.set_camera(mid_az, mid_alt, self.fov)
        self.sky.samples[None] = 1 if self.playing else 2
        self.sky.airless[None] = 1 if self.airless else 0
        apply_state(self.sky, st, show_ground=self.ground,
                    twinkle=self.frame_index * 0.05)
        self.sky.color.fill(0.0)
        self.sky.render()
        self.sky.splat_venus()
        self.sky.splat_stars()
        self.sky.tonemap()
        return self.sky.pixels

    # ------------------------------------------------------------------
    def draw_geometry(self):
        from vmsim.planets import kepler_heliocentric_j2000, orbit_ellipse_j2000, ORBIT_COLOR
        st = self.state
        t = self.Instant(self.jd).t

        # Orbit ellipses move imperceptibly over a simulated day, and pushing
        # ~900 points into fields from Python costs ~10 ms. Rebuild only when
        # the epoch has actually moved.
        epoch_key = int(self.jd)
        if epoch_key != self._orbit_epoch:
            orbits = []
            for name in ("mercury", "venus", "earth", "mars"):
                pts = [(p[0], p[1]) for p in orbit_ellipse_j2000(name, t, 220)]
                c = ORBIT_COLOR[name]
                orbits.append((pts, (c[0] * 0.32, c[1] * 0.32, c[2] * 0.32)))
            self.orrery.load_orbits(orbits)
            self._orbit_epoch = epoch_key

        pos = {n: kepler_heliocentric_j2000(n, t) for n in ("mercury", "venus", "earth", "mars")}
        bodies = [((0.0, 0.0), (1.0, 0.93, 0.62), 9.0)]
        for name, rad in (("mercury", 3.0), ("venus", 5.0), ("earth", 5.2), ("mars", 3.6)):
            c = ORBIT_COLOR[name]
            bodies.append(((pos[name][0], pos[name][1]), c, rad))
        self.orrery.load_bodies(bodies)

        e = (pos["earth"][0], pos["earth"][1])
        v = (pos["venus"][0], pos["venus"][1])
        self.orrery.load_segments([
            (e, (0.0, 0.0), (0.16, 0.15, 0.09)),        # Earth - Sun
            (e, v, (0.20, 0.16, 0.07)),                 # Earth - Venus, the sight line
            (v, (0.0, 0.0), (0.15, 0.13, 0.09)),        # Venus - Sun
        ])
        self.orrery.set_view(min(self.w, self.h) * 0.30, (self.w * 0.38, self.h * 0.5))
        self.orrery.clear()
        self.orrery.draw_segments()
        self.orrery.draw_orbits()
        self.orrery.draw_bodies()

        # --- parallax inset: Venus against the lunar limb, two viewpoints ---
        from vmsim.frames import spherical
        from vmsim.scene import geocentric_bodies
        g = geocentric_bodies(self.Instant(self.jd))
        ra_m, dec_m, _ = spherical(g["moon"])
        ra_v, dec_v, _ = spherical(g["venus"])
        geo = ((ra_v - ra_m) * math.cos(dec_m) / DEG * 3600.0,
               (dec_v - dec_m) / DEG * 3600.0)
        topo = ((st.venus.ra_deg - st.moon.ra_deg) * math.cos(st.moon.dec_deg * DEG) * 3600.0,
                (st.venus.dec_deg - st.moon.dec_deg) * 3600.0)
        limb = st.moon.angular_radius_deg * 3600.0

        if abs(limb - self._inset_limb) > 0.004 * max(limb, 1.0):
            ring = [(limb * math.cos(a * DEG), limb * math.sin(a * DEG))
                    for a in range(0, 360, 2)]
            self.inset.load_orbits([(ring, (0.55, 0.56, 0.62))])
            self._inset_limb = limb
        self.inset.load_bodies([
            (geo, (0.45, 0.55, 0.95), 4.0),
            (topo, (1.0, 0.92, 0.60), 5.0),
        ])
        self.inset.load_segments([(geo, topo, (0.10, 0.10, 0.14))])
        # wide enough that the geocentric and topocentric marks both fit
        span = max(limb * 1.35, abs(geo[0]), abs(geo[1]), abs(topo[0]), abs(topo[1]))
        self.inset.set_view(165.0 / max(span, 1.0), (180.0, 180.0))
        self.inset.clear()
        self.inset.draw_segments()
        self.inset.draw_orbits()
        self.inset.draw_bodies()
        self.orrery.blit(self.inset.pixels, self.w - 380, 20, 0.22)
        return self.orrery.pixels

    # ------------------------------------------------------------------
    def draw_footprint(self):
        if not self.map_ready:
            self.solver.load_times(self.Instant.from_utc(2026, 9, 14, 6, 0), hours=12.0)
            self.solver.solve()
            # pulled back once, not once per site per frame
            self.occults_cache = self.solver.occults.to_numpy()
            self.map_ready = True
        self.map.draw(0)
        return self.map.pixels

    # ------------------------------------------------------------------
    def hud(self, gui):
        from vmsim.timescale import jd_to_utc_string
        st = self.state
        site = self.current_site()
        white, dim, warm = 0xFFFFFF, 0x93A0B4, 0xFFD27F
        y = 0.975

        def line(text, color=white, size=15):
            nonlocal y
            gui.text(text, pos=(0.012, y), color=color, font_size=size)
            y -= 0.030

        if self.view == 1:
            line(f"{jd_to_utc_string(self.jd)}   {site.name}, {site.region}", white, 17)
            local = (self.jd + site.longitude / 360.0 + 0.5) % 1.0 * 24.0
            line(f"local solar time {int(local):02d}:{int(local % 1 * 60):02d}   "
                 f"sun altitude {st.sun.apparent_altitude_deg:+.1f} deg", dim)
            line(f"Moon  alt {st.moon.apparent_altitude_deg:+5.1f}  az {st.moon.azimuth_deg:5.1f}  "
                 f"illum {st.moon.illuminated_fraction * 100:4.1f}%  "
                 f"diam {st.moon.angular_diameter_arcsec / 60:.1f}'", dim)
            line(f"Venus alt {st.venus.apparent_altitude_deg:+5.1f}  az {st.venus.azimuth_deg:5.1f}  "
                 f"mag {st.venus.magnitude:+.2f}  illum {st.venus.illuminated_fraction * 100:4.1f}%  "
                 f"diam {st.venus.angular_diameter_arcsec:.1f}\"", dim)
            gap = st.limb_distance_arcsec
            if st.is_occulted:
                line(f"OCCULTED - Venus is {abs(gap):.0f}\" inside the lunar limb", warm, 17)
            else:
                line(f"separation {st.separation_arcsec:.0f}\"  "
                     f"({gap:+.0f}\" from the limb)   elongation {st.elongation_deg:.1f} deg", dim)
            line(f"field {self.fov:.2f} deg   "
                 f"{'AIRLESS   ' if self.airless else ''}"
                 f"{self.rate:.0f}x real time   "
                 f"step {self.step_seconds:.0f}s   "
                 f"{'PLAYING' if self.playing else 'PAUSED'}", dim, 13)
        elif self.view == 2:
            line("GEOMETRY - inner solar system from ecliptic north", white, 17)
            line(f"{jd_to_utc_string(self.jd, False)}", dim)
            line(f"Venus elongation {st.elongation_deg:.2f} deg east of the Sun", dim)
            line(f"Venus phase angle {st.venus.phase_angle_deg:.1f} deg -> "
                 f"{st.venus.illuminated_fraction * 100:.1f}% lit, mag {st.venus.magnitude:+.2f}", dim)
            line(f"Venus distance {st.venus.distance_km / 1.495978707e8:.4f} AU", dim)
            gui.text("PARALLAX: circle = lunar limb, blue = Venus seen from Earth's centre,",
                     pos=(0.012, 0.10), color=dim, font_size=13)
            gui.text("yellow = Venus seen from this site. The offset is why the event is local.",
                     pos=(0.012, 0.07), color=dim, font_size=13)
        else:
            line("FOOTPRINT - where Venus is hidden, 2026 Sep 14", white, 17)
            line("blue dark sky / green nautical / orange civil twilight / brown daylight", dim, 13)
            for s in self.SITES:
                ix = int((s.longitude + 180.0) / 360.0 * self.solver.n_lon) % self.solver.n_lon
                iy = int((90.0 - s.latitude) / 180.0 * self.solver.n_lat)
                seen = bool(self.occults_cache[ix, iy]) if self.map_ready else False
                gui.circle(((s.longitude + 180.0) / 360.0, (s.latitude + 90.0) / 180.0),
                           color=0xFFFFFF if seen else 0x6A7180, radius=3)
                gui.text(s.name, pos=((s.longitude + 180.0) / 360.0 + 0.006,
                                      (s.latitude + 90.0) / 180.0 + 0.012),
                         color=0xFFFFFF if seen else 0x6A7180, font_size=12)

        if self.show_help:
            yy = 0.115
            for text in HELP_LINES:
                gui.text(text, pos=(0.012, yy), color=0x7E8899, font_size=12)
                yy -= 0.026

    # ------------------------------------------------------------------
    def handle(self, gui):
        for e in gui.get_events(ti.GUI.PRESS):
            k = e.key
            if k in (ti.GUI.ESCAPE, "q"):
                gui.running = False
            elif k == ti.GUI.SPACE:
                self.playing = not self.playing
            elif k == ti.GUI.LEFT:
                self.jd -= self.step_seconds / 86400.0
            elif k == ti.GUI.RIGHT:
                self.jd += self.step_seconds / 86400.0
            elif k == ti.GUI.UP:
                self.rate = min(self.rate * 2.0, 43200.0)
            elif k == ti.GUI.DOWN:
                self.rate = max(self.rate * 0.5, 1.0)
            elif k == ",":
                self.step_seconds = max(self.step_seconds * 0.5, 1.0)
            elif k == ".":
                self.step_seconds = min(self.step_seconds * 2.0, 21600.0)
            elif k in "123":
                self.view = int(k)
            elif k == "n":
                self.site_index = (self.site_index + 1) % len(self.SITES)
            elif k == "p":
                self.site_index = (self.site_index - 1) % len(self.SITES)
            elif k == "[":
                self.fov = min(self.fov * 1.35, 120.0)
            elif k == "]":
                self.fov = max(self.fov / 1.35, 0.03)
            elif k == "g":
                self.ground = not self.ground
            elif k == "a":
                self.airless = not self.airless
            elif k == "f":
                self.follow = not self.follow
            elif k == "h":
                self.show_help = not self.show_help
            elif k == "t":
                self.moment = (self.moment + 1) % len(KEY_MOMENTS)
                self.jd = self.Instant.from_utc(*KEY_MOMENTS[self.moment][1]).jd_ut
            elif k == "r":
                self.moment = 2
                self.jd = self.Instant.from_utc(*KEY_MOMENTS[self.moment][1]).jd_ut

    # ------------------------------------------------------------------
    def run(self):
        gui = ti.GUI("Moon occults Venus - 2026 September",
                     res=(self.w, self.h), fast_gui=False)
        while gui.running:
            self.handle(gui)
            now = time.perf_counter()
            elapsed = 0.0 if self.last_tick is None else min(now - self.last_tick, 0.25)
            self.last_tick = now
            if self.playing:
                # against the wall clock, not the frame counter: otherwise the
                # 75-minute occultation flies past in two seconds on a fast machine
                self.jd += self.rate * elapsed / 86400.0
            self.update_state()
            if self.view == 1:
                img = self.draw_sky()
            elif self.view == 2:
                img = self.draw_geometry()
            else:
                img = self.draw_footprint()
            gui.set_image(img)          # a Taichi field: no numpy round trip
            self.hud(gui)
            gui.show()
            self.frame_index += 1

    # ------------------------------------------------------------------
    def selftest(self, out_dir="out/selftest"):
        """Render one frame of each view to PNG, for machines with no display."""
        from render.png import write_png
        os.makedirs(out_dir, exist_ok=True)
        self.update_state()
        for view, name in ((1, "sky"), (2, "geometry"), (3, "footprint")):
            self.view = view
            field = (self.draw_sky() if view == 1 else
                     self.draw_geometry() if view == 2 else self.draw_footprint())
            img = field.to_numpy()
            rgb = (np.clip(np.transpose(img, (1, 0, 2))[::-1], 0, 1) * 255).astype(np.uint8)
            path = os.path.join(out_dir, f"{name}.png")
            write_png(path, rgb)
            print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default="Chennai", help="observing site to start from")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fov", type=float, default=1.6, help="initial field of view, degrees")
    ap.add_argument("--atmosphere", action="store_true",
                    help="start with the atmosphere on (sky, airlight, extinction); "
                         "the default is airless, and A toggles either way")
    ap.add_argument("--arch", default="cpu", choices=["cpu", "gpu", "metal", "vulkan"])
    ap.add_argument("--grid", type=int, default=480, help="footprint grid latitude count")
    ap.add_argument("--steps", type=int, default=361, help="footprint time steps")
    ap.add_argument("--selftest", action="store_true",
                    help="render one frame of each view to PNG and exit")
    args = ap.parse_args()

    app = App(args)
    if args.selftest:
        app.selftest()
    else:
        app.run()


if __name__ == "__main__":
    main()
