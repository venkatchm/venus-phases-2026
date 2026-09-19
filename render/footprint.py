# NOTE: no `from __future__ import annotations` here -- PEP 563 turns the
# Taichi kernel argument annotations into strings, which Taichi rejects.
"""Where on Earth the occultation is visible.

The expensive part is embarrassingly parallel and is exactly the kind of thing
Taichi is for: the geocentric positions of the Moon, Venus and Sun depend only
on time, so they are computed once per time step on the CPU, uploaded, and then
every point of a latitude/longitude grid is stepped through the whole time
window inside one kernel. A 720x360 grid over 360 time steps is 93 million
site-instants, which no per-site Python loop would finish in reasonable time.
"""
import math

import numpy as np
import taichi as ti

from vmsim.frames import nutation, mean_obliquity_deg
from vmsim.moon import MOON_RADIUS_KM
from vmsim.observer import EARTH_EQUATORIAL_RADIUS_KM, FLATTENING_FACTOR
from vmsim.planets import RADIUS_KM
from vmsim.scene import geocentric_bodies
from vmsim.timescale import Instant, gmst_deg

DEG = math.pi / 180.0


@ti.data_oriented
class FootprintSolver:
    def __init__(self, n_lon=720, n_lat=360, n_steps=361):
        self.n_lon, self.n_lat, self.n_steps = n_lon, n_lat, n_steps
        self.moon = ti.Vector.field(3, ti.f64, shape=n_steps)
        self.venus = ti.Vector.field(3, ti.f64, shape=n_steps)
        self.sun = ti.Vector.field(3, ti.f64, shape=n_steps)
        self.gast = ti.field(ti.f64, shape=n_steps)

        self.occults = ti.field(ti.i32, shape=(n_lon, n_lat))
        self.min_sep = ti.field(ti.f32, shape=(n_lon, n_lat))     # arcsec
        self.moon_alt = ti.field(ti.f32, shape=(n_lon, n_lat))    # deg, at minimum
        self.sun_alt = ti.field(ti.f32, shape=(n_lon, n_lat))     # deg, at minimum
        self.t_start = ti.field(ti.f32, shape=(n_lon, n_lat))     # step index
        self.t_end = ti.field(ti.f32, shape=(n_lon, n_lat))
        self.jd0 = ti.field(ti.f64, shape=())
        self.step_days = ti.field(ti.f64, shape=())

    # ------------------------------------------------------------------
    def load_times(self, start, hours):
        step = hours / (self.n_steps - 1) / 24.0
        self.jd0[None] = start.jd_ut
        self.step_days[None] = step
        moon = np.zeros((self.n_steps, 3))
        venus = np.zeros((self.n_steps, 3))
        sun = np.zeros((self.n_steps, 3))
        gast = np.zeros(self.n_steps)
        for k in range(self.n_steps):
            inst = Instant(start.jd_ut + k * step)
            g = geocentric_bodies(inst)
            moon[k] = g["moon"]
            venus[k] = g["venus"]
            sun[k] = g["sun"]
            eq_eq = g["dpsi"] * math.cos(math.radians(g["eps_true"]))
            gast[k] = (gmst_deg(inst.jd_ut) + eq_eq) % 360.0
        self.moon.from_numpy(moon)
        self.venus.from_numpy(venus)
        self.sun.from_numpy(sun)
        self.gast.from_numpy(gast)

    @ti.func
    def observer_position(self, lat_deg, lon_deg, gast_deg_):
        phi = lat_deg * DEG
        u = ti.atan2(FLATTENING_FACTOR * ti.sin(phi), ti.cos(phi))
        rho_sin = FLATTENING_FACTOR * ti.sin(u)
        rho_cos = ti.cos(u)
        lst = (gast_deg_ + lon_deg) * DEG
        r = EARTH_EQUATORIAL_RADIUS_KM
        return ti.Vector([r * rho_cos * ti.cos(lst),
                          r * rho_cos * ti.sin(lst),
                          r * rho_sin])

    @ti.kernel
    def solve(self):
        for ix, iy in self.occults:
            lon = -180.0 + 360.0 * (ix + 0.5) / self.n_lon
            lat = 90.0 - 180.0 * (iy + 0.5) / self.n_lat
            best = 1e9
            best_k = 0
            first = -1.0
            last = -1.0
            hit = 0
            for k in range(self.n_steps):
                obs = self.observer_position(lat, lon, self.gast[k])
                m = self.moon[k] - obs
                v = self.venus[k] - obs
                mn = m.norm()
                vn = v.norm()
                cosang = ti.math.clamp(m.dot(v) / (mn * vn), -1.0, 1.0)
                sep = ti.acos(cosang) / DEG * 3600.0
                limb = ti.asin(MOON_RADIUS_KM / mn) / DEG * 3600.0
                if sep < best:
                    best = sep
                    best_k = k
                if sep < limb:
                    # only counts if the Moon is actually above the horizon
                    up = obs.normalized()
                    if m.dot(up) > 0.0:
                        hit = 1
                        if first < 0.0:
                            first = float(k)
                        last = float(k)
            obs = self.observer_position(lat, lon, self.gast[best_k])
            up = obs.normalized()
            m = self.moon[best_k] - obs
            s = self.sun[best_k] - obs
            self.occults[ix, iy] = hit
            self.min_sep[ix, iy] = ti.cast(best, ti.f32)
            self.moon_alt[ix, iy] = ti.cast(
                ti.asin(ti.math.clamp(m.dot(up) / m.norm(), -1.0, 1.0)) / DEG, ti.f32)
            self.sun_alt[ix, iy] = ti.cast(
                ti.asin(ti.math.clamp(s.dot(up) / s.norm(), -1.0, 1.0)) / DEG, ti.f32)
            self.t_start[ix, iy] = ti.cast(first, ti.f32)
            self.t_end[ix, iy] = ti.cast(last, ti.f32)

    # ------------------------------------------------------------------
    def summary(self):
        occ = self.occults.to_numpy()
        alt = self.moon_alt.to_numpy()
        sun = self.sun_alt.to_numpy()
        lat_centres = 90.0 - 180.0 * (np.arange(self.n_lat) + 0.5) / self.n_lat
        # cos(latitude) weighting, or an equirectangular grid over-counts the poles
        w = np.cos(np.radians(lat_centres))[None, :]
        total = float(np.sum(np.broadcast_to(w, occ.shape)))
        seen = occ == 1
        return {
            "fraction_of_globe": float(np.sum(seen * w) / total),
            "fraction_in_darkness": float(np.sum((seen & (sun < -6.0)) * w) / total),
            "fraction_in_daylight": float(np.sum((seen & (sun >= 0.0)) * w) / total),
            "fraction_in_twilight": float(
                np.sum((seen & (sun >= -6.0) & (sun < 0.0)) * w) / total),
        }


@ti.data_oriented
class FootprintRenderer:
    """Draws the solved footprint as an equirectangular map."""

    def __init__(self, solver, width=1440, height=720):
        self.solver = solver
        self.width, self.height = width, height
        self.pixels = ti.Vector.field(3, ti.f32, shape=(width, height))

    @ti.kernel
    def draw(self, mode: ti.i32):
        for i, j in self.pixels:
            lon = -180.0 + 360.0 * (i + 0.5) / self.width
            lat = -90.0 + 180.0 * (j + 0.5) / self.height
            ix = ti.cast((lon + 180.0) / 360.0 * self.solver.n_lon, ti.i32)
            iy = ti.cast((90.0 - lat) / 180.0 * self.solver.n_lat, ti.i32)
            ix = ti.math.clamp(ix, 0, self.solver.n_lon - 1)
            iy = ti.math.clamp(iy, 0, self.solver.n_lat - 1)

            occ = self.solver.occults[ix, iy]
            malt = self.solver.moon_alt[ix, iy]
            salt = self.solver.sun_alt[ix, iy]

            col = ti.Vector([0.055, 0.065, 0.095])
            if malt > 0.0:
                col = ti.Vector([0.10, 0.12, 0.17])          # Moon up, no event
            if occ == 1:
                if mode == 0:
                    # colour by sky conditions: the event is useless in daylight
                    if salt > 0.0:
                        col = ti.Vector([0.42, 0.33, 0.14])
                    elif salt > -6.0:
                        col = ti.Vector([0.85, 0.62, 0.22])
                    elif salt > -12.0:
                        col = ti.Vector([0.55, 0.80, 0.45])
                    else:
                        col = ti.Vector([0.30, 0.72, 0.95])
                else:
                    d = self.solver.t_end[ix, iy] - self.solver.t_start[ix, iy]
                    f = ti.math.clamp(d / float(self.solver.n_steps) * 6.0, 0.0, 1.0)
                    col = ti.Vector([0.15 + 0.80 * f, 0.35 + 0.35 * f, 0.85 - 0.45 * f])
                if malt < 8.0:
                    col *= 0.55 + 0.05 * malt          # dim the marginal, low-Moon edge

            # graticule every 30 degrees, and a brighter equator / prime meridian
            gl = ti.math.fract(lon / 30.0 + 0.5)
            gt = ti.math.fract(lat / 30.0 + 0.5)
            line = 0.0
            if ti.min(gl, 1.0 - gl) < 0.004 * (360.0 / 30.0) / (self.width / 100.0) * 3.0:
                line = 0.30
            if ti.min(gt, 1.0 - gt) < 0.004 * (180.0 / 30.0) / (self.height / 100.0) * 3.0:
                line = 0.30
            if ti.abs(lat) < 0.35 or ti.abs(lon) < 0.20:
                line = 0.55
            col = col * (1.0 - line) + ti.Vector([0.55, 0.60, 0.70]) * line
            # explicit cast: this kernel may run under a f64 default float
            self.pixels[i, j] = ti.cast(ti.math.clamp(col, 0.0, 1.0), ti.f32)

    def image(self):
        return self.pixels.to_numpy()
