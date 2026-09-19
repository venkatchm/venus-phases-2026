"""Taichi ray tracer for the heliocentric scene: Sun, Venus, Earth in deep space.

`render.sky` puts the camera on the ground and looks up through an atmosphere.
This one puts the camera anywhere in the inner solar system and looks at the
bodies from outside, which is what it takes to show *why* Venus has phases
rather than only what the phase looks like from here.

Two properties are structural rather than decorative, because the whole point of
the film this feeds is that the geometry is not being faked:

  * The Sun is the only light, and every body's light direction is derived from
    its own position (`normalize(sun - body)`), never set independently. The lit
    hemisphere therefore cannot point anywhere but at the Sun, and the phase is
    an output of the trace.
  * Positions and distances are always true. Body *radii* may be exaggerated for
    visibility (`scale` on `set_body`), which is a lie the caller has to own by
    printing the factor on screen -- but no exaggeration touches where anything
    is, or which way the light runs.

Numerical note, inherited from `render.sky` and just as load-bearing here: the
Sun's radius is 4.7e-3 AU and the Earth's is 4.3e-5 AU, against camera ranges of
several AU. In f32 the textbook ray-sphere discriminant cancels to nothing at
those ratios. Each body is therefore solved in units of its own distance, with
the discriminant written as a difference of squared sines via a cross product,
which never forms the cancelling subtraction.
"""
from __future__ import annotations

import math

import numpy as np
import taichi as ti

DEG = math.pi / 180.0

# body kinds, matched by the shading dispatch in `shade`
SUN, VENUS, EARTH, ROCKY = 0, 1, 2, 3


@ti.data_oriented
class SpaceRenderer:
    MAX_BODIES = 8
    MAX_PATH = 8192
    MAX_STARS = 512

    def __init__(self, width: int = 1920, height: int = 1080):
        self.width, self.height = width, height
        self.color = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.pixels = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.depth = ti.field(ti.f32, shape=(width, height))

        # camera: a real position this time, not just a direction
        self.cam_pos = ti.Vector.field(3, ti.f32, shape=())
        self.cam_f = ti.Vector.field(3, ti.f32, shape=())
        self.cam_r = ti.Vector.field(3, ti.f32, shape=())
        self.cam_u = ti.Vector.field(3, ti.f32, shape=())
        self.tan_half_fov = ti.field(ti.f32, shape=())

        # bodies
        self.body_dir = ti.Vector.field(3, ti.f32, shape=self.MAX_BODIES)
        self.body_light = ti.Vector.field(3, ti.f32, shape=self.MAX_BODIES)
        self.body_axis = ti.Vector.field(3, ti.f32, shape=self.MAX_BODIES)
        self.body_tint = ti.Vector.field(3, ti.f32, shape=self.MAX_BODIES)
        self.body_sin_r = ti.field(ti.f32, shape=self.MAX_BODIES)
        self.body_dist = ti.field(ti.f32, shape=self.MAX_BODIES)
        self.body_kind = ti.field(ti.i32, shape=self.MAX_BODIES)
        self.body_spin = ti.field(ti.f32, shape=self.MAX_BODIES)
        self.body_gain = ti.field(ti.f32, shape=self.MAX_BODIES)
        self.n_body = ti.field(ti.i32, shape=())

        # orbit paths, drawn as depth-tested point trails
        self.path_pt = ti.Vector.field(3, ti.f32, shape=self.MAX_PATH)
        self.path_col = ti.Vector.field(3, ti.f32, shape=self.MAX_PATH)
        self.n_path = ti.field(ti.i32, shape=())

        # point sources: bodies whose disc is smaller than a pixel
        self.glare_pt = ti.Vector.field(3, ti.f32, shape=self.MAX_BODIES)
        self.glare_col = ti.Vector.field(3, ti.f32, shape=self.MAX_BODIES)
        self.glare_flux = ti.field(ti.f32, shape=self.MAX_BODIES)
        self.n_glare = ti.field(ti.i32, shape=())

        # background stars
        self.star_dir = ti.Vector.field(3, ti.f32, shape=self.MAX_STARS)
        self.star_col = ti.Vector.field(3, ti.f32, shape=self.MAX_STARS)
        self.star_mag = ti.field(ti.f32, shape=self.MAX_STARS)
        self.n_stars = ti.field(ti.i32, shape=())

        self.samples = ti.field(ti.i32, shape=())
        self.exposure = ti.field(ti.f32, shape=())
        self.compress = ti.field(ti.f32, shape=())
        self.fade = ti.field(ti.f32, shape=())
        self.corona = ti.field(ti.f32, shape=())
        self.spike = ti.field(ti.f32, shape=())
        self.sun_index = ti.field(ti.i32, shape=())

        self.samples[None] = 2
        self.exposure[None] = 1.0
        self.compress[None] = 0.85
        self.fade[None] = 1.0
        self.corona[None] = 1.0
        self.spike[None] = 0.0
        self.sun_index[None] = -1
        self.fov_deg = 40.0
        self._cam_pos = np.zeros(3)

    # ------------------------------------------------------------------
    # procedural texture support
    # ------------------------------------------------------------------
    @ti.func
    def hash3(self, p):
        q = ti.Vector([ti.math.dot(p, ti.Vector([127.1, 311.7, 74.7])),
                       ti.math.dot(p, ti.Vector([269.5, 183.3, 246.1])),
                       ti.math.dot(p, ti.Vector([113.5, 271.9, 124.6]))])
        return 2.0 * ti.math.fract(ti.sin(q) * 43758.5453) - 1.0

    @ti.func
    def value_noise(self, p):
        i = ti.floor(p)
        f = p - i
        w = f * f * (3.0 - 2.0 * f)
        total = 0.0
        for dx in ti.static(range(2)):
            for dy in ti.static(range(2)):
                for dz in ti.static(range(2)):
                    o = ti.Vector([float(dx), float(dy), float(dz)])
                    g = self.hash3(i + o)
                    wx = w.x if dx == 1 else 1.0 - w.x
                    wy = w.y if dy == 1 else 1.0 - w.y
                    wz = w.z if dz == 1 else 1.0 - w.z
                    total += wx * wy * wz * ti.math.dot(g, f - o)
        return total

    @ti.func
    def fbm3(self, p):
        return (0.5 * self.value_noise(p) + 0.25 * self.value_noise(p * 2.03)
                + 0.125 * self.value_noise(p * 4.12))

    @ti.func
    def fbm4(self, p):
        return (0.5 * self.value_noise(p) + 0.25 * self.value_noise(p * 2.03)
                + 0.125 * self.value_noise(p * 4.12) + 0.0625 * self.value_noise(p * 8.37))

    @ti.func
    def spun(self, n, k):
        """Surface normal rotated into the body's own frame (Rodrigues)."""
        axis = self.body_axis[k]
        a = -self.body_spin[k]
        c, s = ti.cos(a), ti.sin(a)
        return (n * c + ti.math.cross(axis, n) * s
                + axis * (ti.math.dot(axis, n) * (1.0 - c)))

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------
    @ti.func
    def angle_deg(self, a, b):
        """atan2 form: acos(dot) quantises small angles into visible bands."""
        return ti.atan2(ti.math.cross(a, b).norm(), ti.math.dot(a, b)) / DEG

    @ti.func
    def hit_sphere(self, d, centre_dir, sin_r):
        """Ray-sphere in units of the body's distance. Returns (t, normal).

        The sphere sits at distance 1 along `centre_dir` with radius `sin_r`,
        so every quantity in the quadratic is order 1 regardless of whether the
        body is the Sun or the Earth. Discriminant as sin_r^2 - sin^2(angle)
        rather than b^2 - (1 - sin_r^2): those are algebraically identical, but
        the textbook form rounds both terms to 1.0 in f32 once sin_r^2 drops
        below epsilon, and the body silently disappears.
        """
        b = ti.math.dot(d, centre_dir)
        perp = ti.math.cross(d, centre_dir)
        disc = sin_r * sin_r - ti.math.dot(perp, perp)
        t = -1.0
        n = ti.Vector([0.0, 0.0, 1.0])
        if disc > 0.0 and b > 0.0:
            t = b - ti.sqrt(disc)
            if t > 0.0:
                n = (d * t - centre_dir) / sin_r
        return t, n

    # ------------------------------------------------------------------
    # shading -- one dispatch per body kind
    # ------------------------------------------------------------------
    @ti.func
    def shade_sun(self, n, d):
        """Photosphere with visible-band limb darkening.

        I(mu)/I(1) = 0.3 + 0.93 mu - 0.23 mu^2 is the standard quadratic fit for
        the middle of the visible band; it is why the solar disc has a visibly
        softer edge than a uniform ball would.
        """
        mu = ti.max(0.0, -ti.math.dot(n, d))
        limb = 0.30 + 0.93 * mu - 0.23 * mu * mu
        return ti.Vector([1.00, 0.955, 0.88]) * (14.0 * limb)

    @ti.func
    def shade_venus(self, n, k, view):
        """Venus' cloud deck.

        The physics is the same as `render.sky.shade_venus`: sulphuric-acid haze
        carries light measurably past the geometric terminator, which is why the
        cusps of a crescent Venus reach so far round, and the deck brightens
        towards the limb rather than darkening.

        The banding amplitude is deliberately tiny. In visible light Venus is a
        nearly featureless white ball -- the famous swirls are ultraviolet. A
        detailed Venus would be a prettier image and a wrong one, in a film
        whose entire argument is that the picture follows from the geometry.
        """
        light = self.body_light[k]
        mu = ti.math.dot(n, light)
        lit = ti.math.smoothstep(-0.16, 0.22, mu)
        view_mu = ti.max(0.0, -ti.math.dot(n, view))
        limb = 0.72 + 0.28 * ti.pow(view_mu, 0.35)

        p = self.spun(n, k)
        lat = ti.math.dot(p, self.body_axis[k])
        band = 0.035 * ti.sin(lat * 9.0) + 0.045 * self.fbm3(p * 2.6)
        return self.body_tint[k] * (1.32 * lit * limb * (1.0 + band))

    @ti.func
    def shade_earth(self, n, k, view):
        light = self.body_light[k]
        mu = ti.math.dot(n, light)
        p = self.spun(n, k)
        lat = ti.math.dot(p, self.body_axis[k])

        # continents: a threshold on low-frequency noise, with polar ice
        cont = self.fbm4(p * 1.55)
        land = ti.math.smoothstep(0.02, 0.10, cont)
        ice = ti.math.smoothstep(0.72, 0.88, ti.abs(lat))
        ocean = ti.Vector([0.022, 0.055, 0.130])
        green = ti.Vector([0.105, 0.130, 0.070])
        desert = ti.Vector([0.250, 0.205, 0.135])
        arid = ti.math.smoothstep(0.0, 0.45, ti.abs(lat) - 0.12)
        ground = green * (1.0 - arid) + desert * arid
        surface = ocean * (1.0 - land) + ground * land
        surface = surface * (1.0 - ice) + ti.Vector([0.62, 0.66, 0.70]) * ice

        # cloud deck, brighter than anything under it
        cl = self.fbm4(p * 3.1 + ti.Vector([0.0, 0.0, 17.0]))
        cloud = ti.math.smoothstep(0.00, 0.13, cl) * 0.85
        surface = surface * (1.0 - cloud) + ti.Vector([0.72, 0.74, 0.78]) * cloud

        lit = ti.math.smoothstep(-0.09, 0.14, mu)      # twilight band
        col = surface * lit

        # specular glint off water, only where there is no land or cloud
        h = (light - view).normalized()
        spec = ti.pow(ti.max(0.0, ti.math.dot(n, h)), 90.0)
        col += ti.Vector([0.55, 0.60, 0.66]) * (spec * (1.0 - land) * (1.0 - cloud)
                                                * ti.max(0.0, mu) * 0.55)

        # Rayleigh limb: the atmosphere is optically longer at grazing view
        view_mu = ti.max(0.0, -ti.math.dot(n, view))
        rim = ti.pow(1.0 - view_mu, 4.0)
        col += ti.Vector([0.18, 0.34, 0.78]) * (rim * ti.max(0.0, mu) * 0.55)
        return col * self.body_tint[k]

    @ti.func
    def shade_rocky(self, n, k, view):
        """Airless body: Lommel-Seeliger, which does not darken at the limb."""
        light = self.body_light[k]
        mu0 = ti.math.dot(n, light)
        mu = ti.max(1e-3, -ti.math.dot(n, view))
        p = self.spun(n, k)
        albedo = 0.13 * (1.0 + 0.45 * self.fbm4(p * 6.0))
        lit = 0.0
        if mu0 > 0.0:
            lit = 2.0 * mu0 / (mu0 + mu)
        lit *= ti.math.smoothstep(-0.015, 0.045, mu0)
        return self.body_tint[k] * (albedo * lit)

    # ------------------------------------------------------------------
    @ti.func
    def trace(self, d):
        """Radiance along one ray, plus the depth of what it hit (AU)."""
        col = ti.Vector([0.0, 0.0, 0.0])
        best = 1e30
        for k in range(self.n_body[None]):
            t, n = self.hit_sphere(d, self.body_dir[k], self.body_sin_r[k])
            if t > 0.0:
                z = t * self.body_dist[k]
                if z < best:
                    best = z
                    kind = self.body_kind[k]
                    c = ti.Vector([0.0, 0.0, 0.0])
                    if kind == SUN:
                        c = self.shade_sun(n, d)
                    elif kind == VENUS:
                        c = self.shade_venus(n, k, d)
                    elif kind == EARTH:
                        c = self.shade_earth(n, k, d)
                    else:
                        c = self.shade_rocky(n, k, d)
                    col = c * self.body_gain[k]

        # corona and instrument PSF around the Sun, only where nothing nearer
        # than the Sun is in the way
        si = self.sun_index[None]
        if si >= 0 and self.corona[None] > 0.0:
            if best > self.body_dist[si] * 0.999:
                g = self.angle_deg(d, self.body_dir[si])
                r_deg = ti.asin(ti.math.clamp(self.body_sin_r[si], 0.0, 1.0)) / DEG
                # distance from the limb, in solar radii
                x = ti.max(0.0, g - r_deg) / ti.max(1e-4, r_deg)
                # Three decades of falloff inside ~3 solar radii. The real
                # K-corona is ~1e-6 of the disc one radius out and would be
                # invisible here; this is brighter than life, but it has to die
                # away fast or the tone curve lifts the whole frame off black,
                # and deep space stops looking like deep space.
                halo = (0.55 * ti.exp(-x / 0.075) + 0.10 * ti.exp(-x / 0.34)
                        + 0.012 * ti.exp(-x / 1.10))
                col += ti.Vector([1.0, 0.93, 0.80]) * (halo * self.corona[None])
                if self.spike[None] > 0.0:
                    ax = ti.math.dot(d, self.cam_r[None])
                    ay = ti.math.dot(d, self.cam_u[None])
                    cx = ti.math.dot(self.body_dir[si], self.cam_r[None])
                    cy = ti.math.dot(self.body_dir[si], self.cam_u[None])
                    ang = ti.atan2(ay - cy, ax - cx)
                    col += ti.Vector([1.0, 0.96, 0.88]) * (
                        ti.pow(ti.abs(ti.cos(2.0 * ang)), 60.0)
                        * ti.exp(-x / 1.6) * self.spike[None])
        return col, best

    @ti.kernel
    def render(self):
        n = self.samples[None]
        inv = 1.0 / float(n * n)
        for i, j in self.color:
            acc = ti.Vector([0.0, 0.0, 0.0])
            near = 1e30
            for sx in range(n):
                for sy in range(n):
                    ox = (float(sx) + 0.5) / float(n)
                    oy = (float(sy) + 0.5) / float(n)
                    u = (2.0 * (i + ox) / self.width - 1.0) * self.tan_half_fov[None]
                    v = ((2.0 * (j + oy) / self.height - 1.0)
                         * self.tan_half_fov[None] * self.height / self.width)
                    d = (self.cam_f[None] + self.cam_r[None] * u
                         + self.cam_u[None] * v).normalized()
                    c, z = self.trace(d)
                    acc += c
                    near = ti.min(near, z)
            self.color[i, j] = acc * inv
            self.depth[i, j] = near

    @ti.kernel
    def draw_paths(self):
        """Orbit trails, depth-tested against what the trace already hit.

        The depth test is the whole reason this is a kernel rather than a numpy
        overlay: without it the far half of Venus' orbit would be painted over
        the Sun instead of passing behind it, which is exactly the kind of
        quietly impossible picture this renderer exists to avoid.
        """
        for k in range(self.n_path[None]):
            v = self.path_pt[k] - self.cam_pos[None]
            dist = v.norm()
            if dist > 1e-6:
                d = v / dist
                fz = ti.math.dot(d, self.cam_f[None])
                if fz > 1e-6:
                    x = ti.math.dot(d, self.cam_r[None]) / fz / self.tan_half_fov[None]
                    y = (ti.math.dot(d, self.cam_u[None]) / fz
                         / (self.tan_half_fov[None] * self.height / self.width))
                    px = (x + 1.0) * 0.5 * self.width
                    py = (y + 1.0) * 0.5 * self.height
                    bx = ti.cast(ti.floor(px), ti.i32)
                    by = ti.cast(ti.floor(py), ti.i32)
                    for dx, dy in ti.ndrange((-1, 2), (-1, 2)):
                        i, j = bx + dx, by + dy
                        if 0 <= i < self.width and 0 <= j < self.height:
                            if dist < self.depth[i, j]:
                                ddx = float(i) + 0.5 - px
                                ddy = float(j) + 0.5 - py
                                w = ti.exp(-(ddx * ddx + ddy * ddy) / 0.75)
                                self.color[i, j] += self.path_col[k] * w

    @ti.kernel
    def splat_glare(self):
        """Deposit a sub-pixel body's flux as a point source.

        Seen from beside the Earth, Venus subtends 60 arcseconds -- a hundredth
        of a pixel in a 35-degree field. The ray-traced disc would essentially
        never be sampled and the planet would vanish from the shot, even though
        at magnitude -4.5 it is the most obvious thing in that sky after the Sun.
        Below the resolution limit the flux is therefore deposited the way a
        star's is, which conserves brightness instead of losing it between
        sample points. This is the eye's point-spread function, not decoration.
        """
        for k in range(self.n_glare[None]):
            v = self.glare_pt[k] - self.cam_pos[None]
            dist = v.norm()
            if dist > 1e-9:
                d = v / dist
                fz = ti.math.dot(d, self.cam_f[None])
                if fz > 0.0:
                    x = ti.math.dot(d, self.cam_r[None]) / fz / self.tan_half_fov[None]
                    y = (ti.math.dot(d, self.cam_u[None]) / fz
                         / (self.tan_half_fov[None] * self.height / self.width))
                    px = (x + 1.0) * 0.5 * self.width
                    py = (y + 1.0) * 0.5 * self.height
                    bx = ti.cast(ti.floor(px), ti.i32)
                    by = ti.cast(ti.floor(py), ti.i32)
                    flux = self.glare_flux[k]
                    for dx, dy in ti.ndrange((-14, 15), (-14, 15)):
                        i, j = bx + dx, by + dy
                        if 0 <= i < self.width and 0 <= j < self.height:
                            if dist < self.depth[i, j]:
                                ddx = float(i) + 0.5 - px
                                ddy = float(j) + 0.5 - py
                                r2 = ddx * ddx + ddy * ddy
                                rr = ti.sqrt(r2)
                                # core, scattering wing and four-armed spike:
                                # the shape of a bright point seen by any real
                                # optic, and wide enough that a magnitude -4.4
                                # Venus reads as brilliant rather than as one
                                # hot pixel the eye slides straight past
                                core = ti.exp(-r2 / 2.2)
                                wing = 0.060 * ti.exp(-rr / 3.2)
                                ang = ti.atan2(ddy, ddx)
                                spike = (0.050 * ti.pow(ti.abs(ti.cos(2.0 * ang)), 30.0)
                                         * ti.exp(-rr / 5.0))
                                self.color[i, j] += self.glare_col[k] * (
                                    flux * (core + wing + spike))

    @ti.kernel
    def splat_stars(self):
        """Background stars. No extinction and no twinkle: there is no air here."""
        for k in range(self.n_stars[None]):
            d = self.star_dir[k]
            fz = ti.math.dot(d, self.cam_f[None])
            if fz > 0.0:
                hidden = 0
                for b in range(self.n_body[None]):
                    if ti.math.dot(d, self.body_dir[b]) > ti.sqrt(
                            ti.max(0.0, 1.0 - self.body_sin_r[b] ** 2)):
                        hidden = 1
                if hidden == 0:
                    x = ti.math.dot(d, self.cam_r[None]) / fz / self.tan_half_fov[None]
                    y = (ti.math.dot(d, self.cam_u[None]) / fz
                         / (self.tan_half_fov[None] * self.height / self.width))
                    px = (x + 1.0) * 0.5 * self.width
                    py = (y + 1.0) * 0.5 * self.height
                    if 2 <= px < self.width - 2 and 2 <= py < self.height - 2:
                        flux = ti.pow(10.0, -0.4 * (self.star_mag[k] - 1.0)) * 0.055
                        bx = ti.cast(ti.floor(px), ti.i32)
                        by = ti.cast(ti.floor(py), ti.i32)
                        for dx, dy in ti.ndrange((-2, 3), (-2, 3)):
                            ddx = float(bx + dx) + 0.5 - px
                            ddy = float(by + dy) + 0.5 - py
                            w = ti.exp(-(ddx * ddx + ddy * ddy) / 0.62)
                            self.color[bx + dx, by + dy] += self.star_col[k] * (flux * w)

    @ti.kernel
    def tonemap(self):
        k = self.compress[None]
        for i, j in self.pixels:
            c = self.color[i, j]
            c = ti.Vector([ti.pow(ti.max(c.x, 0.0), k),
                           ti.pow(ti.max(c.y, 0.0), k),
                           ti.pow(ti.max(c.z, 0.0), k)]) * self.exposure[None]
            c = c / (1.0 + c)
            c = ti.math.clamp(c * self.fade[None], 0.0, 1.0)
            c = ti.Vector([ti.pow(c.x, 1.0 / 2.2), ti.pow(c.y, 1.0 / 2.2),
                           ti.pow(c.z, 1.0 / 2.2)])
            d = (ti.math.fract(ti.sin(float(i) * 12.9898 + float(j) * 78.233)
                               * 43758.5453) - 0.5) / 255.0
            self.pixels[i, j] = ti.math.clamp(c + d, 0.0, 1.0)

    # ------------------------------------------------------------------
    # python side
    # ------------------------------------------------------------------
    def set_camera(self, pos, target, fov_deg: float, up=(0.0, 0.0, 1.0),
                   roll_deg: float = 0.0) -> None:
        """Look from `pos` at `target`, both heliocentric ecliptic AU."""
        pos = np.asarray(pos, dtype=np.float64)
        f = np.asarray(target, dtype=np.float64) - pos
        f /= np.linalg.norm(f)
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
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        if roll_deg:
            a = roll_deg * DEG
            r, u = r * math.cos(a) + u * math.sin(a), u * math.cos(a) - r * math.sin(a)
        self._cam_pos = pos
        self.cam_pos[None] = pos
        self.cam_f[None] = f
        self.cam_r[None] = r
        self.cam_u[None] = u
        self.tan_half_fov[None] = math.tan(0.5 * fov_deg * DEG)
        self.fov_deg = fov_deg

    def clear_bodies(self) -> None:
        self.n_body[None] = 0
        self.n_glare[None] = 0
        self.sun_index[None] = -1

    def add_body(self, pos, radius_au: float, kind: int, *, scale: float = 1.0,
                 tint=(1.0, 1.0, 1.0), axis=(0.0, 0.0, 1.0), spin: float = 0.0,
                 gain: float = 1.0) -> int:
        """Place a body at true heliocentric `pos` (AU), radius optionally scaled.

        Radii arrive already in AU so that nothing here has to know what an AU is
        in kilometres -- this module stays pure geometry, and the unit lives with
        the astronomy in `vmsim`.

        The light direction is computed here from the position and nowhere else,
        which is what guarantees the terminator cannot be wrong: with the Sun at
        the origin, `normalize(sun - body)` is just `-normalize(pos)`.
        """
        k = int(self.n_body[None])
        if k >= self.MAX_BODIES:
            raise ValueError("SpaceRenderer.MAX_BODIES exceeded")
        pos = np.asarray(pos, dtype=np.float64)
        cam = self._cam_pos
        rel = pos - cam
        dist = float(np.linalg.norm(rel))
        radius_au = radius_au * scale
        if dist <= radius_au * 1.05:
            raise ValueError(f"camera is inside body {k} (dist {dist:.4g} AU, "
                             f"scaled radius {radius_au:.4g} AU)")

        r_helio = float(np.linalg.norm(pos))
        light = -pos / r_helio if r_helio > 1e-9 else np.array([0.0, 0.0, 1.0])

        ax = np.asarray(axis, dtype=np.float64)
        ax = ax / np.linalg.norm(ax)

        self.body_dir[k] = rel / dist
        self.body_light[k] = light
        self.body_axis[k] = ax
        self.body_tint[k] = tint
        self.body_sin_r[k] = radius_au / dist
        self.body_dist[k] = dist
        self.body_kind[k] = kind
        self.body_spin[k] = spin
        self.body_gain[k] = gain
        self.n_body[None] = k + 1
        if kind == SUN:
            self.sun_index[None] = k
        return k

    def load_paths(self, paths) -> None:
        """`paths` is a list of (Nx3 points in AU, colour).

        Assembled in numpy and uploaded in one transfer. Writing the points into
        the field one at a time costs a host-device round trip each, and at a
        few thousand points per orbit that alone dominated the frame time.
        """
        pts = np.zeros((self.MAX_PATH, 3), dtype=np.float32)
        cols = np.zeros((self.MAX_PATH, 3), dtype=np.float32)
        k = 0
        for arr, col in paths:
            arr = np.asarray(arr, dtype=np.float32).reshape(-1, 3)
            n = min(len(arr), self.MAX_PATH - k)
            if n <= 0:
                break
            pts[k:k + n] = arr[:n]
            cols[k:k + n] = col
            k += n
        self.path_pt.from_numpy(pts)
        self.path_col.from_numpy(cols)
        self.n_path[None] = k

    def clear_glare(self) -> None:
        self.n_glare[None] = 0

    def add_glare(self, pos, flux: float, colour=(1.0, 0.975, 0.92)) -> None:
        """A point source at true heliocentric `pos` (AU) carrying `flux`."""
        k = int(self.n_glare[None])
        if k >= self.MAX_BODIES:
            raise ValueError("SpaceRenderer.MAX_BODIES exceeded (glare)")
        self.glare_pt[k] = [pos[0], pos[1], pos[2]]
        self.glare_flux[k] = flux
        self.glare_col[k] = colour
        self.n_glare[None] = k + 1

    def load_stars(self, stars) -> None:
        n = min(len(stars), self.MAX_STARS)
        dirs = np.zeros((self.MAX_STARS, 3), dtype=np.float32)
        cols = np.zeros((self.MAX_STARS, 3), dtype=np.float32)
        mags = np.full(self.MAX_STARS, 99.0, dtype=np.float32)
        for k in range(n):
            d, mag, col = stars[k]
            dirs[k], mags[k], cols[k] = d, mag, col
        self.star_dir.from_numpy(dirs)
        self.star_col.from_numpy(cols)
        self.star_mag.from_numpy(mags)
        self.n_stars[None] = n

    def frame(self) -> np.ndarray:
        """Render and tone-map. Returns the raw (width, height, 3) field."""
        self.color.fill(0.0)
        self.depth.fill(1e30)
        self.render()
        if self.n_path[None] > 0:
            self.draw_paths()
        if self.n_glare[None] > 0:
            self.splat_glare()
        if self.n_stars[None] > 0:
            self.splat_stars()
        self.tonemap()
        return self.pixels.to_numpy()

    def image(self) -> np.ndarray:
        """Render as a (height, width, 3) float image, rows top-down."""
        return np.ascontiguousarray(
            np.transpose(self.frame(), (1, 0, 2))[::-1].astype(np.float32))
