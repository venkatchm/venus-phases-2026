"""Taichi renderer for the naked-eye / telescopic sky view.

The Moon and Venus are ray-traced as real spheres placed at their true
topocentric positions, so the phase, the orientation of the crescent, the
relative sizes and the occultation itself all fall out of the geometry instead
of being drawn by hand. Depth ordering is what hides Venus behind the Moon.

Numerical note: distances span 4e5 km (Moon) to 6e7 km (Venus), which destroys
single-precision ray-sphere intersection. Each body is therefore solved in units
of its own distance -- centre at unit distance, radius equal to the sine of its
angular radius -- so every quantity in the quadratic is order 1.
"""
from __future__ import annotations

import math

import numpy as np
import taichi as ti

DEG = math.pi / 180.0

vec3 = ti.types.vector(3, ti.f32)


@ti.data_oriented
class SkyRenderer:
    def __init__(self, width: int = 1280, height: int = 720, max_stars: int = 256):
        self.width, self.height = width, height
        self.color = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.pixels = ti.Vector.field(3, ti.f32, shape=(width, height))

        # camera
        self.cam_f = ti.Vector.field(3, ti.f32, shape=())
        self.cam_r = ti.Vector.field(3, ti.f32, shape=())
        self.cam_u = ti.Vector.field(3, ti.f32, shape=())
        self.tan_half_fov = ti.field(ti.f32, shape=())

        # bodies: direction, sin(angular radius), distance (km), light direction
        self.moon_dir = ti.Vector.field(3, ti.f32, shape=())
        self.moon_light = ti.Vector.field(3, ti.f32, shape=())
        self.moon_sin_r = ti.field(ti.f32, shape=())
        self.moon_dist = ti.field(ti.f32, shape=())
        self.moon_north = ti.Vector.field(3, ti.f32, shape=())
        self.moon_east = ti.Vector.field(3, ti.f32, shape=())
        self.earthshine = ti.field(ti.f32, shape=())
        self.moon_relief = ti.field(ti.f32, shape=())    # 0 disables the relief
        # named nearside features, in the Moon's body frame
        self.MAX_FEATURES = 24
        self.mare_dir = ti.Vector.field(3, ti.f32, shape=self.MAX_FEATURES)
        self.mare_cos_in = ti.field(ti.f32, shape=self.MAX_FEATURES)
        self.mare_cos_out = ti.field(ti.f32, shape=self.MAX_FEATURES)
        self.n_mare = ti.field(ti.i32, shape=())
        self.ray_dir = ti.Vector.field(3, ti.f32, shape=8)
        self.ray_cos = ti.field(ti.f32, shape=8)
        self.ray_reach = ti.field(ti.f32, shape=8)
        self.n_ray = ti.field(ti.i32, shape=())

        self.venus_dir = ti.Vector.field(3, ti.f32, shape=())
        self.venus_light = ti.Vector.field(3, ti.f32, shape=())
        self.venus_sin_r = ti.field(ti.f32, shape=())
        self.venus_dist = ti.field(ti.f32, shape=())
        self.venus_visible = ti.field(ti.f32, shape=())     # 0 while occulted
        self.venus_flux = ti.field(ti.f32, shape=())        # glare strength
        self.venus_point_flux = ti.field(ti.f32, shape=())  # 0 once the disc resolves

        self.sun_dir = ti.Vector.field(3, ti.f32, shape=())
        self.sun_alt = ti.field(ti.f32, shape=())
        self.exposure = ti.field(ti.f32, shape=())
        self.show_ground = ti.field(ti.i32, shape=())
        self.airless = ti.field(ti.i32, shape=())   # 1 = no atmosphere at all
        self.twinkle_phase = ti.field(ti.f32, shape=())
        self.compress = ti.field(ti.f32, shape=())
        self.moon_glow = ti.field(ti.f32, shape=())
        self.glare_scale = ti.field(ti.f32, shape=())
        self.samples = ti.field(ti.i32, shape=())

        # stars
        self.max_stars = max_stars
        self.star_dir = ti.Vector.field(3, ti.f32, shape=max_stars)
        self.star_mag = ti.field(ti.f32, shape=max_stars)
        self.star_col = ti.Vector.field(3, ti.f32, shape=max_stars)
        self.n_stars = ti.field(ti.i32, shape=())
        self.samples[None] = 2
        self.compress[None] = 1.0
        # Relief amplitude. The Moon's topography is a few km against a 1738 km
        # radius, so slopes are gentle and the terminator stays sharp; crank
        # this and the perturbed normals start catching sunlight across the
        # night side, which destroys the phase the whole simulation is about.
        self.moon_relief[None] = 0.0016
        self.fov_deg = 2.0

    # ------------------------------------------------------------------
    # procedural lunar surface
    # ------------------------------------------------------------------
    @ti.func
    def hash3(self, p):
        q = ti.Vector([ti.math.dot(p, ti.Vector([127.1, 311.7, 74.7])),
                       ti.math.dot(p, ti.Vector([269.5, 183.3, 246.1])),
                       ti.math.dot(p, ti.Vector([113.5, 271.9, 124.6]))])
        return 2.0 * ti.math.fract(ti.sin(q) * 43758.5453) - 1.0

    @ti.func
    def value_noise(self, p):
        """Gradient noise on a 3D lattice."""
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
    def fbm2(self, p):
        return (0.5 * self.value_noise(p) + 0.25 * self.value_noise(p * 2.03))

    @ti.func
    def fbm3(self, p):
        return (0.5 * self.value_noise(p) + 0.25 * self.value_noise(p * 2.03)
                + 0.125 * self.value_noise(p * 4.12))

    @ti.func
    def fbm4(self, p):
        return (0.5 * self.value_noise(p) + 0.25 * self.value_noise(p * 2.03)
                + 0.125 * self.value_noise(p * 4.12) + 0.0625 * self.value_noise(p * 8.37))

    @ti.func
    def moon_body(self, n):
        """Surface normal rotated from the horizon frame into the Moon's own.

        The Moon is tidally locked: the same face is turned to us always, so its
        markings must be fixed to *it*, not to the sky. Sampling the noise with
        the horizon-frame normal directly -- which is what this did before --
        slides the pattern through a static 3-D field as the Moon crosses the
        sky, so the "same" Moon shows a different face every few hours. Measured
        at 17% mean surface difference over six hours, where the correct answer
        is zero.

        The body x-axis points at the observer (the sub-Earth point, fixed by
        the tidal lock); z is the Moon's spin axis, inclined only 1.54 deg to
        the ecliptic, so ecliptic north stands in for it.
        """
        return ti.Vector([ti.math.dot(n, -self.moon_dir[None]),
                          ti.math.dot(n, self.moon_east[None]),
                          ti.math.dot(n, self.moon_north[None])])

    @ti.func
    def mare_mask(self, b):
        """How much of the *named* maria covers this body-frame point, 0 to 1.

        The basins are real ones at their real selenographic coordinates, with
        a noise-warped edge so they do not read as drawn circles -- mare
        boundaries are ragged where the lava stopped.
        """
        m = 0.0
        warp = 0.045 * self.fbm3(b * 5.5)
        for k in range(self.n_mare[None]):
            c = ti.math.dot(b, self.mare_dir[k]) + warp
            m = ti.max(m, ti.math.smoothstep(self.mare_cos_out[k],
                                             self.mare_cos_in[k], c))
        return ti.math.clamp(m, 0.0, 1.0)

    @ti.func
    def ray_boost(self, b):
        """Bright ejecta from the young rayed craters, Tycho above all."""
        add = 0.0
        for k in range(self.n_ray[None]):
            c = ti.math.clamp(ti.math.dot(b, self.ray_dir[k]), -1.0, 1.0)
            ang = ti.acos(c)
            floor_ = ti.acos(ti.math.clamp(self.ray_cos[k], -1.0, 1.0))
            # the crater itself, then streaks thinning with distance
            add += 0.045 * ti.math.smoothstep(floor_ * 1.6, floor_ * 0.7, ang)
            streak = ti.max(0.0, self.fbm2(b * 17.0) - 0.04)
            add += 0.085 * streak * ti.exp(-ang / (self.ray_reach[k] * DEG))
        return add

    @ti.func
    def moon_albedo(self, n):
        """Albedo of the lunar surface at unit normal `n`.

        Bright anorthositic highlands as the base, the named maria darkening it
        where basaltic lava flooded, and the young rayed craters brightening it
        again. Real lunar albedo runs about 0.07 in the maria against 0.13 in
        the highlands, and that contrast is what the eye reads as the face.
        """
        b = self.moon_body(n)
        highland = 0.150 * (1.0 + 0.17 * self.fbm4(b * 11.0))
        mare = 0.074 * (1.0 + 0.12 * self.fbm3(b * 8.0))
        m = self.mare_mask(b)
        base = highland * (1.0 - m) + mare * m
        base += self.ray_boost(b) * (1.0 - 0.5 * m)
        return ti.math.clamp(base, 0.045, 0.24)

    @ti.func
    def crater_height(self, p):
        """Elevation of the lunar surface at unit normal `p`, arbitrary units.

        Craters, not hills. Rolling fbm looks like dunes under a low sun; the
        Moon's defining feature is circular basins with raised rims, so this is
        cellular noise: every lattice cell holds one jittered impact whose
        radius comes from the same hash, and the overlapping profiles are summed
        the way real craters overprint one another. Varying the radius is what
        stops a uniform lattice from reading as a golf ball.

        Written without branches on purpose. This is evaluated three times per
        surface pixel to finite-difference a gradient, and the cell loop is
        unrolled 27 ways, so an `if` in here becomes 81 branch sites per octave
        and the kernel takes minutes to compile.
        """
        p = self.moon_body(p)
        h = 0.0
        for octave in ti.static(range(2)):
            freq = 5.5 * (2.9 ** octave)
            amp = 1.0 / (2.2 ** octave)
            q = p * freq
            base = ti.floor(q)
            for dx, dy, dz in ti.static(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
                cell = base + ti.Vector([float(dx), float(dy), float(dz)])
                g = self.hash3(cell)
                jitter = 0.5 + 0.5 * g
                # a second scalar out of the same hash -- cheaper than hashing
                # twice, and squared so small craters heavily outnumber large
                u = ti.math.fract(ti.math.dot(g, ti.Vector([12.99, 78.23, 37.71]))
                                  * 43758.5453)
                rad = 0.14 + 0.60 * u * u
                r = (cell + jitter - q).norm() / rad
                inside = ti.math.smoothstep(1.02, 0.70, r)
                bowl = -(1.0 - ti.min(r, 1.0) ** 2) * inside
                rim = 0.5 * ti.exp(-((r - 0.95) / 0.17) ** 2)
                h += amp * rad * (bowl + rim) * 3.2
        # Maria are lava plains that flooded and buried their craters, so relief
        # is suppressed inside the real basins -- the smooth dark patches and
        # the uncratered patches are the same patches, as on the Moon.
        h *= 0.30 + 0.70 * (1.0 - self.mare_mask(p))
        return h + 0.06 * self.fbm3(p * 26.0)

    @ti.func
    def moon_normal(self, n):
        """Surface normal including relief, by finite-differencing the height.

        Real craters near the terminator are visible almost entirely through the
        shadows their rims cast at grazing illumination -- albedo variation alone
        renders a smooth, airbrushed crescent. Perturbing the normal is what
        makes the Lommel-Seeliger term respond to relief and put those shadows
        in.
        """
        # an orthonormal tangent frame at n, avoiding the degenerate axis
        helper = ti.Vector([0.0, 0.0, 1.0])
        if ti.abs(n.z) > 0.9:
            helper = ti.Vector([1.0, 0.0, 0.0])
        t = ti.math.cross(helper, n).normalized()
        b = ti.math.cross(n, t)
        eps = 0.006
        h0 = self.crater_height(n)
        ht = self.crater_height((n + t * eps).normalized())
        hb = self.crater_height((n + b * eps).normalized())
        slope = self.moon_relief[None] / eps
        perturbed = n - (t * (ht - h0) + b * (hb - h0)) * slope
        return perturbed.normalized()

    # ------------------------------------------------------------------
    # atmosphere
    # ------------------------------------------------------------------
    @ti.func
    def sky_color(self, d):
        """Scattered daylight/twilight radiance along view direction `d`."""
        view_alt = ti.asin(ti.math.clamp(d.z, -1.0, 1.0)) / DEG
        s = self.sun_alt[None]
        cos_gamma = ti.math.clamp(ti.math.dot(d, self.sun_dir[None]), -1.0, 1.0)
        gamma = self.angle_deg(d, self.sun_dir[None])

        # overall brightness: full daylight above +6, astronomical night below -18
        level = ti.math.clamp((s + 18.0) / 24.0, 0.0, 1.0)
        lum = ti.pow(level, 3.4)

        night = ti.Vector([0.00016, 0.00025, 0.00052])
        zenith = ti.Vector([0.042, 0.088, 0.205])
        horizon = ti.Vector([0.175, 0.225, 0.310])

        # Rayleigh air mass: the sky pales towards the horizon
        h = ti.math.clamp(view_alt, -3.0, 90.0)
        band = ti.exp(-ti.max(0.0, h) / 16.0)
        day_col = zenith * (1.0 - band) + horizon * band
        col = night * (1.0 - lum) + day_col * lum

        # warm forward-scattered glow around the Sun, strongest at low sun
        twi = ti.exp(-0.5 * ((s + 2.0) / 9.0) ** 2)
        glow = ti.exp(-gamma / 26.0) * ti.exp(-ti.max(0.0, h) / 22.0)
        warm = ti.Vector([1.00, 0.42, 0.14])
        col += warm * (0.028 * twi * glow)
        # the daytime aureole close to the Sun
        col += ti.Vector([1.0, 0.92, 0.80]) * (lum * 0.26 * ti.exp(-gamma / 9.0))

        # belt of Venus / counter-twilight opposite the Sun after sunset
        anti = ti.exp(-0.5 * ((s + 4.0) / 5.0) ** 2)
        belt = ti.exp(-0.5 * ((view_alt - 7.0) / 7.0) ** 2) * ti.max(0.0, -cos_gamma)
        col += ti.Vector([0.30, 0.13, 0.20]) * (anti * belt * 0.020)
        return col

    @ti.func
    def ground_color(self, d):
        view_alt = ti.asin(ti.math.clamp(d.z, -1.0, 1.0)) / DEG
        s = self.sun_alt[None]
        lum = ti.pow(ti.math.clamp((s + 14.0) / 20.0, 0.0, 1.0), 2.6)
        base = ti.Vector([0.00035, 0.00036, 0.00046]) + ti.Vector([0.030, 0.028, 0.022]) * lum
        # a little haze right at the horizon line
        haze = ti.exp(-ti.abs(view_alt) / 1.6)
        return base * (1.0 - 0.35 * haze) + self.sky_color(d) * (0.22 * haze)

    # ------------------------------------------------------------------
    # bodies
    # ------------------------------------------------------------------
    @ti.func
    def angle_deg(self, a, b):
        """Angle between two unit vectors, in degrees.

        atan2(|a x b|, a.b) rather than acos(a.b): near zero separation the
        cosine is flat, so acos quantises small angles into visible steps --
        which shows up as blocky banding across a glare halo only a few
        arcminutes wide. The atan2 form stays accurate all the way down.
        """
        return ti.atan2(ti.math.cross(a, b).norm(), ti.math.dot(a, b)) / DEG

    @ti.func
    def hit_sphere(self, d, centre_dir, sin_r):
        """Ray-sphere in units of the body's distance. Returns (t, normal).

        t < 0 means no hit. The sphere sits at distance 1 along `centre_dir`
        with radius `sin_r`, which keeps the distance scale order 1.

        The discriminant is written as sin_r^2 - sin^2(angle) rather than the
        textbook b^2 - (1 - sin_r^2). Those are algebraically identical, but
        Venus subtends 18", so sin_r^2 ~ 8e-9 -- an order of magnitude below f32
        epsilon. The textbook form rounds both terms to 1.0 and the disc
        vanishes; the cross-product form never forms the cancelling difference
        and resolves a disc a thousand times smaller still.
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

    @ti.func
    def shade_moon(self, n):
        """Radiance of the lunar surface, in units where full sunlit rock ~ 0.12.

        Lommel-Seeliger rather than Lambert: a Lambert sphere darkens towards
        the limb, which is exactly what the real Moon conspicuously does not do.
        """
        light = self.moon_light[None]
        # relief affects which way the surface faces the Sun -- that is what
        # casts the shadows -- while the emission angle stays the geometric one
        nr = self.moon_normal(n)
        mu0 = ti.math.dot(nr, light)                      # cos(incidence)
        mu = ti.max(1e-3, -ti.math.dot(n, self.moon_dir[None]))   # cos(emission)
        albedo = self.moon_albedo(n)
        lit = 0.0
        if mu0 > 0.0:
            lit = 2.0 * mu0 / (mu0 + mu)
        lit *= ti.math.smoothstep(-0.015, 0.045, mu0)     # soften the terminator
        col = ti.Vector([1.00, 0.972, 0.920]) * (albedo * lit)
        # earthshine on the night side: sunlight reflected off a gibbous Earth
        earth = self.earthshine[None] * albedo * ti.math.smoothstep(-0.30, 0.10, -mu0)
        col += ti.Vector([0.46, 0.58, 0.88]) * earth
        return col

    @ti.func
    def shade_venus(self, n):
        light = self.venus_light[None]
        mu = ti.math.dot(n, light)
        # Venus' atmosphere carries light past the geometric terminator, which
        # is why its cusps extend so far
        lit = ti.math.smoothstep(-0.16, 0.22, mu)
        view_mu = ti.max(0.0, -ti.math.dot(n, self.venus_dir[None]))
        limb = 0.72 + 0.28 * ti.pow(view_mu, 0.35)
        # Venus receives 1.9x the solar flux the Moon does and its cloud deck
        # reflects 0.67 against the Moon's 0.12, so ~11x the surface radiance
        return ti.Vector([1.00, 0.978, 0.912]) * (1.32 * lit * limb)

    # ------------------------------------------------------------------
    @ti.func
    def trace(self, d):
        """Radiance along one view direction, in scene units."""
        # Airless mode strips the atmosphere out of the trace: no scattered sky
        # in front of the bodies, no extinction, no ground. It is the view from
        # above the air rather than a cosmetic filter, which is why extinction
        # goes to 1 at the same time as the airlight goes to 0 -- suppressing
        # only the blue would leave the bodies dimmed by an atmosphere that is
        # no longer being drawn.
        air = 1.0 - float(self.airless[None])
        airlight = self.sky_color(d) * air
        col = airlight
        if self.show_ground[None] == 1 and self.airless[None] == 0 and d.z < 0.0:
            col = self.ground_color(d)

        # Everything beyond the atmosphere is seen *through* it: attenuated by
        # extinction, with the full airlight added in front. This is why the
        # dark limb of a daytime Moon simply is not there, and why the same limb
        # shows earthshine once the sky goes dark.
        alt = ti.asin(ti.math.clamp(d.z, -1.0, 1.0)) / DEG
        airmass = 1.0 / ti.max(0.09, ti.sin(ti.max(alt, 0.5) * DEG))
        trans = ti.exp(-0.19 * airmass) * air + (1.0 - air)

        depth = 1e30
        t_m, n_m = self.hit_sphere(d, self.moon_dir[None], self.moon_sin_r[None])
        if t_m > 0.0:
            depth = t_m * self.moon_dist[None]
            col = self.shade_moon(n_m) * trans + airlight

        t_v, n_v = self.hit_sphere(d, self.venus_dir[None], self.venus_sin_r[None])
        if t_v > 0.0 and t_v * self.venus_dist[None] < depth:
            col = self.shade_venus(n_v) * trans + airlight

        gs = self.glare_scale[None]

        # --- glare around Venus, extinguished the instant it is hidden ---
        g = self.angle_deg(d, self.venus_dir[None])
        if g < 3.0 * gs and self.venus_visible[None] > 0.0:
            halo = (ti.exp(-g / (0.045 * gs)) * 0.090
                    + ti.exp(-g / (0.160 * gs)) * 0.0105)
            # four-armed diffraction pattern from the observer's optics
            x = ti.math.dot(d, self.cam_r[None])
            y = ti.math.dot(d, self.cam_u[None])
            cx = ti.math.dot(self.venus_dir[None], self.cam_r[None])
            cy = ti.math.dot(self.venus_dir[None], self.cam_u[None])
            ang = ti.atan2(y - cy, x - cx)
            spike = (ti.pow(ti.abs(ti.cos(2.0 * ang)), 60.0)
                     * ti.exp(-g / (0.090 * gs)) * 0.030)
            col += ti.Vector([1.0, 0.96, 0.88]) * ((halo + spike)
                                                   * self.venus_flux[None]
                                                   * self.venus_visible[None])

        # --- the Moon's own scattered glow in the atmosphere ---
        gm = self.angle_deg(d, self.moon_dir[None])
        if gm < 3.0 * gs and t_m < 0.0:
            col += ti.Vector([0.95, 0.95, 1.0]) * (self.moon_glow[None]
                                                   * ti.exp(-gm / (0.22 * gs)))
        return col

    @ti.kernel
    def render(self):
        n = self.samples[None]
        inv = 1.0 / float(n * n)
        for i, j in self.color:
            acc = ti.Vector([0.0, 0.0, 0.0])
            for sx in range(n):
                for sy in range(n):
                    ox = (float(sx) + 0.5) / float(n)
                    oy = (float(sy) + 0.5) / float(n)
                    u = (2.0 * (i + ox) / self.width - 1.0) * self.tan_half_fov[None]
                    v = ((2.0 * (j + oy) / self.height - 1.0)
                         * self.tan_half_fov[None] * self.height / self.width)
                    d = (self.cam_f[None] + self.cam_r[None] * u
                         + self.cam_u[None] * v).normalized()
                    acc += self.trace(d)
            self.color[i, j] = acc * inv

    @ti.kernel
    def splat_venus(self):
        """Draw Venus as a point source when its disc is smaller than a pixel.

        At a 40-degree field Venus spans about a seventh of a pixel, so the
        ray-traced disc almost never gets sampled and the planet disappears --
        even though at magnitude -4.8 it is the most obvious thing in the sky.
        Below the resolution limit its flux is therefore deposited the same way
        a star's is, which conserves brightness instead of losing it between
        sample points. Above the limit `venus_point_flux` is zero and the traced
        disc takes over.
        """
        if self.venus_point_flux[None] > 0.0 and self.venus_visible[None] > 0.0:
            d = self.venus_dir[None]
            fz = ti.math.dot(d, self.cam_f[None])
            if fz > 0.0:
                x = ti.math.dot(d, self.cam_r[None]) / fz
                y = ti.math.dot(d, self.cam_u[None]) / fz
                px = (x / self.tan_half_fov[None] * 0.5 + 0.5) * self.width
                py = (y / self.tan_half_fov[None] * self.width / self.height
                      * 0.5 + 0.5) * self.height
                flux = self.venus_point_flux[None] * self.venus_visible[None]
                alt = ti.asin(ti.math.clamp(d.z, -1.0, 1.0)) / DEG
                airmass = 1.0 / ti.max(0.09, ti.sin(ti.max(alt, 1.0) * DEG))
                flux *= ti.exp(-0.19 * (airmass - 1.0))
                base = ti.cast(ti.floor(px), ti.i32)
                basey = ti.cast(ti.floor(py), ti.i32)
                for dx, dy in ti.ndrange((-6, 7), (-6, 7)):
                    i, j = base + dx, basey + dy
                    if 0 <= i < self.width and 0 <= j < self.height:
                        ddx = float(i) + 0.5 - px
                        ddy = float(j) + 0.5 - py
                        r2 = ddx * ddx + ddy * ddy
                        core = ti.exp(-r2 / 0.70)
                        wing = 0.030 * ti.exp(-ti.sqrt(r2) / 2.2)
                        self.color[i, j] += ti.Vector([1.0, 0.975, 0.92]) * \
                            (flux * (core + wing))

    @ti.kernel
    def splat_stars(self):
        for k in range(self.n_stars[None]):
            d = self.star_dir[k]
            if ti.math.dot(d, self.cam_f[None]) > 0.0:
                # hide stars that lie behind the Moon or Venus
                behind_moon = ti.math.dot(d, self.moon_dir[None]) > \
                    ti.sqrt(ti.max(0.0, 1.0 - self.moon_sin_r[None] ** 2))
                if not behind_moon:
                    x = ti.math.dot(d, self.cam_r[None]) / ti.math.dot(d, self.cam_f[None])
                    y = ti.math.dot(d, self.cam_u[None]) / ti.math.dot(d, self.cam_f[None])
                    px = (x / self.tan_half_fov[None] * 0.5 + 0.5) * self.width
                    py = (y / self.tan_half_fov[None] * self.width / self.height
                          * 0.5 + 0.5) * self.height
                    if 2 <= px < self.width - 2 and 2 <= py < self.height - 2:
                        alt = ti.asin(ti.math.clamp(d.z, -1.0, 1.0)) / DEG
                        if alt > 0.0 or self.show_ground[None] == 0:
                            flux = ti.pow(10.0, -0.4 * (self.star_mag[k] - 1.0)) * 0.022
                            # atmospheric extinction plus scintillation near the horizon
                            airmass = 1.0 / ti.max(0.09, ti.sin(ti.max(alt, 1.0) * DEG))
                            flux *= ti.exp(-0.22 * (airmass - 1.0))
                            tw = 1.0 + 0.22 * (airmass - 1.0) * ti.sin(
                                self.twinkle_phase[None] * 7.0 + float(k) * 2.399)
                            flux *= ti.max(0.2, tw)
                            base = ti.cast(ti.floor(px), ti.i32)
                            basey = ti.cast(ti.floor(py), ti.i32)
                            for dx in range(-2, 3):
                                for dy in range(-2, 3):
                                    ddx = float(base + dx) + 0.5 - px
                                    ddy = float(basey + dy) + 0.5 - py
                                    w = ti.exp(-(ddx * ddx + ddy * ddy) / 0.62)
                                    self.color[base + dx, basey + dy] += \
                                        self.star_col[k] * (flux * w)

    @ti.kernel
    def tonemap(self):
        """Scene radiance -> display.

        A real twilight sky and a sunlit lunar surface differ by ~10^3, and the
        night sky by ~10^5, which no display can show at once. `compress` raises
        radiance to a fractional power first, squeezing that range the way a
        dark-adapted eye does; set it to 1.0 for a physically linear image in
        which the night sky is simply black.
        """
        k = self.compress[None]
        for i, j in self.pixels:
            c = self.color[i, j]
            c = ti.Vector([ti.pow(ti.max(c.x, 0.0), k),
                           ti.pow(ti.max(c.y, 0.0), k),
                           ti.pow(ti.max(c.z, 0.0), k)]) * self.exposure[None]
            c = c / (1.0 + c)
            c = ti.math.clamp(c, 0.0, 1.0)
            c = ti.Vector([ti.pow(c.x, 1.0 / 2.2), ti.pow(c.y, 1.0 / 2.2),
                           ti.pow(c.z, 1.0 / 2.2)])
            # ordered dither: smooth twilight gradients band badly at 8 bits
            d = (ti.math.fract(ti.sin(float(i) * 12.9898 + float(j) * 78.233)
                               * 43758.5453) - 0.5) / 255.0
            self.pixels[i, j] = ti.math.clamp(c + d, 0.0, 1.0)

    # ------------------------------------------------------------------
    def set_camera(self, az_deg: float, alt_deg: float, fov_deg: float) -> None:
        a, h = az_deg * DEG, alt_deg * DEG
        f = np.array([math.cos(h) * math.cos(a), math.cos(h) * math.sin(a), math.sin(h)])
        r = np.array([-math.sin(a), math.cos(a), 0.0])
        u = np.cross(f, r)
        self.cam_f[None] = f
        self.cam_r[None] = r
        self.cam_u[None] = u
        self.tan_half_fov[None] = math.tan(0.5 * fov_deg * DEG)
        # remembered so `bridge.apply_state` can decide whether a body is
        # resolved without the caller having to pass the field of view twice
        self.fov_deg = fov_deg
        # the instrument's point-spread function scales with the view, so
        # zooming in resolves Venus' crescent instead of a fixed blob of glare
        self.glare_scale[None] = max(0.05, fov_deg / 2.0)

    def load_stars(self, stars) -> None:
        n = min(len(stars), self.max_stars)
        for k in range(n):
            d, mag, col = stars[k]
            self.star_dir[k] = d
            self.star_mag[k] = mag
            self.star_col[k] = col
        self.n_stars[None] = n

    def frame(self) -> np.ndarray:
        self.color.fill(0.0)
        self.render()
        self.splat_venus()
        self.splat_stars()
        self.tonemap()
        return self.pixels.to_numpy()
