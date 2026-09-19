"""Turn a `SkyState` into the numbers the Taichi kernels want.

Kept separate from both the ephemeris and the kernels: the astronomy modules
know nothing about rendering, and the kernels know nothing about time.
"""
from __future__ import annotations

import math

import numpy as np

from vmsim.planets import RADIUS_KM
from vmsim.scene import Body, SkyState
from vmsim.stars import precessed_equatorial

DEG = math.pi / 180.0


def horizon_unit(body: Body, refracted: bool = True) -> np.ndarray:
    """Unit vector in the (north, east, up) horizon frame."""
    alt = (body.apparent_altitude_deg if refracted else body.altitude_deg) * DEG
    az = body.azimuth_deg * DEG
    return np.array([math.cos(alt) * math.cos(az),
                     math.cos(alt) * math.sin(az),
                     math.sin(alt)], dtype=np.float64)


def ecliptic_north_horizon(state) -> np.ndarray:
    """Ecliptic north pole as a unit vector in the (north, east, up) frame."""
    from vmsim.frames import ecl_to_equ, equ_to_altaz, mean_obliquity_deg, nutation

    t = state.instant.t
    eps = mean_obliquity_deg(t) + nutation(t)[1]
    equ = ecl_to_equ((0.0, 0.0, 1.0), eps)
    alt, az = equ_to_altaz(equ, state.site.latitude, state.local_sidereal_deg)
    a, h = az * DEG, alt * DEG
    return np.array([math.cos(h) * math.cos(a), math.cos(h) * math.sin(a),
                     math.sin(h)], dtype=np.float64)


def moon_body_frame(state) -> tuple[np.ndarray, np.ndarray]:
    """The Moon's (east, north) body axes in the horizon frame.

    Together with the sub-Earth direction -- which the shader already has as
    `moon_dir` -- these fix the Moon's orientation, so its markings stay put on
    the Moon instead of sliding across it as it crosses the sky.
    """
    sub_earth = -horizon_unit(state.moon)             # Moon -> observer
    north = ecliptic_north_horizon(state)
    # orthogonalise against the sub-Earth axis, then complete the triad
    north = north - sub_earth * float(np.dot(north, sub_earth))
    n = np.linalg.norm(north)
    north = north / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
    east = np.cross(north, sub_earth)
    return east / np.linalg.norm(east), north


def load_lunar_features(renderer) -> None:
    """Push the named nearside features into the renderer, once.

    The catalogue is static, so this is guarded -- `apply_state` runs every
    frame and there is no reason to re-upload eighteen vectors at 50 Hz.
    """
    if getattr(renderer, "_lunar_features_loaded", False):
        return
    from vmsim.moon import MARIA, RAY_CRATERS, selenographic_vector

    for k, (_name, lat, lon, radius) in enumerate(MARIA[:renderer.MAX_FEATURES]):
        renderer.mare_dir[k] = selenographic_vector(lat, lon)
        # a soft edge either side of the nominal rim
        renderer.mare_cos_in[k] = math.cos(radius * 0.80 * DEG)
        renderer.mare_cos_out[k] = math.cos(radius * 1.20 * DEG)
    renderer.n_mare[None] = min(len(MARIA), renderer.MAX_FEATURES)

    for k, (_name, lat, lon, radius, reach) in enumerate(RAY_CRATERS[:8]):
        renderer.ray_dir[k] = selenographic_vector(lat, lon)
        renderer.ray_cos[k] = math.cos(radius * DEG)
        renderer.ray_reach[k] = reach
    renderer.n_ray[None] = min(len(RAY_CRATERS), 8)
    renderer._lunar_features_loaded = True


def overlap_fraction(sep_deg: float, r_front_deg: float, r_back_deg: float) -> float:
    """Fraction of the back disc's area hidden by the front disc."""
    d, r0, r1 = sep_deg, r_front_deg, r_back_deg
    if d >= r0 + r1:
        return 0.0
    if d <= abs(r0 - r1):
        return 1.0 if r1 <= r0 else (r0 / r1) ** 2
    a0 = math.acos(max(-1.0, min(1.0, (d * d + r1 * r1 - r0 * r0) / (2 * d * r1))))
    a1 = math.acos(max(-1.0, min(1.0, (d * d + r0 * r0 - r1 * r1) / (2 * d * r0))))
    area = (r1 * r1 * (a0 - math.sin(2 * a0) / 2.0)
            + r0 * r0 * (a1 - math.sin(2 * a1) / 2.0))
    return min(1.0, area / (math.pi * r1 * r1))


def venus_visible_fraction(state: SkyState) -> float:
    if state.venus.distance_km < state.moon.distance_km:
        return 1.0
    hidden = overlap_fraction(state.separation_deg,
                              state.moon.angular_radius_deg,
                              state.venus.angular_radius_deg)
    return 1.0 - hidden


def sky_brightness_level(sun_alt_deg: float) -> float:
    """The same 0..1 daylight ramp the shader uses, for exposure control."""
    return max(0.0, min(1.0, (sun_alt_deg + 18.0) / 24.0)) ** 3.4


def apply_state(renderer, state: SkyState, exposure_bias: float = 1.0,
                show_ground: bool = True, twinkle: float = 0.0,
                compress: float = 1.0) -> None:
    """Push one sky state into the renderer's fields."""
    moon_u = horizon_unit(state.moon)
    venus_u = horizon_unit(state.venus)
    sun_u = horizon_unit(state.sun, refracted=False)

    moon_p = moon_u * state.moon.distance_km
    venus_p = venus_u * state.venus.distance_km
    sun_p = sun_u * state.sun.distance_km

    def unit(v):
        return v / np.linalg.norm(v)

    renderer.moon_dir[None] = moon_u
    renderer.moon_light[None] = unit(sun_p - moon_p)
    renderer.moon_sin_r[None] = math.sin(state.moon.angular_radius_deg * DEG)
    renderer.moon_dist[None] = state.moon.distance_km
    # Earth's phase seen from the Moon is the complement of the Moon's phase, so
    # a thin crescent Moon has a nearly full Earth overhead and the strongest
    # earthshine. Physically it is ~1e-4 of the sunlit surface; it is lifted here
    # for the same reason the night sky is (see `tonemap`), to about 1/50.
    renderer.earthshine[None] = 0.025 * (1.0 - state.moon.illuminated_fraction)

    renderer.venus_dir[None] = venus_u
    renderer.venus_light[None] = unit(sun_p - venus_p)
    renderer.venus_sin_r[None] = math.sin(state.venus.angular_radius_deg * DEG)
    renderer.venus_dist[None] = state.venus.distance_km

    vis = venus_visible_fraction(state)
    renderer.venus_visible[None] = vis

    # Below about a pixel the traced disc is unreliable, so hand Venus over to
    # the point-source path on the same magnitude scale the stars use.
    radius_px = (state.venus.angular_radius_deg
                 / max(renderer.fov_deg, 1e-6) * renderer.width)
    if radius_px < 1.2:
        fade = 1.0 - max(0.0, min(1.0, (radius_px - 0.6) / 0.6))
        renderer.venus_point_flux[None] = (
            10.0 ** (-0.4 * (state.venus.magnitude - 1.0)) * 0.022 * fade)
    else:
        renderer.venus_point_flux[None] = 0.0
    lum = sky_brightness_level(state.sun.apparent_altitude_deg)
    flux = 10.0 ** (-0.4 * (state.venus.magnitude + 4.8))
    renderer.venus_flux[None] = flux / (1.0 + 45.0 * lum)

    renderer.sun_dir[None] = sun_u
    load_lunar_features(renderer)
    east, north = moon_body_frame(state)
    renderer.moon_east[None] = east
    renderer.moon_north[None] = north
    renderer.sun_alt[None] = state.sun.apparent_altitude_deg
    renderer.show_ground[None] = 1 if show_ground else 0
    renderer.twinkle_phase[None] = twinkle
    renderer.compress[None] = compress

    # Auto-exposure keyed to the sky itself, so the picture stays readable from
    # full daylight through to a dark sky without the user touching anything.
    # `target` is the pre-shoulder value the sky should land on: chosen so the
    # night sky renders near 6% grey and a daylight sky near 45%.
    sky_ref = 0.00030 + 0.155 * lum
    # Daylight is exposed for the *subject*, not for the sky -- which is what a
    # photographer shooting the daytime Moon does, and why their pictures show a
    # deep blue sky rather than the pale wash your eye reports. A daytime Moon
    # really is low contrast: sunlit regolith runs about 4,900 cd/m2 against a
    # clear sky's few thousand, so the two are genuinely comparable and no
    # exposure can separate them by much. Placing the sky lower on the tone
    # curve keeps the little contrast there is in the steeper part of it, and
    # renders the blue as blue instead of white.
    target = 0.0024 + 0.285 * lum
    if int(renderer.airless[None]) == 1:
        # nothing to meter off, so hold the dark-sky exposure: sunlit regolith
        # then lands around mid-grey instead of being crushed or blown out
        renderer.exposure[None] = exposure_bias * 8.0
    else:
        renderer.exposure[None] = exposure_bias * target / (sky_ref ** compress)
    renderer.moon_glow[None] = 0.010 * state.moon.illuminated_fraction


def load_star_field(renderer, state: SkyState) -> None:
    """Precess the catalogue and convert it to horizon-frame unit vectors."""
    lat = state.site.latitude * DEG
    lst = state.local_sidereal_deg * DEG
    out = []
    for v, mag, col in precessed_equatorial(state.instant.t):
        ra = math.atan2(v[1], v[0])
        dec = math.asin(max(-1.0, min(1.0, v[2])))
        h = lst - ra
        sin_alt = (math.sin(dec) * math.sin(lat)
                   + math.cos(dec) * math.cos(lat) * math.cos(h))
        alt = math.asin(max(-1.0, min(1.0, sin_alt)))
        az = math.atan2(math.sin(h),
                        math.cos(h) * math.sin(lat) - math.tan(dec) * math.cos(lat))
        az += math.pi
        out.append((np.array([math.cos(alt) * math.cos(az),
                              math.cos(alt) * math.sin(az),
                              math.sin(alt)]), mag, np.array(col)))
    renderer.load_stars(out)


# ---------------------------------------------------------------------------
# heliocentric scene -> render.space
# ---------------------------------------------------------------------------
# Sidereal rotation periods in days. Negative means retrograde: the body turns
# the other way about its IAU north pole, which for Venus is the whole point.
ROTATION_DAYS = {"mercury": 58.646, "venus": -243.025, "earth": 0.9972696,
                 "mars": 1.0259565}

# IAU WGCCRE (2015) pole orientations: J2000 equatorial right ascension and
# declination of each body's north pole, in degrees.
#
# These are converted to the ecliptic frame below rather than written out as
# vectors by hand. The hand-written versions this replaces were wrong for three
# of the four bodies -- the Earth's had the sign of its y component flipped,
# which tilted the planet 47 degrees off true while still looking entirely
# plausible on screen, and Venus' pointed at the south ecliptic pole while also
# carrying a negative period, two negatives that multiplied into a Venus
# rotating *prograde*. Deriving them from published numbers makes the claim
# checkable instead of trusting a typed vector.
IAU_POLE_RADEC = {
    "mercury": (281.0103, 61.4155),
    "venus": (272.76, 67.16),
    "earth": (0.0, 90.0),            # the north celestial pole, by definition
    "mars": (317.269, 54.432),
}


def _ecliptic_pole(ra_deg: float, dec_deg: float):
    """IAU equatorial pole -> unit vector in the J2000 ecliptic frame.

    Uses the J2000 obliquity because that is the frame the IAU values are
    quoted in. The film renders in the ecliptic of date, but over the ten
    months it covers the two frames differ by a few arcseconds of precession --
    far below the angular size of anything on screen.
    """
    import math

    from vmsim.frames import equ_to_ecl, from_spherical, mean_obliquity_deg

    v = from_spherical(math.radians(ra_deg), math.radians(dec_deg), 1.0)
    e = equ_to_ecl(v, mean_obliquity_deg(0.0))
    n = math.sqrt(sum(c * c for c in e))
    return tuple(c / n for c in e)


POLE = {name: _ecliptic_pole(*radec) for name, radec in IAU_POLE_RADEC.items()}


def radius_au(name: str) -> float:
    """Body radius in AU, for `SpaceRenderer.add_body`."""
    from vmsim.planets import AU_KM, RADIUS_KM
    return RADIUS_KM[name] / AU_KM


def spin_phase(name: str, jd: float) -> float:
    """Rotation angle in radians at `jd`, from the true sidereal period.

    The zero point is arbitrary -- no attempt is made to put a real meridian in
    a real place, because at these scales no surface feature is identifiable.
    What is real is the rate and the direction.
    """
    import math
    return 2.0 * math.pi * ((jd - 2451545.0) / ROTATION_DAYS[name] % 1.0)


def point_flux(magnitude: float, reference: float = -4.0,
               scale: float = 3.0) -> float:
    """Flux for `SpaceRenderer.add_glare`, from an apparent magnitude.

    Pogson's ratio against a reference magnitude, so a brightening Venus really
    does splat more light rather than being drawn at a fixed size.
    """
    return scale * 10.0 ** (-0.4 * (magnitude - reference))


def heliocentric_scene(renderer, instant, *, planet_scale: float = 1.0,
                       sun_scale: float = 1.0, bodies=("venus", "earth"),
                       include_sun: bool = True, sun_gain: float = 1.0):
    """Populate `renderer` with the inner solar system at `instant`.

    Returns the dict of heliocentric positions used, so a caller that wants to
    annotate the frame projects exactly the points that were rendered rather
    than recomputing them and risking a drift.

    `planet_scale` and `sun_scale` inflate radii only. Positions, distances and
    light directions are always true, and a caller that passes a scale other
    than 1.0 owes the viewer a note on screen saying so.
    """
    from vmsim.planets import heliocentric_of_date

    from .space import EARTH, ROCKY, SUN, VENUS

    kinds = {"venus": VENUS, "earth": EARTH, "mercury": ROCKY, "mars": ROCKY}
    tints = {"venus": (1.00, 0.978, 0.912), "earth": (1.0, 1.0, 1.0),
             "mercury": (0.80, 0.76, 0.71), "mars": (0.86, 0.52, 0.36)}

    jde, jd = instant.jde, instant.jd_ut
    positions = {"sun": (0.0, 0.0, 0.0)}
    renderer.clear_bodies()
    if include_sun:
        renderer.add_body((0.0, 0.0, 0.0), radius_au("sun"), SUN,
                          scale=sun_scale, gain=sun_gain)
    for name in bodies:
        pos = heliocentric_of_date(name, jde) if name in ("earth", "venus") \
            else _kepler_pos(name, instant)
        positions[name] = pos
        renderer.add_body(pos, radius_au(name), kinds[name], scale=planet_scale,
                          tint=tints[name], axis=POLE[name],
                          spin=spin_phase(name, jd))
    return positions


def _kepler_pos(name: str, instant):
    """Mercury and Mars, which exist in these frames only as context."""
    from vmsim.planets import kepler_heliocentric_j2000
    return kepler_heliocentric_j2000(name, instant.t)


def orbit_paths(instant, names=("venus", "earth"), samples: int = 1800,
                colours=None):
    """Orbit ellipses as (points, colour) pairs for `SpaceRenderer.load_paths`."""
    from vmsim.planets import orbit_ellipse_j2000
    default = {"venus": (0.62, 0.50, 0.30), "earth": (0.24, 0.40, 0.64),
               "mercury": (0.26, 0.24, 0.22), "mars": (0.32, 0.19, 0.13)}
    colours = colours or default
    return [(orbit_ellipse_j2000(n, instant.t, samples), colours[n]) for n in names]
