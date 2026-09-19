"""Assemble a complete observable sky state for one instant and one site.

This is the single place where the ephemeris modules combine and where
geocentric quantities become topocentric. Everything the renderers and the event
search need comes out of `sky_state()`.

Frame chain, following Meeus ch. 33: VSOP87D and the lunar theory both give
ecliptic coordinates of date, so the planets need no precession matrix. To each
geocentric ecliptic position we apply the FK5 correction, then nutation in
longitude, then annual aberration, then rotate to the true equator of date with
the true obliquity; finally we subtract the observer's geocentric position.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from . import planets, vsop87
from .frames import (Vec3, angle_between, ecl_to_equ, equ_to_altaz,
                     from_spherical, mean_obliquity_deg, nutation,
                     position_angle, refracted_altitude_deg, spherical, vadd,
                     vnorm, vscale, vsub)
from .moon import MOON_RADIUS_KM, geocentric_ecliptic
from .observer import Site
from .planets import AU_KM
from .timescale import Instant


@dataclass
class Body:
    name: str
    vec_topo: Vec3            # topocentric equatorial-of-date, km
    ra_deg: float
    dec_deg: float
    distance_km: float
    angular_radius_deg: float
    altitude_deg: float
    apparent_altitude_deg: float
    azimuth_deg: float
    illuminated_fraction: float = 1.0
    phase_angle_deg: float = 0.0
    magnitude: float = 0.0
    bright_limb_pa_deg: float = 0.0

    @property
    def angular_diameter_arcsec(self) -> float:
        return 2.0 * self.angular_radius_deg * 3600.0


@dataclass
class SkyState:
    instant: Instant
    site: Site
    moon: Body
    venus: Body
    sun: Body
    separation_deg: float
    separation_arcsec: float
    is_occulted: bool
    venus_moon_pa_deg: float      # position angle of Venus from the Moon's centre
    elongation_deg: float         # Venus' elongation from the Sun
    local_sidereal_deg: float

    @property
    def limb_distance_arcsec(self) -> float:
        """Venus' centre relative to the lunar limb; negative means hidden."""
        return (self.separation_deg - self.moon.angular_radius_deg) * 3600.0


def _make_body(name: str, vec_topo: Vec3, radius_km: float,
               site: Site, lst_deg: float) -> Body:
    ra, dec, dist = spherical(vec_topo)
    alt, az = equ_to_altaz(vec_topo, site.latitude, lst_deg)
    return Body(name=name, vec_topo=vec_topo,
                ra_deg=math.degrees(ra), dec_deg=math.degrees(dec),
                distance_km=dist,
                angular_radius_deg=math.degrees(math.asin(min(1.0, radius_km / dist))),
                altitude_deg=alt,
                apparent_altitude_deg=refracted_altitude_deg(alt),
                azimuth_deg=az)


def _apparent_equatorial_km(vec_ecl_au: Vec3, dpsi: float, eps_true: float,
                            earth_vel: Vec3, jde: float, fk5: bool = True) -> Vec3:
    """Geocentric ecliptic-of-date AU vector -> apparent equatorial-of-date km."""
    vec = planets.apply_aberration(vec_ecl_au, earth_vel)
    lon, lat, r = spherical(vec)
    if fk5:
        lon, lat = vsop87.to_fk5(lon, lat, jde)
    lon += math.radians(dpsi)
    v = from_spherical(lon, lat, r * AU_KM)
    return ecl_to_equ(v, eps_true)


def geocentric_bodies(instant: Instant):
    """Apparent geocentric equatorial-of-date vectors (km) plus phase inputs.

    Split out from `sky_state` because the occultation footprint solver needs
    the geocentric vectors thousands of times per site-independent step.
    """
    jde = instant.jde
    t = instant.t
    dpsi, deps = nutation(t)
    eps_true = mean_obliquity_deg(t) + deps

    earth_vel = planets.earth_velocity_of_date(jde)
    venus_ecl, venus_lt, venus_helio = planets.geocentric_of_date("venus", jde)
    sun_ecl = planets.sun_geocentric_of_date(jde)

    venus_equ = _apparent_equatorial_km(venus_ecl, dpsi, eps_true, earth_vel, jde)
    sun_equ = _apparent_equatorial_km(sun_ecl, dpsi, eps_true, earth_vel, jde, fk5=False)

    lon, lat, dist = geocentric_ecliptic(t)
    moon_equ = ecl_to_equ(from_spherical(math.radians(lon + dpsi),
                                         math.radians(lat), dist), eps_true)
    return {
        "moon": moon_equ, "venus": venus_equ, "sun": sun_equ,
        "venus_helio": venus_helio, "venus_geo_ecl": venus_ecl,
        "dpsi": dpsi, "eps_true": eps_true, "venus_light_time": venus_lt,
    }


def sky_state(instant: Instant, site: Site) -> SkyState:
    g = geocentric_bodies(instant)
    dpsi, eps_true = g["dpsi"], g["eps_true"]

    lst = site.local_sidereal_deg(instant.jd_ut,
                                  dpsi * math.cos(math.radians(eps_true)))
    obs = site.position_equatorial_km(lst)

    moon = _make_body("Moon", vsub(g["moon"], obs), MOON_RADIUS_KM, site, lst)
    venus = _make_body("Venus", vsub(g["venus"], obs),
                       planets.RADIUS_KM["venus"], site, lst)
    sun = _make_body("Sun", vsub(g["sun"], obs), planets.RADIUS_KM["sun"], site, lst)

    venus_r = vnorm(g["venus_helio"])
    venus_delta = vnorm(g["venus_geo_ecl"])
    venus.phase_angle_deg = planets.phase_angle(g["venus_helio"], g["venus_geo_ecl"])
    venus.illuminated_fraction = planets.illuminated_fraction(venus.phase_angle_deg)
    venus.magnitude = planets.venus_magnitude(venus_r, venus_delta, venus.phase_angle_deg)
    venus.bright_limb_pa_deg = position_angle(venus.vec_topo, sun.vec_topo)

    moon.phase_angle_deg = math.degrees(angle_between(
        vscale(moon.vec_topo, -1.0), vsub(sun.vec_topo, moon.vec_topo)))
    moon.illuminated_fraction = planets.illuminated_fraction(moon.phase_angle_deg)
    # Allen's empirical phase law for the Moon's integrated magnitude
    i = moon.phase_angle_deg
    moon.magnitude = -12.73 + 0.026 * abs(i) + 4.0e-9 * i ** 4
    moon.bright_limb_pa_deg = position_angle(moon.vec_topo, sun.vec_topo)

    sep = math.degrees(angle_between(moon.vec_topo, venus.vec_topo))
    occulted = sep < moon.angular_radius_deg and venus.distance_km > moon.distance_km

    return SkyState(
        instant=instant, site=site, moon=moon, venus=venus, sun=sun,
        separation_deg=sep, separation_arcsec=sep * 3600.0, is_occulted=occulted,
        venus_moon_pa_deg=position_angle(moon.vec_topo, venus.vec_topo),
        elongation_deg=math.degrees(angle_between(venus.vec_topo, sun.vec_topo)),
        local_sidereal_deg=lst,
    )


# --------------------------------------------------------------------------
# site-independent Venus geometry
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class VenusGeometry:
    """Everything the Sun-Venus-Earth triangle determines, for one instant.

    Geocentric rather than topocentric: the parallax that matters so much to an
    occultation shifts Venus by at most 30" from the Earth's centre, which is
    below a pixel in any view wide enough to hold the orbit. Dropping the site
    makes the whole apparition solvable without picking somewhere to stand.
    """
    instant: Instant
    sun_au: Vec3                  # heliocentric Earth->Sun is just -earth_au
    earth_au: Vec3                # heliocentric ecliptic of date
    venus_au: Vec3                # heliocentric, at the instant light left it
    elongation_deg: float         # Sun-Earth-Venus, measured at the Earth
    phase_angle_deg: float        # Sun-Venus-Earth, measured at Venus
    illuminated_fraction: float
    magnitude: float
    r_au: float                   # Sun to Venus
    delta_au: float               # Earth to Venus
    angular_diameter_arcsec: float

    @property
    def is_superior_side(self) -> bool:
        """True when Venus is beyond the Sun rather than between us and it."""
        return self.delta_au > vnorm(self.earth_au)


def venus_geometry(instant: Instant) -> VenusGeometry:
    """Solve the Sun-Venus-Earth triangle. No observer required."""
    jde = instant.jde
    earth = planets.heliocentric_of_date("earth", jde)
    venus_geo, _lt, venus_helio = planets.geocentric_of_date("venus", jde)
    sun_geo = planets.sun_geocentric_of_date(jde)

    r = vnorm(venus_helio)
    delta = vnorm(venus_geo)
    phase = planets.phase_angle(venus_helio, venus_geo)
    return VenusGeometry(
        instant=instant,
        sun_au=sun_geo,
        earth_au=earth,
        venus_au=venus_helio,
        elongation_deg=math.degrees(angle_between(venus_geo, sun_geo)),
        phase_angle_deg=phase,
        illuminated_fraction=planets.illuminated_fraction(phase),
        magnitude=planets.venus_magnitude(r, delta, phase),
        r_au=r,
        delta_au=delta,
        angular_diameter_arcsec=2.0 * 3600.0 * planets.angular_radius_deg(
            planets.RADIUS_KM["venus"], delta),
    )
