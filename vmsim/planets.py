"""Planet positions.

Venus and the Earth come from the truncated VSOP87D series in `vsop87` (sub-
arcsecond); the other planets come from Standish's Keplerian elements, which are
only ever used for the schematic orrery view where arcseconds are meaningless.
"""
from __future__ import annotations

import math

from . import vsop87
from .frames import (Vec3, angle_between, matvec, rot_x, rot_z, vadd, vnorm,
                     vscale, vsub, vunit)

AU_KM = 149597870.7
C_AU_PER_DAY = 173.144632674240
DEG = math.pi / 180.0

_ELEMENTS = {
    "mercury": ((0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593),
                (0.00000037, 0.00001906, -0.00594749, 149472.67411175, 0.16047689, -0.12534081)),
    "venus": ((0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255),
              (0.00000390, -0.00004107, -0.00078890, 58517.81538729, 0.00268329, -0.27769418)),
    "earth": ((1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0),
              (0.00000562, -0.00004392, -0.01294668, 35999.37244981, 0.32327364, 0.0)),
    "mars": ((1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891),
             (0.00001847, 0.00007882, -0.00813131, 19140.30268499, 0.44441088, -0.29257343)),
    "jupiter": ((5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909),
                (-0.00011607, -0.00013253, -0.00183714, 3034.74612775, 0.21252668, 0.20469106)),
    "saturn": ((9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448),
               (-0.00125060, -0.00050991, 0.00193609, 1222.49362201, -0.41897216, -0.28867794)),
}

RADIUS_KM = {"mercury": 2439.7, "venus": 6051.8, "earth": 6378.137, "moon": 1737.4,
             "mars": 3389.5, "jupiter": 69911.0, "saturn": 58232.0, "sun": 695700.0}

ORBIT_COLOR = {"mercury": (0.72, 0.68, 0.62), "venus": (1.00, 0.87, 0.62),
               "earth": (0.42, 0.68, 1.00), "mars": (0.95, 0.52, 0.34),
               "jupiter": (0.90, 0.78, 0.60), "saturn": (0.94, 0.86, 0.66)}


# --------------------------------------------------------------------------
# schematic orrery positions (Keplerian, J2000 ecliptic)
# --------------------------------------------------------------------------
def _solve_kepler(m_rad: float, e: float) -> float:
    ea = m_rad + e * math.sin(m_rad)
    for _ in range(30):
        dm = m_rad - (ea - e * math.sin(ea))
        de = dm / (1.0 - e * math.cos(ea))
        ea += de
        if abs(de) < 1e-13:
            break
    return ea


def kepler_heliocentric_j2000(name: str, t: float) -> Vec3:
    """Heliocentric J2000 ecliptic position in AU. t = centuries TT."""
    (a0, e0, i0, l0, w0, o0), (da, de, di, dl, dw, do) = _ELEMENTS[name]
    a, e = a0 + da * t, e0 + de * t
    inc, node = (i0 + di * t) * DEG, (o0 + do * t) * DEG
    lam, peri = l0 + dl * t, w0 + dw * t
    m = math.radians(((lam - peri + 180.0) % 360.0) - 180.0)
    ea = _solve_kepler(m, e)
    v = (a * (math.cos(ea) - e), a * math.sqrt(1.0 - e * e) * math.sin(ea), 0.0)
    v = matvec(rot_z(peri * DEG - node), v)
    v = matvec(rot_x(inc), v)
    return matvec(rot_z(node), v)


def orbit_ellipse_j2000(name: str, t: float, samples: int = 256) -> list[Vec3]:
    """Points tracing the planet's orbit, for drawing the orrery."""
    (a0, e0, i0, _l0, w0, o0), (da, de, di, _dl, dw, do) = _ELEMENTS[name]
    a, e = a0 + da * t, e0 + de * t
    inc, node = (i0 + di * t) * DEG, (o0 + do * t) * DEG
    peri = w0 + dw * t
    pts = []
    for k in range(samples):
        ea = 2.0 * math.pi * k / samples
        v = (a * (math.cos(ea) - e), a * math.sqrt(1.0 - e * e) * math.sin(ea), 0.0)
        v = matvec(rot_z(peri * DEG - node), v)
        v = matvec(rot_x(inc), v)
        pts.append(matvec(rot_z(node), v))
    return pts


# --------------------------------------------------------------------------
# precise positions (VSOP87D, ecliptic of date)
# --------------------------------------------------------------------------
def heliocentric_of_date(body: str, jde: float) -> Vec3:
    """Earth or Venus, heliocentric rectangular, ecliptic of date, AU."""
    return vsop87.rectangular(body, jde, fk5=False)


def earth_velocity_of_date(jde: float, h_days: float = 0.5) -> Vec3:
    """Earth's heliocentric velocity in AU/day, by central difference."""
    p1 = heliocentric_of_date("earth", jde + h_days)
    p0 = heliocentric_of_date("earth", jde - h_days)
    return vscale(vsub(p1, p0), 1.0 / (2.0 * h_days))


def geocentric_of_date(body: str, jde: float) -> tuple[Vec3, float, Vec3]:
    """Light-time corrected geocentric vector, ecliptic of date, AU.

    Returns (earth->body vector, light time in days, body's heliocentric vector
    at the emitting instant) -- the last is what the phase angle needs.
    """
    earth = heliocentric_of_date("earth", jde)
    tau = 0.0
    vec = (0.0, 0.0, 0.0)
    target = earth
    for _ in range(4):
        target = heliocentric_of_date(body, jde - tau)
        vec = vsub(target, earth)
        new_tau = vnorm(vec) / C_AU_PER_DAY
        if abs(new_tau - tau) < 1e-11:
            tau = new_tau
            break
        tau = new_tau
    return vec, tau, target


def sun_geocentric_of_date(jde: float) -> Vec3:
    """Earth -> Sun vector, ecliptic of date, AU, light-time corrected."""
    earth = heliocentric_of_date("earth", jde)
    tau = vnorm(earth) / C_AU_PER_DAY
    return vscale(heliocentric_of_date("earth", jde - tau), -1.0)


def apply_aberration(vec: Vec3, earth_velocity_au_day: Vec3) -> Vec3:
    """Annual aberration, first order (0.001" residual -- below our budget)."""
    d = vnorm(vec)
    u = vunit(vec)
    beta = vscale(earth_velocity_au_day, 1.0 / C_AU_PER_DAY)
    return vscale(vunit(vadd(u, beta)), d)


def phase_angle(sun_to_planet: Vec3, earth_to_planet: Vec3) -> float:
    """Sun-planet-Earth angle in degrees.

    Measured at the planet, so it is the angle between (planet->Sun) and
    (planet->Earth); negating both inputs leaves the angle unchanged, which is
    why the heliocentric and geocentric vectors can be passed in directly.
    """
    return math.degrees(angle_between(sun_to_planet, earth_to_planet))


def illuminated_fraction(phase_angle_deg: float) -> float:
    return (1.0 + math.cos(math.radians(phase_angle_deg))) / 2.0


def venus_magnitude(r_au: float, delta_au: float, phase_angle_deg: float) -> float:
    """Apparent visual magnitude of Venus (Astronomical Almanac / Hilton 2005)."""
    i = min(phase_angle_deg, 163.6)
    return (-4.384 - 1.044e-3 * i + 3.687e-4 * i ** 2 - 2.814e-6 * i ** 3
            + 8.938e-9 * i ** 4 + 5.0 * math.log10(r_au * delta_au))


def angular_radius_deg(radius_km: float, distance_au: float) -> float:
    return math.degrees(math.asin(radius_km / (distance_au * AU_KM)))
