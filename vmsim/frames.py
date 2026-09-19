"""Reference frames: obliquity, nutation, precession, and coordinate conversions.

Convention used throughout the package:
  * vectors are plain 3-tuples of floats
  * "ecl" = ecliptic rectangular, "equ" = equatorial rectangular
  * "of date" means referred to the mean/true equinox and ecliptic of the epoch
"""
from __future__ import annotations

import math

Vec3 = tuple[float, float, float]

ARCSEC = 1.0 / 3600.0


# --------------------------------------------------------------------------
# small vector helpers (kept plain-Python: this runs once per frame, not per pixel)
# --------------------------------------------------------------------------
def vadd(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vsub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vscale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def vdot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vcross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def vnorm(a: Vec3) -> float:
    return math.sqrt(vdot(a, a))


def vunit(a: Vec3) -> Vec3:
    n = vnorm(a)
    return (a[0] / n, a[1] / n, a[2] / n)


def angle_between(a: Vec3, b: Vec3) -> float:
    """Robust angle between two vectors, in radians."""
    ua, ub = vunit(a), vunit(b)
    c = vdot(ua, ub)
    s = vnorm(vcross(ua, ub))
    return math.atan2(s, c)


def matvec(m, v: Vec3) -> Vec3:
    return (m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2])


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def transpose(m):
    return [[m[j][i] for j in range(3)] for i in range(3)]


def rot_x(theta: float):
    c, s = math.cos(theta), math.sin(theta)
    return [[1, 0, 0], [0, c, -s], [0, s, c]]


def rot_z(theta: float):
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]


def rot_y(theta: float):
    c, s = math.cos(theta), math.sin(theta)
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]]


def spherical(v: Vec3) -> tuple[float, float, float]:
    """Rectangular -> (longitude/RA rad in [0,2pi), latitude/Dec rad, radius)."""
    r = vnorm(v)
    lon = math.atan2(v[1], v[0]) % (2 * math.pi)
    lat = math.asin(max(-1.0, min(1.0, v[2] / r)))
    return lon, lat, r


def from_spherical(lon: float, lat: float, r: float = 1.0) -> Vec3:
    cl = math.cos(lat)
    return (r * cl * math.cos(lon), r * cl * math.sin(lon), r * math.sin(lat))


# --------------------------------------------------------------------------
# obliquity and nutation
# --------------------------------------------------------------------------
def mean_obliquity_deg(t: float) -> float:
    """Mean obliquity of the ecliptic (IAU 1980 / Laskar), degrees. t = centuries TT."""
    u = t / 100.0
    seconds = (21.448 - u * (4680.93 + u * (1.55 - u * (1999.25 - u * (51.38
               + u * (249.67 + u * (39.05 - u * (7.12 - u * (27.87
               + u * (5.79 + u * 2.45)))))))))) 
    return 23.0 + (26.0 + seconds / 60.0) / 60.0


# IAU 1980 nutation, abridged (Meeus table 22.A): the terms above ~0.03".
# arguments: multiples of D, M, M', F, Omega; then dpsi (0.0001"), ddpsi*T, deps, ddeps*T
_NUTATION_TERMS = (
    (0, 0, 0, 0, 1, -171996, -174.2, 92025, 8.9),
    (-2, 0, 0, 2, 2, -13187, -1.6, 5736, -3.1),
    (0, 0, 0, 2, 2, -2274, -0.2, 977, -0.5),
    (0, 0, 0, 0, 2, 2062, 0.2, -895, 0.5),
    (0, 1, 0, 0, 0, 1426, -3.4, 54, -0.1),
    (0, 0, 1, 0, 0, 712, 0.1, -7, 0.0),
    (-2, 1, 0, 2, 2, -517, 1.2, 224, -0.6),
    (0, 0, 0, 2, 1, -386, -0.4, 200, 0.0),
    (0, 0, 1, 2, 2, -301, 0.0, 129, -0.1),
    (-2, -1, 0, 2, 2, 217, -0.5, -95, 0.3),
    (-2, 0, 1, 0, 0, -158, 0.0, 0, 0.0),
    (-2, 0, 0, 2, 1, 129, 0.1, -70, 0.0),
    (0, 0, -1, 2, 2, 123, 0.0, -53, 0.0),
    (2, 0, 0, 0, 0, 63, 0.0, 0, 0.0),
    (0, 0, 1, 0, 1, 63, 0.1, -33, 0.0),
    (2, 0, -1, 2, 2, -59, 0.0, 26, 0.0),
    (0, 0, -1, 0, 1, -58, -0.1, 32, 0.0),
    (0, 0, 1, 2, 1, -51, 0.0, 27, 0.0),
    (-2, 0, 2, 0, 0, 48, 0.0, 0, 0.0),
    (0, 0, -2, 2, 1, 46, 0.0, -24, 0.0),
    (2, 0, 0, 2, 2, -38, 0.0, 16, 0.0),
    (0, 0, 2, 2, 2, -31, 0.0, 13, 0.0),
    (0, 0, 2, 0, 0, 29, 0.0, 0, 0.0),
    (-2, 0, 1, 2, 2, 29, 0.0, -12, 0.0),
    (0, 0, 0, 2, 0, 26, 0.0, 0, 0.0),
    (-2, 0, 0, 2, 0, -22, 0.0, 0, 0.0),
    (0, 0, -1, 2, 1, 21, 0.0, -10, 0.0),
    (0, 2, 0, 0, 0, 17, -0.1, 0, 0.0),
    (2, 0, -1, 0, 1, 16, 0.0, -8, 0.0),
    (-2, 2, 0, 2, 2, -16, 0.1, 7, 0.0),
    (0, 1, 0, 0, 1, -15, 0.0, 9, 0.0),
    (-2, 0, 1, 0, 1, -13, 0.0, 7, 0.0),
    (0, -1, 0, 0, 1, -12, 0.0, 6, 0.0),
    (0, 0, 2, -2, 0, 11, 0.0, 0, 0.0),
    (2, 0, -1, 2, 1, -10, 0.0, 5, 0.0),
    (2, 0, 1, 2, 2, -8, 0.0, 3, 0.0),
    (0, 1, 0, 2, 2, 7, 0.0, -3, 0.0),
    (-2, 1, 1, 0, 0, -7, 0.0, 0, 0.0),
    (0, -1, 0, 2, 2, -7, 0.0, 3, 0.0),
    (2, 0, 0, 2, 1, -7, 0.0, 3, 0.0),
    (2, 0, 1, 0, 0, 6, 0.0, 0, 0.0),
    (-2, 0, 2, 2, 2, 6, 0.0, -3, 0.0),
    (-2, 0, 1, 2, 1, 6, 0.0, -3, 0.0),
    (2, 0, -2, 0, 1, -6, 0.0, 3, 0.0),
    (2, 0, 0, 0, 1, -6, 0.0, 3, 0.0),
    (0, -1, 1, 0, 0, 5, 0.0, 0, 0.0),
    (-2, -1, 0, 2, 1, -5, 0.0, 3, 0.0),
    (-2, 0, 0, 0, 1, -5, 0.0, 3, 0.0),
    (0, 0, 2, 2, 1, -5, 0.0, 3, 0.0),
    (-2, 0, 2, 0, 1, 4, 0.0, 0, 0.0),
    (-2, 1, 0, 2, 1, 4, 0.0, 0, 0.0),
    (0, 0, 1, -2, 0, 4, 0.0, 0, 0.0),
    (-1, 0, 1, 0, 0, -4, 0.0, 0, 0.0),
    (-2, 1, 0, 0, 0, -4, 0.0, 0, 0.0),
    (1, 0, 0, 0, 0, -4, 0.0, 0, 0.0),
    (0, 0, 1, 2, 0, 3, 0.0, 0, 0.0),
    (0, 0, -2, 2, 2, -3, 0.0, 0, 0.0),
    (-1, -1, 1, 0, 0, -3, 0.0, 0, 0.0),
    (0, 1, 1, 0, 0, -3, 0.0, 0, 0.0),
    (0, -1, 1, 2, 2, -3, 0.0, 0, 0.0),
    (2, -1, -1, 2, 2, -3, 0.0, 0, 0.0),
    (0, 0, 3, 2, 2, -3, 0.0, 0, 0.0),
    (2, -1, 0, 2, 2, -3, 0.0, 0, 0.0),
)


def nutation(t: float) -> tuple[float, float]:
    """Nutation in longitude and obliquity, in degrees. t = centuries TT."""
    d = math.radians((297.85036 + 445267.111480 * t - 0.0019142 * t * t + t ** 3 / 189474.0) % 360.0)
    m = math.radians((357.52772 + 35999.050340 * t - 0.0001603 * t * t - t ** 3 / 300000.0) % 360.0)
    mp = math.radians((134.96298 + 477198.867398 * t + 0.0086972 * t * t + t ** 3 / 56250.0) % 360.0)
    f = math.radians((93.27191 + 483202.017538 * t - 0.0036825 * t * t + t ** 3 / 327270.0) % 360.0)
    om = math.radians((125.04452 - 1934.136261 * t + 0.0020708 * t * t + t ** 3 / 450000.0) % 360.0)

    dpsi = 0.0
    deps = 0.0
    for (cd, cm, cmp_, cf, com, sp, spt, ce, cet) in _NUTATION_TERMS:
        arg = cd * d + cm * m + cmp_ * mp + cf * f + com * om
        if sp:
            dpsi += (sp + spt * t) * math.sin(arg)
        if ce:
            deps += (ce + cet * t) * math.cos(arg)
    return dpsi * 1e-4 * ARCSEC, deps * 1e-4 * ARCSEC


def nutation_matrix(t: float, dpsi_deg: float, deps_deg: float):
    """Rotation taking mean-equinox-of-date equatorial vectors to true-of-date."""
    eps0 = math.radians(mean_obliquity_deg(t))
    eps = eps0 + math.radians(deps_deg)
    dpsi = math.radians(dpsi_deg)
    return matmul(rot_x(-eps), matmul(rot_z(-dpsi), rot_x(eps0)))


# --------------------------------------------------------------------------
# precession (IAU 1976, Lieske) -- J2000 equatorial -> mean equator of date
# --------------------------------------------------------------------------
def precession_matrix_j2000_to_date(t: float):
    zeta = math.radians((2306.2181 * t + 0.30188 * t * t + 0.017998 * t ** 3) * ARCSEC)
    z = math.radians((2306.2181 * t + 1.09468 * t * t + 0.018203 * t ** 3) * ARCSEC)
    theta = math.radians((2004.3109 * t - 0.42665 * t * t - 0.041833 * t ** 3) * ARCSEC)
    return matmul(rot_z(z), matmul(rot_y(-theta), rot_z(zeta)))


def ecl_to_equ(v: Vec3, obliquity_deg: float) -> Vec3:
    """Ecliptic rectangular -> equatorial rectangular (same equinox)."""
    return matvec(rot_x(math.radians(obliquity_deg)), v)


def equ_to_ecl(v: Vec3, obliquity_deg: float) -> Vec3:
    return matvec(rot_x(-math.radians(obliquity_deg)), v)


def equ_to_altaz(v_equ_of_date: Vec3, lat_deg: float, local_sidereal_deg: float) -> tuple[float, float]:
    """Topocentric equatorial-of-date vector -> (altitude, azimuth) in degrees.

    Azimuth is measured from North through East (0=N, 90=E), the convention a
    sky-watcher uses rather than Meeus' south-based azimuth.
    """
    ra, dec, _ = spherical(v_equ_of_date)
    h = math.radians(local_sidereal_deg) - ra          # local hour angle
    phi = math.radians(lat_deg)
    sin_alt = math.sin(dec) * math.sin(phi) + math.cos(dec) * math.cos(phi) * math.cos(h)
    alt = math.asin(max(-1.0, min(1.0, sin_alt)))
    az = math.atan2(math.sin(h), math.cos(h) * math.sin(phi) - math.tan(dec) * math.cos(phi))
    az_north = (math.degrees(az) + 180.0) % 360.0
    return math.degrees(alt), az_north


def refracted_altitude_deg(true_alt_deg: float) -> float:
    """Apparent altitude including standard atmospheric refraction (Bennett)."""
    if true_alt_deg < -2.0:
        return true_alt_deg
    h = true_alt_deg
    r = 1.02 / math.tan(math.radians(h + 10.3 / (h + 5.11))) / 60.0
    return h + r


def position_angle(from_v: Vec3, to_v: Vec3) -> float:
    """Position angle of `to_v` as seen from `from_v`, degrees east of north.

    Both vectors must be in the same equatorial frame.
    """
    ra0, dec0, _ = spherical(from_v)
    ra1, dec1, _ = spherical(to_v)
    dra = ra1 - ra0
    y = math.sin(dra) * math.cos(dec1)
    x = math.cos(dec0) * math.sin(dec1) - math.sin(dec0) * math.cos(dec1) * math.cos(dra)
    return math.degrees(math.atan2(y, x)) % 360.0
