"""Lunar position from the truncated ELP-2000/82 series (Meeus ch. 47).

Accuracy of this truncation is ~10" in longitude and ~4" in latitude -- roughly
1/200 of the Moon's apparent diameter, which is far finer than the geometry an
occultation simulation needs to get right.
"""
from __future__ import annotations

import math

from .frames import Vec3, ecl_to_equ, from_spherical, mean_obliquity_deg, nutation

MOON_RADIUS_KM = 1737.4
EARTH_MOON_MASS_RATIO = 81.30056          # Earth mass / Moon mass

# Table 47.A -- arguments (D, M, M', F), sine coeff for longitude (1e-6 deg),
# cosine coeff for distance (1e-3 km).
_TERMS_LR = (
    (0, 0, 1, 0, 6288774, -20905355), (2, 0, -1, 0, 1274027, -3699111),
    (2, 0, 0, 0, 658314, -2955968), (0, 0, 2, 0, 213618, -569925),
    (0, 1, 0, 0, -185116, 48888), (0, 0, 0, 2, -114332, -3149),
    (2, 0, -2, 0, 58793, 246158), (2, -1, -1, 0, 57066, -152138),
    (2, 0, 1, 0, 53322, -170733), (2, -1, 0, 0, 45758, -204586),
    (0, 1, -1, 0, -40923, -129620), (1, 0, 0, 0, -34720, 108743),
    (0, 1, 1, 0, -30383, 104755), (2, 0, 0, -2, 15327, 10321),
    (0, 0, 1, 2, -12528, 0), (0, 0, 1, -2, 10980, 79661),
    (4, 0, -1, 0, 10675, -34782), (0, 0, 3, 0, 10034, -23210),
    (4, 0, -2, 0, 8548, -21636), (2, 1, -1, 0, -7888, 24208),
    (2, 1, 0, 0, -6766, 30824), (1, 0, -1, 0, -5163, -8379),
    (1, 1, 0, 0, 4987, -16675), (2, -1, 1, 0, 4036, -12831),
    (2, 0, 2, 0, 3994, -10445), (4, 0, 0, 0, 3861, -11650),
    (2, 0, -3, 0, 3665, 14403), (0, 1, -2, 0, -2689, -7003),
    (2, 0, -1, 2, -2602, 0), (2, -1, -2, 0, 2390, 10056),
    (1, 0, 1, 0, -2348, 6322), (2, -2, 0, 0, 2236, -9884),
    (0, 1, 2, 0, -2120, 5751), (0, 2, 0, 0, -2069, 0),
    (2, -2, -1, 0, 2048, -4950), (2, 0, 1, -2, -1773, 4130),
    (2, 0, 0, 2, -1595, 0), (4, -1, -1, 0, 1215, -3958),
    (0, 0, 2, 2, -1110, 0), (3, 0, -1, 0, -892, 3258),
    (2, 1, 1, 0, -810, 2616), (4, -1, -2, 0, 759, -1897),
    (0, 2, -1, 0, -713, -2117), (2, 2, -1, 0, -700, 2354),
    (2, 1, -2, 0, 691, 0), (2, -1, 0, -2, 596, 0),
    (4, 0, 1, 0, 549, -1423), (0, 0, 4, 0, 537, -1117),
    (4, -1, 0, 0, 520, -1571), (1, 0, -2, 0, -487, -1739),
    (2, 1, 0, -2, -399, 0), (0, 0, 2, -2, -381, -4421),
    (1, 1, 1, 0, 351, 0), (3, 0, -2, 0, -340, 0),
    (4, 0, -3, 0, 330, 0), (2, -1, 2, 0, 327, 0),
    (0, 2, 1, 0, -323, 1165), (1, 1, -1, 0, 299, 0),
    (2, 0, 3, 0, 294, 0), (2, 0, -1, -2, 0, 8752),
)

# Table 47.B -- arguments (D, M, M', F), sine coeff for latitude (1e-6 deg).
_TERMS_B = (
    (0, 0, 0, 1, 5128122), (0, 0, 1, 1, 280602), (0, 0, 1, -1, 277693),
    (2, 0, 0, -1, 173237), (2, 0, -1, 1, 55413), (2, 0, -1, -1, 46271),
    (2, 0, 0, 1, 32573), (0, 0, 2, 1, 17198), (2, 0, 1, -1, 9266),
    (0, 0, 2, -1, 8822), (2, -1, 0, -1, 8216), (2, 0, -2, -1, 4324),
    (2, 0, 1, 1, 4200), (2, 1, 0, -1, -3359), (2, -1, -1, 1, 2463),
    (2, -1, 0, 1, 2211), (2, -1, -1, -1, 2065), (0, 1, -1, -1, -1870),
    (4, 0, -1, -1, 1828), (0, 1, 0, 1, -1794), (0, 0, 0, 3, -1749),
    (0, 1, -1, 1, -1565), (1, 0, 0, 1, -1491), (0, 1, 1, 1, -1475),
    (0, 1, 1, -1, -1410), (0, 1, 0, -1, -1344), (1, 0, 0, -1, -1335),
    (0, 0, 3, 1, 1107), (4, 0, 0, -1, 1021), (4, 0, -1, 1, 833),
    (0, 0, 1, -3, 777), (4, 0, -2, 1, 671), (2, 0, 0, -3, 607),
    (2, 0, 2, -1, 596), (2, -1, 1, -1, 491), (2, 0, -2, 1, -451),
    (0, 0, 3, -1, 439), (2, 0, 2, 1, 422), (2, 0, -3, -1, 421),
    (2, 1, -1, 1, -366), (2, 1, 0, 1, -351), (4, 0, 0, 1, 331),
    (2, -1, 1, 1, 315), (2, -2, 0, -1, 302), (0, 0, 1, 3, -283),
    (2, 1, 1, -1, -229), (1, 1, 0, -1, 223), (1, 1, 0, 1, 223),
    (0, 1, -2, -1, -220), (2, 1, -1, -1, -220), (1, 0, 1, 1, -185),
    (2, -1, -2, -1, 181), (0, 1, 2, 1, -177), (4, 0, -2, -1, 176),
    (4, -1, -1, -1, 166), (1, 0, 1, -1, -164), (4, 0, 1, -1, 132),
    (1, 0, -1, -1, -119), (4, -1, 0, -1, 115), (2, -2, 0, 1, 107),
)


def _fundamental_arguments(t: float) -> tuple[float, float, float, float, float]:
    """(L', D, M, M', F) in degrees."""
    lp = (218.3164477 + 481267.88123421 * t - 0.0015786 * t * t
          + t ** 3 / 538841.0 - t ** 4 / 65194000.0)
    d = (297.8501921 + 445267.1114034 * t - 0.0018819 * t * t
         + t ** 3 / 545868.0 - t ** 4 / 113065000.0)
    m = (357.5291092 + 35999.0502909 * t - 0.0001536 * t * t + t ** 3 / 24490000.0)
    mp = (134.9633964 + 477198.8675055 * t + 0.0087414 * t * t
          + t ** 3 / 69699.0 - t ** 4 / 14712000.0)
    f = (93.2720950 + 483202.0175233 * t - 0.0036539 * t * t
         - t ** 3 / 3526000.0 + t ** 4 / 863310000.0)
    return lp % 360.0, d % 360.0, m % 360.0, mp % 360.0, f % 360.0


def geocentric_ecliptic(t: float) -> tuple[float, float, float]:
    """Geocentric ecliptic longitude/latitude (deg, mean equinox of date) and
    distance (km) of the Moon's centre. t = Julian centuries TT."""
    lp, d, m, mp, f = _fundamental_arguments(t)
    rd = math.radians
    D, M, MP, F = rd(d), rd(m), rd(mp), rd(f)

    # eccentricity correction for terms involving the Sun's anomaly
    e = 1.0 - 0.002516 * t - 0.0000074 * t * t

    a1 = rd((119.75 + 131.849 * t) % 360.0)
    a2 = rd((53.09 + 479264.290 * t) % 360.0)
    a3 = rd((313.45 + 481266.484 * t) % 360.0)

    sum_l = 0.0
    sum_r = 0.0
    for (cd, cm, cmp_, cf, cl, cr) in _TERMS_LR:
        arg = cd * D + cm * M + cmp_ * MP + cf * F
        ecc = e ** abs(cm) if cm else 1.0
        if cl:
            sum_l += cl * ecc * math.sin(arg)
        if cr:
            sum_r += cr * ecc * math.cos(arg)

    sum_b = 0.0
    for (cd, cm, cmp_, cf, cb) in _TERMS_B:
        arg = cd * D + cm * M + cmp_ * MP + cf * F
        ecc = e ** abs(cm) if cm else 1.0
        sum_b += cb * ecc * math.sin(arg)

    # additive terms from Venus (A1), Jupiter (A2) and Earth's flattening
    sum_l += 3958.0 * math.sin(a1) + 1962.0 * math.sin(rd(lp) - F) + 318.0 * math.sin(a2)
    sum_b += (-2235.0 * math.sin(rd(lp)) + 382.0 * math.sin(a3)
              + 175.0 * math.sin(a1 - F) + 175.0 * math.sin(a1 + F)
              + 127.0 * math.sin(rd(lp) - MP) - 115.0 * math.sin(rd(lp) + MP))

    lon = (lp + sum_l / 1e6) % 360.0
    lat = sum_b / 1e6
    dist = 385000.56 + sum_r / 1000.0
    return lon, lat, dist


def geocentric_equatorial_km(t: float, apparent: bool = True) -> Vec3:
    """Geocentric equatorial rectangular position of the Moon, in km.

    With `apparent`, nutation in longitude is applied and the true obliquity is
    used, giving coordinates referred to the true equinox of date.
    """
    lon, lat, dist = geocentric_ecliptic(t)
    eps = mean_obliquity_deg(t)
    if apparent:
        dpsi, deps = nutation(t)
        lon += dpsi
        eps += deps
    v_ecl = from_spherical(math.radians(lon), math.radians(lat), dist)
    return ecl_to_equ(v_ecl, eps)


def angular_radius_deg(distance_km: float) -> float:
    return math.degrees(math.asin(MOON_RADIUS_KM / distance_km))


# ---------------------------------------------------------------------------
# named nearside features
# ---------------------------------------------------------------------------
# Selenographic latitude and longitude in degrees (longitude positive east, the
# IAU convention) and an angular radius in degrees of arc across the lunar
# sphere, from the feature's diameter over the Moon's 1737.4 km radius.
#
# These are here rather than in a texture file because this project ships no
# image assets: every marking is generated. Placing the real maria at their real
# coordinates is what turns a plausible grey sphere into *the* Moon -- Crisium
# alone, small and round and off on the eastern limb, is recognisable at a
# glance, and no amount of good noise substitutes for it.
MARIA = (
    # name,                  lat,    lon,  angular radius
    ("Oceanus Procellarum",  18.4, -57.4, 25.0),
    ("Mare Imbrium",         32.8, -15.6, 12.8),
    ("Mare Serenitatis",     28.0,  17.5, 11.2),
    ("Mare Tranquillitatis",  8.5,  31.4, 14.3),
    ("Mare Fecunditatis",    -7.8,  51.3, 12.0),
    ("Mare Crisium",         17.0,  59.1,  8.6),
    ("Mare Nubium",         -21.3, -16.6, 11.0),
    ("Mare Humorum",        -24.4, -38.6,  6.4),
    ("Mare Nectaris",       -15.2,  35.5,  5.6),
    ("Mare Vaporum",         13.3,   3.6,  3.8),
    ("Mare Frigoris",        56.0,   1.4,  9.0),
    ("Mare Cognitum",       -10.0, -23.1,  4.5),
    ("Mare Insularum",        7.5, -30.9,  6.0),
    ("Mare Australe",       -38.9,  93.0,  8.0),
)

# The bright rayed craters, which are the other half of what makes the face
# recognisable: Tycho's rays reach a third of the way round the disc.
# name, lat, lon, crater angular radius, ray extent (degrees)
RAY_CRATERS = (
    ("Tycho",       -43.3, -11.4, 1.4, 26.0),
    ("Copernicus",    9.6, -20.1, 1.6, 12.0),
    ("Kepler",        8.1, -38.0, 0.6,  8.0),
    ("Aristarchus",  23.7, -47.4, 0.7,  7.0),
)


def selenographic_vector(lat_deg: float, lon_deg: float) -> Vec3:
    """Unit vector in the Moon's body frame: x to the sub-Earth point, z north.

    Longitude is measured east from the mean sub-Earth meridian, so Mare Crisium
    at +59 deg lands near the eastern limb where it belongs.
    """
    la, lo = math.radians(lat_deg), math.radians(lon_deg)
    return (math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la))
