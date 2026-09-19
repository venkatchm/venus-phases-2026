"""A compact bright-star catalogue for the sky background.

Positions are J2000 (degrees), magnitudes visual, and B-V colour indices are
used only to tint the rendered points. Coverage is roughly everything brighter
than magnitude 3, which is what an urban observer actually sees around a
twilight Moon-Venus pairing, plus the Virgo/Libra field the September 2026 event
takes place in.
"""
from __future__ import annotations

import math

from .frames import (from_spherical, matvec, precession_matrix_j2000_to_date,
                     spherical)

# (name, RA_deg, Dec_deg, Vmag, B-V)
CATALOG: tuple[tuple[str, float, float, float, float], ...] = (
    ("Sirius", 101.287, -16.716, -1.46, 0.00),
    ("Canopus", 95.988, -52.696, -0.74, 0.15),
    ("Rigil Kentaurus", 219.902, -60.834, -0.27, 0.71),
    ("Arcturus", 213.915, 19.182, -0.05, 1.23),
    ("Vega", 279.234, 38.784, 0.03, 0.00),
    ("Capella", 79.172, 45.998, 0.08, 0.80),
    ("Rigel", 78.634, -8.202, 0.13, -0.03),
    ("Procyon", 114.826, 5.225, 0.34, 0.42),
    ("Achernar", 24.429, -57.237, 0.46, -0.16),
    ("Betelgeuse", 88.793, 7.407, 0.50, 1.85),
    ("Hadar", 210.956, -60.373, 0.61, -0.23),
    ("Altair", 297.696, 8.868, 0.77, 0.22),
    ("Acrux", 186.650, -63.099, 0.77, -0.24),
    ("Aldebaran", 68.980, 16.509, 0.85, 1.54),
    ("Antares", 247.352, -26.432, 0.96, 1.83),
    ("Spica", 201.298, -11.161, 1.04, -0.23),
    ("Pollux", 116.329, 28.026, 1.14, 1.00),
    ("Fomalhaut", 344.413, -29.622, 1.16, 0.09),
    ("Deneb", 310.358, 45.280, 1.25, 0.09),
    ("Mimosa", 191.930, -59.689, 1.25, -0.24),
    ("Regulus", 152.093, 11.967, 1.35, -0.11),
    ("Adhara", 104.656, -28.972, 1.50, -0.21),
    ("Castor", 113.650, 31.888, 1.58, 0.03),
    ("Shaula", 263.402, -37.104, 1.62, -0.22),
    ("Gacrux", 187.791, -57.113, 1.63, 1.60),
    ("Bellatrix", 81.283, 6.350, 1.64, -0.22),
    ("Elnath", 81.573, 28.608, 1.65, -0.13),
    ("Miaplacidus", 138.300, -69.717, 1.67, 0.07),
    ("Alnilam", 84.053, -1.202, 1.69, -0.18),
    ("Alnair", 332.058, -46.961, 1.74, -0.13),
    ("Alnitak", 85.190, -1.943, 1.77, -0.21),
    ("Alioth", 193.507, 55.960, 1.77, -0.02),
    ("Dubhe", 165.932, 61.751, 1.79, 1.07),
    ("Mirfak", 51.081, 49.861, 1.79, 0.48),
    ("Wezen", 107.098, -26.393, 1.83, 0.67),
    ("Kaus Australis", 276.043, -34.385, 1.85, -0.03),
    ("Alkaid", 206.885, 49.313, 1.86, -0.19),
    ("Avior", 125.628, -59.510, 1.86, 1.20),
    ("Sargas", 264.330, -42.998, 1.87, 0.40),
    ("Menkalinan", 89.882, 44.947, 1.90, 0.08),
    ("Atria", 252.166, -69.028, 1.92, 1.44),
    ("Alhena", 99.428, 16.399, 1.93, 0.00),
    ("Peacock", 306.412, -56.735, 1.94, -0.12),
    ("Delta Velorum", 131.176, -54.709, 1.96, 0.04),
    ("Polaris", 37.955, 89.264, 1.97, 0.60),
    ("Mirzam", 95.675, -17.956, 1.98, -0.24),
    ("Alphard", 141.897, -8.659, 1.98, 1.44),
    ("Hamal", 31.793, 23.462, 2.00, 1.15),
    ("Diphda", 10.897, -17.987, 2.04, 1.02),
    ("Nunki", 283.816, -26.297, 2.05, -0.22),
    ("Menkent", 211.671, -36.370, 2.06, 1.01),
    ("Alpheratz", 2.097, 29.090, 2.06, -0.11),
    ("Mirach", 17.433, 35.620, 2.06, 1.58),
    ("Kochab", 222.676, 74.156, 2.08, 1.47),
    ("Rasalhague", 263.734, 12.560, 2.08, 0.16),
    ("Saiph", 86.939, -9.670, 2.09, -0.17),
    ("Algol", 47.042, 40.956, 2.09, -0.05),
    ("Almach", 30.975, 42.330, 2.10, 1.37),
    ("Denebola", 177.265, 14.572, 2.14, 0.09),
    ("Aspidiske", 139.273, -59.275, 2.21, 0.19),
    ("Suhail", 136.999, -43.433, 2.21, 1.66),
    ("Alphecca", 233.672, 26.715, 2.22, -0.02),
    ("Sadr", 305.557, 40.257, 2.23, 0.68),
    ("Mizar", 200.981, 54.925, 2.23, 0.06),
    ("Eltanin", 269.152, 51.489, 2.23, 1.52),
    ("Schedar", 10.127, 56.537, 2.24, 1.17),
    ("Naos", 120.896, -40.003, 2.25, -0.27),
    ("Mintaka", 83.002, -0.299, 2.25, -0.18),
    ("Caph", 2.294, 59.150, 2.27, 0.34),
    ("Algieba", 154.993, 19.842, 2.28, 1.13),
    ("Fang", 239.713, -26.114, 2.29, -0.19),
    ("Larawag", 252.541, -34.293, 2.29, 1.15),
    ("Wei", 252.968, -38.048, 2.29, 1.16),
    ("Dschubba", 240.083, -22.622, 2.32, -0.12),
    ("Izar", 221.247, 27.074, 2.35, 0.97),
    ("Merak", 165.460, 56.382, 2.37, 0.03),
    ("Enif", 326.046, 9.875, 2.38, 1.53),
    ("Girtab", 262.691, -37.296, 2.39, -0.21),
    ("Ankaa", 6.571, -42.306, 2.40, 1.08),
    ("Scheat", 345.944, 28.083, 2.42, 1.67),
    ("Sabik", 257.595, -15.725, 2.43, 0.06),
    ("Phecda", 178.458, 53.695, 2.44, 0.04),
    ("Aludra", 111.024, -29.303, 2.45, -0.08),
    ("Markab", 346.190, 15.205, 2.49, -0.04),
    ("Han", 249.290, -10.567, 2.54, 0.04),
    ("Gienah", 183.952, -17.542, 2.59, -0.11),
    ("Zubeneschamali", 229.252, -9.383, 2.61, -0.07),
    ("Acrab", 241.359, -19.805, 2.62, -0.07),
    ("Unukalhai", 236.067, 6.426, 2.63, 1.17),
    ("Muphrid", 208.671, 18.398, 2.68, 0.58),
    ("Yed Prior", 243.586, -3.694, 2.73, 1.58),
    ("Porrima", 190.415, -1.449, 2.74, 0.36),
    ("Zubenelgenubi", 222.720, -16.042, 2.75, 0.15),
    ("Cebalrai", 265.868, 4.567, 2.76, 1.17),
    ("Kornephoros", 247.555, 21.490, 2.78, 0.94),
    ("Vindemiatrix", 195.544, 10.959, 2.83, 0.94),
    ("Alniyat", 245.297, -25.593, 2.89, 0.15),
    ("Cor Caroli", 194.007, 38.318, 2.89, -0.11),
    ("Seginus", 218.020, 38.308, 3.03, 0.19),
    ("Rasalgethi", 258.662, 14.390, 3.35, 1.17),
    ("Auva", 193.901, 3.397, 3.38, 1.57),
    ("Heze", 197.264, -0.667, 3.38, 0.11),
    ("Nekkar", 225.487, 40.390, 3.49, 0.95),
    ("Zavijava", 177.674, 1.765, 3.59, 0.55),
    ("Zaniah", 184.977, -0.667, 3.89, 0.02),
    ("Syrma", 202.761, -5.994, 4.07, 0.55),
)


def bv_to_rgb(bv: float) -> tuple[float, float, float]:
    """Very simple B-V to linear RGB tint, normalised to peak 1."""
    bv = max(-0.4, min(2.0, bv))
    # approximate colour temperature from B-V (Ballesteros)
    tk = 4600.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62))
    t = tk / 100.0
    if t <= 66.0:
        r = 1.0
        g = 0.3900816 * math.log(t) - 0.6318414
    else:
        r = 1.2929362 * (t - 60.0) ** -0.1332047
        g = 1.1298909 * (t - 60.0) ** -0.0755148
    if t >= 66.0:
        b = 1.0
    elif t <= 19.0:
        b = 0.0
    else:
        b = 0.5432068 * math.log(t - 10.0) - 1.1962541
    r, g, b = (max(0.0, min(1.0, c)) for c in (r, g, b))
    peak = max(r, g, b, 1e-6)
    return r / peak, g / peak, b / peak


def precessed_equatorial(t: float):
    """Catalogue directions precessed to the mean equator of date.

    Returns a list of (unit vector, magnitude, (r, g, b)).
    """
    m = precession_matrix_j2000_to_date(t)
    out = []
    for (_name, ra, dec, mag, bv) in CATALOG:
        v = from_spherical(math.radians(ra), math.radians(dec), 1.0)
        out.append((matvec(m, v), mag, bv_to_rgb(bv)))
    return out
