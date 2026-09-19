"""Time scales: calendar <-> Julian Day, Delta-T, sidereal time.

All ephemeris routines in this package take Julian Ephemeris Day (JDE, i.e. TT).
Earth-rotation routines (sidereal time) take JD in UT1 ~= UTC.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

J2000 = 2451545.0
DAYS_PER_CENTURY = 36525.0


def julian_day(year: int, month: int, day: float) -> float:
    """Gregorian calendar date -> Julian Day (Meeus ch. 7)."""
    if month <= 2:
        year -= 1
        month += 12
    a = year // 100
    b = 2 - a + a // 4
    return (math.floor(365.25 * (year + 4716))
            + math.floor(30.6001 * (month + 1))
            + day + b - 1524.5)


def calendar_date(jd: float) -> tuple[int, int, float]:
    """Julian Day -> (year, month, fractional day)."""
    jd += 0.5
    z = math.floor(jd)
    f = jd - z
    if z < 2299161:
        a = z
    else:
        alpha = math.floor((z - 1867216.25) / 36524.25)
        a = z + 1 + alpha - math.floor(alpha / 4)
    b = a + 1524
    c = math.floor((b - 122.1) / 365.25)
    d = math.floor(365.25 * c)
    e = math.floor((b - d) / 30.6001)
    day = b - d - math.floor(30.6001 * e) + f
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715
    return int(year), int(month), day


def utc_to_jd(year: int, month: int, day: int,
              hour: int = 0, minute: int = 0, second: float = 0.0) -> float:
    return julian_day(year, month, day + (hour + minute / 60.0 + second / 3600.0) / 24.0)


def jd_to_utc_string(jd: float, seconds: bool = True) -> str:
    y, m, d = calendar_date(jd)
    day = int(math.floor(d))
    frac = (d - day) * 24.0
    hh = int(frac)
    mm_f = (frac - hh) * 60.0
    mm = int(mm_f)
    ss = (mm_f - mm) * 60.0
    if round(ss, 0) >= 60.0:           # carry rounding upward
        ss = 0.0
        mm += 1
    if mm >= 60:
        mm -= 60
        hh += 1
    if hh >= 24:
        hh -= 24
        day += 1
    if seconds:
        return f"{y:04d}-{m:02d}-{day:02d} {hh:02d}:{mm:02d}:{ss:04.1f} UT"
    return f"{y:04d}-{m:02d}-{day:02d} {hh:02d}:{mm:02d} UT"


def delta_t(jd_ut: float) -> float:
    """TT - UT1 in seconds.

    Espenak & Meeus polynomial for 2005-2050, which for the 2020s tracks the
    observed value to well under a second -- far below what this simulation needs
    (1 s of Delta-T moves the Moon by ~0.5").
    """
    y, m, _ = calendar_date(jd_ut)
    year = y + (m - 0.5) / 12.0
    if 2005.0 <= year <= 2050.0:
        t = year - 2000.0
        return 62.92 + 0.32217 * t + 0.005589 * t * t
    if year > 2050.0:
        return -20.0 + 32.0 * ((year - 1820.0) / 100.0) ** 2 - 0.5628 * (2150.0 - year)
    # 1986-2005
    t = year - 2000.0
    return (63.86 + 0.3345 * t - 0.060374 * t * t + 0.0017275 * t ** 3
            + 0.000651814 * t ** 4 + 0.00002373599 * t ** 5)


def jd_ut_to_jde(jd_ut: float) -> float:
    """UT -> TT (Julian Ephemeris Day)."""
    return jd_ut + delta_t(jd_ut) / 86400.0


def jde_to_jd_ut(jde: float) -> float:
    return jde - delta_t(jde) / 86400.0


def centuries_tt(jde: float) -> float:
    """Julian centuries of TT since J2000.0."""
    return (jde - J2000) / DAYS_PER_CENTURY


def gmst_deg(jd_ut: float) -> float:
    """Greenwich mean sidereal time in degrees (Meeus 12.4)."""
    t = (jd_ut - J2000) / DAYS_PER_CENTURY
    theta = (280.46061837
             + 360.98564736629 * (jd_ut - J2000)
             + 0.000387933 * t * t
             - t ** 3 / 38710000.0)
    return theta % 360.0


def gast_deg(jd_ut: float, nutation_longitude_deg: float, true_obliquity_deg: float) -> float:
    """Greenwich apparent sidereal time in degrees (equation of the equinoxes)."""
    eq = nutation_longitude_deg * math.cos(math.radians(true_obliquity_deg))
    return (gmst_deg(jd_ut) + eq) % 360.0


@dataclass(frozen=True)
class Instant:
    """A moment in time, carrying both UT and TT forms."""
    jd_ut: float

    @property
    def jde(self) -> float:
        return jd_ut_to_jde(self.jd_ut)

    @property
    def t(self) -> float:
        return centuries_tt(self.jde)

    def __str__(self) -> str:
        return jd_to_utc_string(self.jd_ut)

    @staticmethod
    def from_utc(year: int, month: int, day: int,
                 hour: int = 0, minute: int = 0, second: float = 0.0) -> "Instant":
        return Instant(utc_to_jd(year, month, day, hour, minute, second))

    def plus_seconds(self, seconds: float) -> "Instant":
        return Instant(self.jd_ut + seconds / 86400.0)

    def plus_days(self, days: float) -> "Instant":
        return Instant(self.jd_ut + days)
