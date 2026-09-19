"""Event finding: occultation contacts, conjunctions, greatest brilliancy.

Everything here is a root/extremum search over `scene.sky_state`, refined by
bisection or golden section rather than by any closed-form approximation, so the
answers are only as good as the ephemeris -- which is the point.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .observer import Site
from .scene import SkyState, sky_state
from .timescale import Instant, jd_to_utc_string


@dataclass
class Contact:
    """One limb crossing."""
    label: str
    instant: Instant
    moon_altitude_deg: float
    sun_altitude_deg: float
    position_angle_deg: float       # on the lunar limb, east of north
    cusp_angle_deg: float           # from the bright-limb midpoint

    def __str__(self) -> str:
        return (f"{self.label:<24s} {jd_to_utc_string(self.instant.jd_ut)}  "
                f"Moon alt {self.moon_altitude_deg:+5.1f}deg  "
                f"Sun alt {self.sun_altitude_deg:+5.1f}deg  PA {self.position_angle_deg:5.1f}deg")


@dataclass
class OccultationResult:
    site: Site
    occurs: bool
    contacts: list[Contact] = field(default_factory=list)
    min_separation_arcsec: float = 0.0
    min_separation_instant: Instant | None = None
    moon_radius_arcsec: float = 0.0
    moon_altitude_at_min_deg: float = 0.0
    sun_altitude_at_min_deg: float = 0.0
    duration_seconds: float = 0.0
    visibility: str = ""

    @property
    def disappearance(self) -> Contact | None:
        return next((c for c in self.contacts if c.label.startswith("Disappear")), None)

    @property
    def reappearance(self) -> Contact | None:
        return next((c for c in self.contacts if c.label.startswith("Reappear")), None)


def _limb_distance(instant: Instant, site: Site) -> float:
    """Venus' centre minus the lunar limb, in arcsec. Negative = hidden."""
    st = sky_state(instant, site)
    return (st.separation_deg - st.moon.angular_radius_deg) * 3600.0


def _bisect(site: Site, t_lo: float, t_hi: float, offset: float,
            tolerance_s: float = 0.05) -> Instant:
    """Refine a sign change of (limb distance - offset) between two JDs."""
    f_lo = _limb_distance(Instant(t_lo), site) - offset
    for _ in range(60):
        mid = 0.5 * (t_lo + t_hi)
        f_mid = _limb_distance(Instant(mid), site) - offset
        if (f_lo < 0.0) == (f_mid < 0.0):
            t_lo, f_lo = mid, f_mid
        else:
            t_hi = mid
        if (t_hi - t_lo) * 86400.0 < tolerance_s:
            break
    return Instant(0.5 * (t_lo + t_hi))


def _contact(site: Site, instant: Instant, label: str) -> Contact:
    st = sky_state(instant, site)
    pa = st.venus_moon_pa_deg
    cusp = abs(((pa - st.moon.bright_limb_pa_deg + 180.0) % 360.0) - 180.0)
    return Contact(label=label, instant=instant,
                   moon_altitude_deg=st.moon.apparent_altitude_deg,
                   sun_altitude_deg=st.sun.apparent_altitude_deg,
                   position_angle_deg=pa, cusp_angle_deg=cusp)


def _describe_visibility(moon_alt: float, sun_alt: float) -> str:
    if moon_alt < 0.0:
        return "below the horizon"
    if sun_alt > 0.0:
        return "in daylight (Moon and Venus both up; needs optics or a very clear sky)"
    if sun_alt > -6.0:
        return "in bright twilight, low in the west"
    if sun_alt > -12.0:
        return "in civil-to-nautical twilight"
    return "in a dark sky"


def find_occultation(site: Site, start: Instant, hours: float = 14.0,
                     coarse_step_s: float = 30.0) -> OccultationResult:
    """Scan a window for a lunar occultation of Venus at one site."""
    n = int(hours * 3600.0 / coarse_step_s)
    times, values, states = [], [], []
    for k in range(n + 1):
        inst = start.plus_seconds(k * coarse_step_s)
        st = sky_state(inst, site)
        times.append(inst.jd_ut)
        values.append((st.separation_deg - st.moon.angular_radius_deg) * 3600.0)
        states.append(st)

    i_min = min(range(len(values)), key=lambda i: values[i])
    st_min = states[i_min]
    result = OccultationResult(
        site=site, occurs=values[i_min] < 0.0,
        min_separation_arcsec=st_min.separation_arcsec,
        min_separation_instant=st_min.instant,
        moon_radius_arcsec=st_min.moon.angular_radius_deg * 3600.0,
        moon_altitude_at_min_deg=st_min.moon.apparent_altitude_deg,
        sun_altitude_at_min_deg=st_min.sun.apparent_altitude_deg,
    )
    result.visibility = _describe_visibility(result.moon_altitude_at_min_deg,
                                             result.sun_altitude_at_min_deg)
    if not result.occurs:
        return result

    # Venus has a finite disc: first and last contact bracket a gradual fade
    venus_r = st_min.venus.angular_radius_deg * 3600.0
    edges = [("Disappearance begins", +venus_r), ("Disappearance complete", -venus_r),
             ("Reappearance begins", -venus_r), ("Reappearance complete", +venus_r)]
    ingress_done = False
    for label, offset in edges:
        target = offset
        found = None
        rng = range(len(values) - 1)
        for i in rng:
            a, b = values[i] - target, values[i + 1] - target
            if a == 0.0 or (a > 0.0) != (b > 0.0):
                descending = a > b
                if label.startswith("Disappear") and not descending:
                    continue
                if label.startswith("Reappear") and descending:
                    continue
                found = _bisect(site, times[i], times[i + 1], target)
                break
        if found is not None:
            result.contacts.append(_contact(site, found, label))

    d = result.disappearance
    r = result.reappearance
    if d and r:
        result.duration_seconds = (r.instant.jd_ut - d.instant.jd_ut) * 86400.0
    return result


# --------------------------------------------------------------------------
# apparition milestones
# --------------------------------------------------------------------------
def _golden_extremum(f, lo: float, hi: float, maximise: bool, iterations: int = 80) -> float:
    """Golden-section search for an extremum of f over [lo, hi] (JD)."""
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = f(c), f(d)
    if maximise:
        fc, fd = -fc, -fd
    for _ in range(iterations):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c) if not maximise else -f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d) if not maximise else -f(d)
        if (b - a) * 86400.0 < 1.0:
            break
    return 0.5 * (a + b)


def greatest_brilliancy(site: Site, search_start: Instant, days: float = 40.0) -> tuple[Instant, float]:
    """Instant of minimum (brightest) apparent magnitude, and that magnitude."""
    f = lambda jd: sky_state(Instant(jd), site).venus.magnitude
    jd = _golden_extremum(f, search_start.jd_ut, search_start.jd_ut + days, maximise=False)
    return Instant(jd), f(jd)


def greatest_elongation(site: Site, search_start: Instant, days: float = 60.0) -> tuple[Instant, float]:
    f = lambda jd: sky_state(Instant(jd), site).elongation_deg
    jd = _golden_extremum(f, search_start.jd_ut, search_start.jd_ut + days, maximise=True)
    return Instant(jd), f(jd)


def inferior_conjunction(site: Site, search_start: Instant, days: float = 90.0) -> tuple[Instant, float]:
    f = lambda jd: sky_state(Instant(jd), site).elongation_deg
    jd = _golden_extremum(f, search_start.jd_ut, search_start.jd_ut + days, maximise=False)
    return Instant(jd), f(jd)


def conjunction_in_right_ascension(start: Instant, hours: float = 24.0) -> Instant:
    """Geocentric Moon-Venus conjunction in RA, the classical event definition."""
    from .frames import spherical
    from .scene import geocentric_bodies

    def dra(jd: float) -> float:
        g = geocentric_bodies(Instant(jd))
        a = math.degrees(spherical(g["moon"])[0])
        b = math.degrees(spherical(g["venus"])[0])
        return ((a - b + 180.0) % 360.0) - 180.0

    lo, hi = start.jd_ut, start.jd_ut + hours / 24.0
    f_lo = dra(lo)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f_mid = dra(mid)
        if (f_lo < 0.0) == (f_mid < 0.0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return Instant(0.5 * (lo + hi))

def sunset_instant(site: Site, year: int, month: int, day: int) -> Instant | None:
    """The evening sunset at `site` on the given date, or None if there is none.

    Located as the descending zero crossing of the Sun's apparent altitude, so
    it needs no timezone table and works at any longitude.
    """
    base = Instant.from_utc(year, month, day).jd_ut
    prev = None
    lo = None
    for k in range(0, 289):
        alt = sky_state(Instant(base + k / 288.0), site).sun.apparent_altitude_deg
        if prev is not None and prev > 0.0 >= alt:
            lo = base + (k - 1) / 288.0
            break
        prev = alt
    if lo is None:
        return None
    hi = lo + 1.0 / 288.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if sky_state(Instant(mid), site).sun.apparent_altitude_deg > 0.0:
            lo = mid
        else:
            hi = mid
    return Instant(0.5 * (lo + hi))


def evening_instant(site: Site, year: int, month: int, day: int,
                    minutes_after_sunset: float = 30.0) -> Instant:
    """A fixed interval after sunset -- the moment Venus is best placed."""
    s = sunset_instant(site, year, month, day)
    if s is None:                       # polar summer: fall back to local midnight-ish
        s = Instant(Instant.from_utc(year, month, day).jd_ut + 0.75)
    return Instant(s.jd_ut + minutes_after_sunset / 1440.0)


# --------------------------------------------------------------------------
# geocentric apparition milestones (no observer needed)
# --------------------------------------------------------------------------
def geocentric_elongation(jd: float) -> float:
    """Sun-Earth-Venus angle in degrees, geocentric."""
    from .scene import venus_geometry
    return venus_geometry(Instant(jd)).elongation_deg


def longitude_difference(jd: float) -> float:
    """Venus' apparent geocentric ecliptic longitude minus the Sun's, degrees.

    Wrapped to (-180, +180]. Zero at conjunction, by the standard definition.
    """
    from .frames import spherical
    from .planets import geocentric_of_date, sun_geocentric_of_date

    jde = Instant(jd).jde
    venus, _lt, _helio = geocentric_of_date("venus", jde)
    sun = sun_geocentric_of_date(jde)
    a = math.degrees(spherical(venus)[0])
    b = math.degrees(spherical(sun)[0])
    return ((a - b + 180.0) % 360.0) - 180.0


def solar_conjunction(search_start: Instant, days: float = 200.0) -> tuple[Instant, str]:
    """The next conjunction with the Sun, and which kind it is.

    Conjunction is located as the instant Venus and the Sun share an apparent
    geocentric *ecliptic longitude*, which is the definition the almanacs
    tabulate -- not as the minimum of the angular separation.

    Those are not the same instant, and the difference is not small. Venus'
    orbit is inclined 3.4 degrees, so at conjunction it passes above or below
    the Sun rather than across it; the separation therefore bottoms out about
    ten hours away from the longitude crossing. Solving the minimum instead put
    the 2026 inferior conjunction at 14:19 UT against a published 04 UT. The
    longitude crossing lands at 03:49 UT.

    The kind is read off the geometry rather than passed in: beyond the Sun
    (delta greater than the Earth's own orbital radius) is superior, between us
    and the Sun is inferior. A caller can therefore scan forward and let each
    event name itself.
    """
    from .scene import venus_geometry

    lo, hi = search_start.jd_ut, search_start.jd_ut + days
    steps = max(8, int(days * 4.0))
    prev_t, prev_f = lo, longitude_difference(lo)
    bracket = None
    for k in range(1, steps + 1):
        t = lo + (hi - lo) * k / steps
        f = longitude_difference(t)
        # the jump from +180 to -180 half a period away is not a crossing
        if abs(f - prev_f) < 90.0 and (f < 0.0) != (prev_f < 0.0):
            bracket = (prev_t, t)
            break
        prev_t, prev_f = t, f
    if bracket is None:
        raise ValueError("no solar conjunction in the search window")

    a, b = bracket
    fa = longitude_difference(a)
    for _ in range(80):
        mid = 0.5 * (a + b)
        fm = longitude_difference(mid)
        if (fa < 0.0) == (fm < 0.0):
            a, fa = mid, fm
        else:
            b = mid
        if (b - a) * 86400.0 < 1.0:
            break
    jd = 0.5 * (a + b)
    g = venus_geometry(Instant(jd))
    return Instant(jd), ("superior" if g.is_superior_side else "inferior")


def geocentric_greatest_elongation(search_start: Instant,
                                   days: float = 200.0) -> tuple[Instant, float]:
    """Greatest elongation from the Sun, and its value in degrees."""
    jd = _golden_extremum(geocentric_elongation, search_start.jd_ut,
                          search_start.jd_ut + days, maximise=True)
    return Instant(jd), geocentric_elongation(jd)


def elongation_side(instant: Instant) -> str:
    """"eastern" or "western", from the geocentric ecliptic longitude difference.

    East of the Sun means Venus trails it across the sky and so sets after it:
    the evening star. The comparison has to be made on the *geocentric* Venus
    vector -- the heliocentric one points somewhere else entirely, and using it
    silently swaps the two labels.
    """
    from .frames import spherical, vsub
    from .scene import venus_geometry
    g = venus_geometry(instant)
    venus_geo = vsub(g.venus_au, g.earth_au)
    lon_v = math.degrees(spherical(venus_geo)[0])
    lon_s = math.degrees(spherical(g.sun_au)[0])
    return "eastern" if (lon_v - lon_s) % 360.0 < 180.0 else "western"


def apparition_beats(start: Instant, span_days: float = 420.0) -> list[tuple[str, Instant]]:
    """Scan forward and name every conjunction and elongation extremum found.

    A coarse daily sample of the elongation curve, then a golden-section refine
    inside each bracketed turning point. Nothing about Venus' 584-day synodic
    period is assumed -- the turning points are simply where the sampled curve
    changes direction, so the same code would work for any inferior planet.
    """
    step = 1.0
    jds = [start.jd_ut + k * step for k in range(int(span_days / step) + 1)]
    elong = [geocentric_elongation(j) for j in jds]

    beats: list[tuple[str, Instant]] = []
    for i in range(1, len(jds) - 1):
        a, b, c = elong[i - 1], elong[i], elong[i + 1]
        if b < a and b < c:
            # the elongation minimum only locates the event; the conjunction
            # itself is up to half a day either side of it, so hand the solver a
            # window wide enough to contain the longitude crossing
            when, kind = solar_conjunction(Instant(jds[i - 1] - 2.0 * step),
                                           days=6.0 * step)
            beats.append((f"{kind} conjunction", when))
        elif b > a and b > c:
            when, _ = geocentric_greatest_elongation(Instant(jds[i - 1]), days=2.0 * step)
            beats.append((f"greatest {elongation_side(when)} elongation", when))
    return beats
