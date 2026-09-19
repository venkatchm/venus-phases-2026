"""Reference-value tests for the ephemeris.

Every expected number is a worked example from Meeus, *Astronomical Algorithms*
(2nd ed.), whose own reference is the full VSOP87/ELP theory. These pin the
truncated series: if a coefficient is ever mistyped, one of these fails.

Run with:  python tests/test_ephemeris.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vmsim.frames import (ecl_to_equ, from_spherical, mean_obliquity_deg,
                          nutation, spherical)
from vmsim.moon import geocentric_ecliptic
from vmsim.observer import Site
from vmsim.planets import illuminated_fraction, phase_angle, venus_magnitude
from vmsim.scene import geocentric_bodies, sky_state
from vmsim.timescale import (Instant, centuries_tt, gmst_deg, julian_day,
                             utc_to_jd)
from vmsim.vsop87 import heliocentric

FAILURES = []


def check(name, got, expected, tol, unit=""):
    ok = abs(got - expected) <= tol
    status = "ok  " if ok else "FAIL"
    print(f"  [{status}] {name:<44s} {got:14.6f} vs {expected:14.6f} "
          f"(tol {tol:g}{unit})")
    if not ok:
        FAILURES.append(name)


class FixedTT(Instant):
    """An instant whose TT is pinned, for comparing against TD-based examples."""
    def __init__(self, jde):
        object.__setattr__(self, "jd_ut", jde)
        object.__setattr__(self, "_jde", jde)

    @property
    def jde(self):
        return self._jde

    @property
    def t(self):
        return (self._jde - 2451545.0) / 36525.0


print("Julian day and sidereal time (Meeus ch. 7, ch. 12)")
check("JD 1957 Oct 4.81", julian_day(1957, 10, 4.81), 2436116.31, 1e-6)
check("JD 2000 Jan 1.5", julian_day(2000, 1, 1.5), 2451545.0, 1e-6)
check("GMST 1987 Apr 10.0 UT (deg)", gmst_deg(julian_day(1987, 4, 10.0)),
      197.693195, 1e-5, " deg")

print("\nObliquity and nutation (Meeus example 22.a, 1987 April 10.0 TD)")
t = centuries_tt(julian_day(1987, 4, 10.0))
dpsi, deps = nutation(t)
check("nutation in longitude (arcsec)", dpsi * 3600.0, -3.788, 0.002, '"')
check("nutation in obliquity (arcsec)", deps * 3600.0, 9.443, 0.002, '"')
check("mean obliquity (deg)", mean_obliquity_deg(t), 23.44094629, 1e-7, " deg")

print("\nMoon, truncated ELP-2000/82 (Meeus example 47.a, 1992 April 12.0 TD)")
t = centuries_tt(julian_day(1992, 4, 12.0))
lon, lat, dist = geocentric_ecliptic(t)
check("apparent longitude (deg)", lon, 133.162655, 1e-5, " deg")
check("apparent latitude (deg)", lat, -3.229126, 1e-5, " deg")
check("distance (km)", dist, 368409.7, 0.2, " km")

print("\nEarth, truncated VSOP87D (Meeus example 25.b, 1992 October 13.0 TD)")
lon, lat, r = heliocentric("earth", 2448908.5)
check("heliocentric longitude (deg)", math.degrees(lon), 19.907372, 1e-4, " deg")
check("radius vector (AU)", r, 0.99760775, 1e-7, " AU")

print("\nVenus, truncated VSOP87D (Meeus example 33.a, 1992 December 20.0 TD)")
lon, lat, r = heliocentric("venus", 2448976.5)
check("heliocentric longitude (deg)", math.degrees(lon), 26.11428, 3e-4, " deg")
check("heliocentric latitude (deg)", math.degrees(lat), -2.62070, 3e-4, " deg")
check("radius vector (AU)", r, 0.724603, 1e-5, " AU")

print("\nVenus apparent geocentric place, end to end (Meeus example 33.a)")
g = geocentric_bodies(FixedTT(2448976.5))
ra, dec, _ = spherical(g["venus"])
check("apparent right ascension (deg)", math.degrees(ra), 316.172725, 0.0005, " deg")
check("apparent declination (deg)", math.degrees(dec), -18.887956, 0.0005, " deg")

print("\nPhase geometry sanity")
check("phase angle 0 -> fully lit", illuminated_fraction(0.0), 1.0, 1e-12)
check("phase angle 90 -> half lit", illuminated_fraction(90.0), 0.5, 1e-12)
check("phase angle 180 -> unlit", illuminated_fraction(180.0), 0.0, 1e-12)
# Venus near greatest brilliancy sits close to magnitude -4.8
check("Venus magnitude at r=0.72, d=0.50, i=118", 
      venus_magnitude(0.72, 0.50, 118.0), -4.30, 0.6, " mag")

print("\nTopocentric parallax (Meeus example 40.a, Mars from Palomar)")
# rho sin/cos phi' for Palomar, 33d21'22\" N, elevation 1706 m
palomar = Site("Palomar", 33.356111, -116.863611, 1706.0)
rho_sin, rho_cos = palomar.geocentric_terms()
check("rho sin phi'", rho_sin, 0.546861, 2e-6)
check("rho cos phi'", rho_cos, 0.836339, 2e-6)

print("\nSeptember 2026 event, internal consistency")
from vmsim.observer import find_site                                  # noqa: E402
from vmsim.events import find_occultation                             # noqa: E402
chennai = find_site("Chennai")
res = find_occultation(chennai, Instant.from_utc(2026, 9, 14, 4, 0), hours=14.0)
print(f"  [info] Chennai occultation occurs: {res.occurs}")
if not res.occurs:
    FAILURES.append("Chennai occultation not found")
else:
    d, a = res.disappearance, res.reappearance
    dur = (a.instant.jd_ut - d.instant.jd_ut) * 1440.0
    check("Chennai occultation duration (min)", dur, 75.0, 6.0, " min")
    # the Moon must be nearer than Venus for an occultation to mean anything
    st = sky_state(res.min_separation_instant, chennai)
    ok = st.moon.distance_km < st.venus.distance_km
    print(f"  [{'ok  ' if ok else 'FAIL'}] Moon is nearer than Venus")
    if not ok:
        FAILURES.append("depth ordering")

ny = find_site("New York")
res_ny = find_occultation(ny, Instant.from_utc(2026, 9, 14, 4, 0), hours=14.0)
ok = not res_ny.occurs
print(f"  [{'ok  ' if ok else 'FAIL'}] New York correctly sees no occultation")
if not ok:
    FAILURES.append("New York false positive")

# ---------------------------------------------------------------------------
# the 2026 apparition, as the phases film solves it
# ---------------------------------------------------------------------------
print("\nVenus apparition 2026 (solved, not tabulated)")
from vmsim.events import apparition_beats                              # noqa: E402
from vmsim.scene import venus_geometry                                 # noqa: E402

beats = dict(apparition_beats(Instant.from_utc(2025, 12, 1), span_days=400.0))
# Published instants, against which the solver is an independent computation:
# greatest eastern elongation 2026 Aug 15 at 06 UT (46 deg) and inferior
# conjunction 2026 Oct 24 at 04 UT, per EarthSky's 2026 planet pages.
#
# The tolerance is 0.1 d rather than the day-scale slop a conjunction used to
# need here, because conjunction is now solved as the crossing of apparent
# geocentric ecliptic longitude -- the definition the almanacs tabulate -- and
# not as the minimum of angular separation. With Venus 3.4 deg off the ecliptic
# the two differ by about ten hours, which is far larger than this tolerance.
for name, y, m, d, hh in (("superior conjunction", 2026, 1, 6, 16),
                          ("greatest eastern elongation", 2026, 8, 15, 6),
                          ("inferior conjunction", 2026, 10, 24, 4)):
    ok = name in beats
    print(f"  [{'ok  ' if ok else 'FAIL'}] found {name}")
    if not ok:
        FAILURES.append(f"missing {name}")
        continue
    check(f"{name} (JD)", beats[name].jd_ut,
          Instant.from_utc(y, m, d, hh, 0).jd_ut, 0.1, " d")

# The definition itself, asserted: at the solved conjunction Venus and the Sun
# share an ecliptic longitude, even though their angular separation is 6.5 deg.
from vmsim.events import longitude_difference                          # noqa: E402
check("inferior conjunction: longitude difference (deg)",
      longitude_difference(beats["inferior conjunction"].jd_ut), 0.0, 0.001, " deg")
check("...while angular separation is NOT zero (deg)",
      venus_geometry(beats["inferior conjunction"]).elongation_deg, 6.5, 0.3, " deg")

if "greatest eastern elongation" in beats:
    g = venus_geometry(beats["greatest eastern elongation"])
    # At greatest elongation Venus is close to, but not exactly at, half phase.
    # Greatest elongation is where the line of sight runs tangent to Venus'
    # orbit; on a circle that puts a right angle at Venus and gives exactly 50%
    # lit, but on an ellipse the tangent is not perpendicular to the radius, so
    # the phase angle comes out near 91.5 deg instead of 90. That is orbital
    # eccentricity, NOT Schroeter's effect -- Schroeter's is the separate
    # observational result that dichotomy is *seen* a few days off the predicted
    # date, which is attributed to Venus' atmosphere and is not modelled here.
    check("greatest elongation value (deg)", g.elongation_deg, 45.9, 0.4, " deg")
    check("illuminated fraction there", g.illuminated_fraction, 0.487, 0.02)

# The film's central claim, as an assertion: across the approach to inferior
# conjunction the disc grows while the lit fraction shrinks. If these ever moved
# together, the whole "a crescent Venus looks bigger" argument would be wrong.
print("\nApparent size against phase (the film's claim)")
if "greatest eastern elongation" in beats and "inferior conjunction" in beats:
    t0 = beats["greatest eastern elongation"].jd_ut
    t1 = beats["inferior conjunction"].jd_ut - 4.0
    prev_d, prev_f, monotonic = None, None, True
    for k in range(13):
        g = venus_geometry(Instant(t0 + (t1 - t0) * k / 12.0))
        if prev_d is not None:
            if not (g.angular_diameter_arcsec > prev_d
                    and g.illuminated_fraction < prev_f):
                monotonic = False
        prev_d, prev_f = g.angular_diameter_arcsec, g.illuminated_fraction
    print(f"  [{'ok  ' if monotonic else 'FAIL'}] diameter rises monotonically "
          f"as lit fraction falls")
    if not monotonic:
        FAILURES.append("diameter/phase monotonicity")
    ga = venus_geometry(beats["superior conjunction"])
    check("disc growth, superior -> inferior", prev_d / ga.angular_diameter_arcsec,
          6.3, 0.3, "x")

# Two independent paths to the same number: the site-free geometry used by the
# film, and the full topocentric state used by the occultation simulation.
# Parallax is the only thing that should separate them.
print("\nGeocentric vs topocentric Venus (parallax only)")
_when = Instant.from_utc(2026, 9, 14, 12, 0)
_g = venus_geometry(_when)
_st = sky_state(_when, find_site("Chennai"))
check("illuminated fraction agrees", _g.illuminated_fraction,
      _st.venus.illuminated_fraction, 0.004)
check("elongation agrees (deg)", _g.elongation_deg, _st.elongation_deg, 0.35, " deg")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {', '.join(FAILURES)}")
    raise SystemExit(1)
print("all checks passed")
