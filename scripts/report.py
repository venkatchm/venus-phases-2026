"""Print the September 2026 Moon-Venus timeline computed from the ephemeris.

Nothing here is a stored fact: every number is solved for at run time from the
truncated ELP-2000/82 and VSOP87D series in `vmsim`.
"""
from __future__ import annotations

import argparse
import io
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vmsim.events import (conjunction_in_right_ascension, evening_instant,
                          find_occultation, greatest_brilliancy,
                          inferior_conjunction)
from vmsim.observer import SITES, find_site
from vmsim.scene import geocentric_bodies, sky_state
from vmsim.frames import angle_between
from vmsim.timescale import Instant, jd_to_utc_string


def rule(out, ch="-"):
    print(ch * 78, file=out)


def heading(out, text):
    print(file=out)
    rule(out, "=")
    print(text, file=out)
    rule(out, "=")


def sunset_state(day, site, minutes_after=30.0):
    """Sky state a fixed interval after the evening sunset at `site`."""
    return sky_state(evening_instant(site, 2026, 9, day, minutes_after), site)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="Chennai")
    ap.add_argument("--save", default="")
    args = ap.parse_args()
    out = io.StringIO()
    site = find_site(args.site)

    print("SEPTEMBER 2026: THE MOON SWEEPS PAST VENUS", file=out)
    print("computed from truncated ELP-2000/82 (Moon) and VSOP87D (Venus, Earth)", file=out)
    print("positions are apparent, topocentric, referred to the true equinox of date",
          file=out)

    # ------------------------------------------------------------------
    heading(out, "1. SEPTEMBER 13-14  THE APPROACH")
    print(f"{'UT':<22s}{'separation':>12s}{'Moon limb':>12s}{'Moon illum':>12s}",
          file=out)
    for day, hour in ((13, 0), (13, 6), (13, 12), (13, 18),
                      (14, 0), (14, 6), (14, 9), (14, 12), (14, 15), (14, 18), (15, 0)):
        inst = Instant.from_utc(2026, 9, day, hour)
        g = geocentric_bodies(inst)
        sep = math.degrees(angle_between(g["moon"], g["venus"]))
        st = sky_state(inst, site)
        print(f"{jd_to_utc_string(inst.jd_ut, False):<22s}"
              f"{sep:>10.3f} d{st.moon.angular_radius_deg * 3600:>10.0f}\""
              f"{st.moon.illuminated_fraction * 100:>11.1f}%", file=out)
    conj = conjunction_in_right_ascension(Instant.from_utc(2026, 9, 14, 6), hours=18)
    g = geocentric_bodies(conj)
    print(file=out)
    print(f"geocentric conjunction in right ascension: {jd_to_utc_string(conj.jd_ut)}",
          file=out)
    print(f"  separation then: "
          f"{math.degrees(angle_between(g['moon'], g['venus'])) * 60:.1f} arcmin", file=out)
    print("  the Moon's disc is only ~31 arcmin across, so from the Earth's centre", file=out)
    print("  Venus misses it -- only observers displaced by lunar parallax see a hit.",
          file=out)

    # ------------------------------------------------------------------
    heading(out, "2. SEPTEMBER 14  THE OCCULTATION, SITE BY SITE")
    print(f"{'site':<14s}{'disappears':>10s}{'reappears':>11s}{'dur':>7s}"
          f"{'Moon alt':>10s}{'Sun alt':>9s}  sky", file=out)
    seen = 0
    for s in SITES:
        r = find_occultation(s, Instant.from_utc(2026, 9, 14, 4, 0), hours=14.0)
        if not r.occurs:
            continue
        seen += 1
        d, a = r.disappearance, r.reappearance
        dd = jd_to_utc_string(d.instant.jd_ut)[11:16] if d else "  -  "
        aa = jd_to_utc_string(a.instant.jd_ut)[11:16] if a else "  -  "
        sky = ("daylight" if r.sun_altitude_at_min_deg > 0 else
               "civil twi" if r.sun_altitude_at_min_deg > -6 else
               "nautical twi" if r.sun_altitude_at_min_deg > -12 else "dark")
        print(f"{s.name:<14s}{dd:>10s}{aa:>11s}"
              f"{r.duration_seconds / 60:>6.0f}m{r.moon_altitude_at_min_deg:>9.1f}"
              f"{r.sun_altitude_at_min_deg:>9.1f}  {sky}", file=out)
    print(file=out)
    print(f"{seen} of the {len(SITES)} listed sites see it. Times are UT; the Moon", file=out)
    print("covers Venus at its dark limb and gives it back at the bright limb,", file=out)
    print("because a waxing Moon travels eastward with its dark edge leading.", file=out)

    r = find_occultation(site, Instant.from_utc(2026, 9, 14, 4, 0), hours=14.0)
    if r.occurs:
        print(file=out)
        print(f"contacts in detail for {site.name}:", file=out)
        for c in r.contacts:
            print(f"  {c}", file=out)

    # ------------------------------------------------------------------
    heading(out, "3. SEPTEMBER 18-24  GREATEST BRILLIANCY")
    inst, mag = greatest_brilliancy(site, Instant.from_utc(2026, 9, 5), days=35.0)
    print(f"brightest moment: {jd_to_utc_string(inst.jd_ut, False)} at magnitude {mag:.2f}",
          file=out)
    print("the maximum is very flat -- Venus stays within 0.02 mag of it for over a week,",
          file=out)
    print("which is why published dates for 'greatest brilliancy' vary by several days.",
          file=out)
    print(file=out)
    print(f"{'date':<12s}{'mag':>7s}{'elong':>8s}{'illum':>8s}{'diam':>8s}"
          f"{'alt 30m after sunset':>22s}", file=out)
    for day in (13, 14, 16, 18, 20, 22, 24, 26, 28, 30):
        st = sunset_state(day, site)
        print(f"2026-09-{day:02d}  {st.venus.magnitude:>6.2f}"
              f"{st.elongation_deg:>8.1f}{st.venus.illuminated_fraction * 100:>7.1f}%"
              f"{st.venus.angular_diameter_arcsec:>7.1f}\""
              f"{st.venus.apparent_altitude_deg:>21.1f}", file=out)

    # ------------------------------------------------------------------
    heading(out, "4. LATE SEPTEMBER  THE FALL TOWARDS INFERIOR CONJUNCTION")
    ic, elong = inferior_conjunction(site, Instant.from_utc(2026, 9, 25), days=60.0)
    print(f"inferior conjunction: {jd_to_utc_string(ic.jd_ut, False)} "
          f"at {elong:.1f} deg from the Sun", file=out)
    a = sunset_state(14, site)
    b = sunset_state(30, site)
    print(f"between Sep 14 and Sep 30, seen from {site.name} half an hour after sunset:",
          file=out)
    print(f"  altitude   {a.venus.apparent_altitude_deg:5.1f} deg -> "
          f"{b.venus.apparent_altitude_deg:5.1f} deg", file=out)
    print(f"  elongation {a.elongation_deg:5.1f} deg -> {b.elongation_deg:5.1f} deg", file=out)
    print(f"  disc       {a.venus.angular_diameter_arcsec:5.1f}\" -> "
          f"{b.venus.angular_diameter_arcsec:5.1f}\"  (Venus is closing on us)", file=out)
    print(f"  lit        {a.venus.illuminated_fraction * 100:5.1f}% -> "
          f"{b.venus.illuminated_fraction * 100:5.1f}%  (a thinning crescent)", file=out)
    print(file=out)
    print("Venus grows and dims at the same time: it is approaching, so the disc", file=out)
    print("swells, but the lit fraction shrinks faster. The product peaks in", file=out)
    print("late September and then falls away into the solar glare.", file=out)

    text = out.getvalue()
    print(text)
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w") as fh:
            fh.write(text)
        print(f"saved to {args.save}")


if __name__ == "__main__":
    main()
