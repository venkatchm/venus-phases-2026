# The Phases of Venus — why they happen, solved rather than drawn

A physically-grounded visualisation, in Python + [Taichi](https://www.taichi-lang.org/),
of why Venus shows phases, why it never strays far from the Sun, and why the thin
crescent looks *bigger* than the full disc.

**Start here: `media/venus_the_whole_story.mp4`** — 1920×1080, 3:25. One file,
the whole thing, with title cards: why Venus shows phases, then the night the
Moon passed in front of it. This is the version to watch or upload.

The parts are also there separately if you want them: `venus_phases.mp4` (2:30),
`september2026.mp4` (30 s), `venus_phases_preview.mp4` (45 s, small), and
`venus_phases_sheet.png`, which puts the whole apparition on one page.

---

## The claim

Nothing in this is drawn by hand or keyed to an artist's judgement. Every
position comes from a truncated VSOP87D series evaluated at run time; the phase
of Venus falls out of the ray trace because the Sun is the only light source and
each body's light direction is derived from its own position. The three moments
the film is built around are **solved**, not typed in as dates.

| Beat | Solved by this project | Published | Difference |
|---|---|---|---|
| Superior conjunction | 2026-01-06 16:01 UT | Jan 6 | **76 seconds** |
| Greatest eastern elongation | 2026-08-15 06:30 UT, 45.90° | Aug 15 06 UT, 46° | **30 minutes** |
| Inferior conjunction | 2026-10-24 03:49 UT | Oct 24 04 UT | **10 minutes** |

## The argument the film has to land

A crescent Venus appears **larger** than a full one, because it is nearer — not
because it is growing. That only means anything if the viewer can trust the
scale, so every telescopic view in the second half is locked to a fixed
**0.13 arcseconds per pixel** and says so on screen. Across the apparition the
disc grows **6.3×** while the lit fraction falls from 100 % to under 1 %.

`tests/test_render_geometry.py` measures the *rendered pixels* against the
ephemeris and asserts the three panel widths agree.

## Verification: 74 automated checks

Run `make test`. Nothing is compared against this project's own earlier output —
every expected value comes from outside it: Meeus' worked examples, the IAU
WGCCRE pole tables, and published 2026 phenomena.

Four checks measure rendered pixels rather than code, because that is the only
way to know the picture inherited the physics:

- **The lit side faces the Sun** — the brightness centroid of a partly lit sphere
  must sit off its geometric centre, sunward. Measured cos = 0.98.
- **The crescent points at the Sun** in the telescopic shots, where the Sun is
  off-frame and so nothing in the image could have been fitted to it. r = 0.999–1.000.
- **One fixed angular scale** — rendered disc diameter against `angular_radius_deg`,
  identical at 1920 / 960 / 640 px panel widths.
- **Depth ordering** — Venus behind the Sun must not punch through.

## Errors found and fixed during review

Each of these rendered plausibly while being wrong, which is the point of
checking against outside sources rather than against appearances:

1. **Earth's rotation axis pointed to ecliptic longitude 270°** instead of 90° —
   46.9° off true. Mars was 33.5° off, Mercury 7.0°. All four poles are now
   derived from IAU WGCCRE right ascensions and declinations rather than typed.
2. **Venus rotated prograde.** A south-pointing axis multiplied by a negative
   period is two negatives; Venus famously turns retrograde.
3. **"Inferior conjunction" used the wrong definition** — the minimum of angular
   separation rather than the crossing of apparent geocentric ecliptic longitude
   that almanacs tabulate. Because Venus sits 3.4° off the ecliptic, those differ
   by about ten hours.

## Known limits, stated rather than hidden

- Magnitude within ~6 days of inferior conjunction is **extrapolated** past
  Hilton's (2005) phase-angle fit. The HUD marks those frames with an asterisk.
- Body sizes in the orbital shots are **exaggerated** (planets ×900–1500, Sun
  ×14–22, separately). Positions, orbit radii, distances and light directions are
  never scaled, and the factor is printed on every frame that uses it.
- **No transit in 2026.** Venus' 3.4° orbital inclination puts it above the Sun at
  inferior conjunction; minimum elongation is 6.5°, not zero, and that is left as
  the ephemeris gives it.
- Lunar surface features are at their **real selenographic coordinates with real
  albedos**, but their outlines are procedural — Mare Crisium is a correctly sized
  circle in the correct place, not a traced boundary.

## Also included

`september2026.mp4` — the companion simulation this work grew out of: the lunar
occultation of Venus on 2026 September 14, visible from Asia, Africa and Europe.
The Moon there is tidally locked with 14 named maria and the rayed craters, and
craters cast real shadows near the terminator.

## Where to look

| File | |
|---|---|
| `WATCHING.md` | shot-by-shot guide to the films — what each shot establishes and what to check |
| `REPRODUCE.md` | how every file in `media/` was made, with commands and timings |
| `README.md` | full technical documentation, including *Scientific validation* and *Known limits* |
| `source/rebuild.sh` | one command that regenerates every artifact from scratch |

## Reproducing it

```bash
./run.sh                 # builds the venv, launches the interactive simulation
make test                # the 74 checks
make phases              # re-render the film (~30 min)
make phases-sheet        # the contact sheet (~1 min)
```

Full technical detail, including the *Scientific validation* and *Known limits*
sections, is in `README.md`.
