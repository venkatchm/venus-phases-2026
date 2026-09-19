# The Moon sweeps past Venus — September 2026

A physically-grounded simulation, in Python + [Taichi](https://www.taichi-lang.org/),
of three linked events in the western evening sky of September 2026:

| When | Event |
|---|---|
| **Sep 13–14** | A thin waxing crescent Moon closes on Venus, night by night. |
| **Sep 14** | The Moon passes **in front of Venus** — a lunar occultation visible from parts of **Asia, Africa and Europe**, and from nowhere else. |
| **Sep 18–24** | Venus reaches **greatest brilliancy**, around magnitude **−4.8**. |
| **Late Sep** | Venus drops toward the horizon, heading for inferior conjunction in **October**. |

Nothing about those events is hard-coded. Every position, contact time and
magnitude in this repository is **solved at run time** from analytic planetary
and lunar theory, and the code is pinned to published reference values by a
test suite. The occultation happens in the renderer for the same reason it
happens in the sky: the Moon's sphere is nearer to the observer than Venus'
sphere, and it gets in the way.

---

## Quick start

```bash
./run.sh                 # creates .venv on first run, then launches the sim
```

or, if you already have a Python 3.12 environment with `taichi` and `numpy`:

```bash
pip install -r requirements.txt
python main.py                       # interactive
python main.py --site Athens --fov 1.0
python main.py --selftest            # no display? render one frame per view to PNG
```

> **Python version.** Taichi supports CPython ≤ 3.12. `run.sh` builds the venv
> with 3.12 explicitly; a 3.13+ interpreter will fail to install Taichi.

Everything else is behind `make`:

```
make test        # ephemeris regression tests (Meeus reference values)
make report      # the full September timeline as text  -> out/report.txt
make map         # global occultation footprint         -> out/footprint.png
make sequence    # contact sheets of the occultation    -> out/sequence_*.png
make stills      # individual sky frames                -> out/*.png
make animation   # the whole story as a 30 s MP4        -> out/september2026.mp4
make preview     # fast low-res version of the same
make phases      # why Venus shows phases, as a 2:30 film -> out/venus_phases.mp4
make phases-sheet    # the apparition on one sheet        -> out/venus_phases_sheet.png
make phases-preview  # fast low-res version of the film
make phases-storyboard  # one still per shot, for review  -> out/storyboard/
make all
```

## The animation

`make animation` renders the entire month as one continuous film. The encounter
spans 19 days but the part that matters lasts 90 seconds, so **the clock is
deliberately not linear** — each storyboard segment carries its own easing:

| Segment | Covers | Clock |
|---|---|---|
| Night by night, the Moon closes in | dusk of Sep 11, 12, 13 | one frame-hold per evening |
| September 14 — the day of | 05:30 → 10:45 UT | linear, 11° field |
| Closing in | 10:45 → 11:59 UT | **decelerating** into first contact, zooming 11° → 1.5° |
| Disappearance | 11:59 → 12:03 UT | near real time, 0.3° field |
| Hidden behind the Moon | 12:03 → 13:13 UT | eased through, Moon centred |
| Reappearance | 13:13 → 13:18 UT | near real time |
| The Moon pulls away | 13:18 → 14:20 UT | **accelerating** out, zooming back to 9° |
| Late September | dusk of Sep 15 → 30 | one frame-hold per evening |

During the approach and departure a dotted track shows where the Moon has been
and is going. It is computed by converting the Moon's apparent place at each
past instant into alt/az using the *current* sidereal time, which subtracts the
Earth's rotation and leaves only the Moon's own motion against the stars — the
straight line the eye actually follows past Venus.

Two things the film is built to show that a still cannot: the disappearance
happens at the Moon's **dark** limb and the reappearance at its **bright** one
(a waxing Moon travels east, dark edge leading), and the sky darkens from full
daylight at first contact to nautical twilight by the time Venus comes back.

Encoding uses `imageio-ffmpeg`, which ships its own ffmpeg binary — nothing
system-wide is needed. Everything else in the project runs on Taichi and NumPy
alone.

---

## The three views

Press `1`, `2`, `3` to switch.

**1 — SKY.** A ray-traced view from any of 26 sites. The Moon and Venus are real
spheres at their true topocentric positions, shaded by the real direction of
sunlight, seen through a real atmosphere. The crescent's orientation, the
relative disc sizes, the moment Venus winks out and the moment it returns are
all consequences of the geometry, not of drawing code.

**2 — GEOMETRY.** The Sun–Venus–Earth triangle that sets Venus' elongation and
phase, plus a parallax inset: the lunar limb as a circle, with Venus marked
both as seen from the Earth's centre and as seen from your site. The gap
between those two marks *is* the reason an occultation is a local event.

**3 — FOOTPRINT.** Where on Earth Venus is actually hidden, coloured by whether
the sky there is dark, twilit or broad daylight.

### Controls

```
SPACE  play / pause            LEFT RIGHT  step time      UP DOWN  speed
1 2 3  switch view             N P         next/prev site , .  step size
[ ]    zoom out / in           G  ground   F  follow      T  next key moment
R      reset to just before disappearance  H  help        Q  quit
```

Time runs against the wall clock, so `UP`/`DOWN` set a genuine multiple of real
time (120x by default: the 75-minute occultation plays out in about 37 seconds).
The simulation opens ~4 minutes of simulated time before first contact, so the
disappearance happens on screen shortly after launch; `R` returns there.

---

## What the simulation computes

Run `make report`. Abridged output:

```
geocentric conjunction in right ascension: 2026-09-14 11:10:55 UT
  separation then: 31.2 arcmin
  the Moon's disc is only ~31 arcmin across, so from the Earth's centre
  Venus misses it -- only observers displaced by lunar parallax see a hit.

site          disappears  reappears    dur  Moon alt  Sun alt  sky
Chennai            12:00      13:15    75m     27.4     -0.1  civil twi
Colombo            12:03      13:27    84m     28.9     -2.6  civil twi
Kolkata            12:19      12:54    34m     16.4     -7.3  nautical twi
Cairo              09:42      11:07    85m     35.8     62.1  daylight
Athens             09:33      10:51    78m     24.2     55.3  daylight
London             09:29      10:34    65m      1.6     36.3  daylight
...
17 of the 26 listed sites see it.

brightest moment: 2026-09-24 06:43 UT at magnitude -4.80
inferior conjunction: 2026-10-24 15:39 UT at 6.5 deg from the Sun
```

That is the event as described: **hidden across Asia, Africa and Europe**, and
not from the Americas, southern Africa, East Asia or Australia. Two details the
simulation surfaces that a summary usually omits:

- **For Europe, Africa and the Middle East this is a daytime event.** The Sun is
  35–63° above the horizon. It is real and it is observable, but it needs
  optics and a clean sky, not just a glance west after dinner.
- **India and Sri Lanka get the good version.** Venus disappears in daylight and
  *reappears* from the Moon's bright limb in deepening twilight — Chennai at
  13:15 UT with the Sun already 9° below the horizon.

On greatest brilliancy: the maximum is extremely flat. Venus sits within
0.02 magnitude of its peak from about **Sep 18 to Sep 27**, which is why
different sources quote different dates. This simulation puts the formal
minimum at Sep 24, and Sep 18 at −4.79 — a difference no eye can detect.

---

## Where Taichi is actually doing the work

Taichi is not decoration here; three distinct kernels carry the load.

**1. The sky is ray-traced per pixel.** `render/sky.py` intersects the Moon and
Venus as spheres, shades the Moon with a Lommel–Seeliger law over a procedural
albedo (maria, crater speckle, ray systems from 3D value noise), gives Venus a
soft atmospheric terminator, and adds glare and diffraction spikes. With 2×2
supersampling that is ~3.7M rays per frame.

**2. The occultation footprint is a brute-force global search.**
`render/footprint.py` is the clearest case for a GPU-shaped language. The
geocentric positions of the Moon, Venus and Sun depend only on time, so they are
computed once per time step in Python and uploaded; then one kernel steps
**every point of a lat/lon grid through every time step**, computing the
topocentric separation and comparing it against the lunar limb.

```
footprint: 779M site-instants in 0.85s   (911M/s, single CPU kernel)
```

A per-site Python loop doing the same work would take hours.

**3. The orrery** draws anti-aliased orbit ellipses and sight lines per pixel.

Measured on this machine (Apple silicon, Taichi's **CPU** backend, 1280×720):

```
sky        30.7 ms/frame   32.5 fps      geometry  16.7 ms/frame   60.0 fps
footprint  20.4 ms/frame   49.0 fps
```

Pass `--arch gpu` for the Metal backend. The first frame of each view pays a
one-off Taichi JIT cost; the offline cache absorbs it on later runs.

### Two numerical traps this hit, and how they are fixed

Both are documented in the source, because both produced plausible-looking but
wrong images first:

- **Ray–sphere intersection at tiny angular radii.** Venus subtends 18″, so
  `sin²r ≈ 8×10⁻⁹` — an order of magnitude *below* f32 epsilon. The textbook
  discriminant `b² − (1 − sin²r)` rounds both terms to 1.0 and Venus' disc
  silently disappears; only its glare renders. Writing the discriminant as
  `sin²r − sin²θ` via a cross product never forms the cancelling difference.
- **`acos(dot)` for small angles.** The cosine is flat near zero separation, so
  `acos` quantises small angles into visible steps — blocky banding across a
  glare halo a few arcminutes wide. `atan2(|a×b|, a·b)` stays accurate.

Distances also span 4×10⁵ km (Moon) to 6×10⁷ km (Venus), so each body is solved
in units of *its own* distance, keeping every term in the quadratic order 1.

---

## The astronomy

The chain, following Meeus, *Astronomical Algorithms* (2nd ed.):

| Step | Source | Module |
|---|---|---|
| Time scales, ΔT, sidereal time | Espenak–Meeus polynomial | `vmsim/timescale.py` |
| Moon | truncated **ELP-2000/82** (60+60 terms) | `vmsim/moon.py` |
| Earth, Venus | truncated **VSOP87D** | `vmsim/vsop87.py` |
| Light-time, aberration, FK5 | iterated; Earth velocity by central differencing | `vmsim/planets.py` |
| Nutation, obliquity, precession | IAU 1980 / Lieske | `vmsim/frames.py` |
| Topocentric parallax | IAU 1976 ellipsoid | `vmsim/observer.py` |
| Assembly | apparent topocentric places | `vmsim/scene.py` |
| Contacts, milestones | bisection / golden section | `vmsim/events.py` |

VSOP87D and the lunar theory both work in the **ecliptic of date**, so the two
combine with no precession matrix between them. Positions are apparent and
topocentric: light-time corrected, aberrated, nutated, refracted, and offset by
the observer's position on a flattened Earth. That last step is not a refinement
here — lunar parallax is up to ~1°, more than twice the Moon's diameter, and it
is the entire reason this occultation is visible from Chennai and not Delhi.

### Accuracy, and how it is checked

`make test` compares against worked examples whose reference is the full theory:

```
Moon apparent longitude   133.162655 deg   vs Meeus 47.a  133.162655   (1e-5)
Moon distance             368409.68 km     vs Meeus 47.a  368409.7     (0.2 km)
Earth heliocentric lon     19.907374 deg   vs Meeus 25.b   19.907372   (1e-4)
Venus apparent RA         316.172853 deg   vs Meeus 33.a  316.172725   (~0.5")
Venus apparent Dec        -18.888098 deg   vs Meeus 33.a  -18.887956   (~0.5")
nutation in longitude      -3.7879"        vs Meeus 22.a   -3.788
rho sin/cos phi' (Palomar)  0.546861 / 0.836339  vs Meeus 40.a
```

**End-to-end error is well under an arcsecond**, against a lunar disc 1860″ in
radius. Contact times should be good to a few seconds; the visibility limit
lines on the footprint map to a few km. Ephemeris error is not the limiting
factor in anything this simulation claims.

An earlier draft used Standish's Keplerian elements for Venus and reproduced
Meeus 33.a only to ~35″, because a pure two-body fit cannot absorb the Earth's
periodic perturbation of Venus. Switching Venus and Earth to truncated VSOP87D
cut that by a factor of 70. The Keplerian elements survive in
`planets.kepler_heliocentric_j2000`, used only for the schematic orrery, where
arcseconds are meaningless.

---

## The phases of Venus

`make phases` renders a second, separate film that answers the question the
occultation raises but cannot settle from the ground: **why does Venus have
phases, why does it never stray far from the Sun, and why is the thin crescent
the biggest?**

It is built on the same VSOP87D series, but it needs a camera that can leave the
Earth's surface, so it uses a second renderer — `render/space.py` — that drops
the atmosphere and takes a real position in space instead of an azimuth and an
altitude.

**The beats are solved, not typed.** `vmsim.events.apparition_beats` samples
the elongation curve daily, refines every turning point it finds, and names each
one from the geometry: an elongation maximum is a greatest elongation, and a
conjunction is *superior* if Venus is farther from us than the Sun at that
instant and *inferior* if it is nearer. No date is supplied to it. For 2026 that
yields, against independently published values:

<!-- The one date this film does not derive is the occultation cross-reference
     at 2026-09-14, which is written as a constant because it belongs to the
     companion simulation rather than to this one; it is a caption marking the
     other film's subject as it goes past, not an input to any geometry here. -->

| Beat | Solved | Published | Elongation | Lit | Disc | Distance |
|---|---|---|---|---|---|---|
| Superior conjunction | 2026-01-06 16:01 UT | Jan 6 | 0.7° | 100 % | 9.8″ | 1.711 AU |
| Greatest eastern elongation | 2026-08-15 06:30 UT | Aug 15, 06 UT, 46° | 45.9° | 48.7 % | 24.3″ | 0.686 AU |
| *(the occultation, in passing)* | 2026-09-14 | — | 41.1° | 29.1 % | 36.8″ | 0.454 AU |
| Inferior conjunction | 2026-10-24 03:49 UT | Oct 24, 04 UT | 6.5° | 0.6 % | 61.1″ | 0.273 AU |

**Conjunction is solved by the definition the almanacs use**, which is not the
obvious one. It is the instant Venus and the Sun share an apparent geocentric
*ecliptic longitude* — not the instant their angular separation is smallest.
Because Venus sits 3.4° off the ecliptic at conjunction, those differ by about
ten hours: solving the separation minimum put the 2026 inferior conjunction at
14:19 UT against a published 04 UT, while the longitude crossing lands at
03:49 UT. Note that the elongation at that instant is 6.5°, not zero, and the
test asserts both halves of that — longitude difference zero, angular separation
emphatically not.

Two structural choices carry the argument:

- **One camera for every shot.** The top-down "diagram" is not a schematic drawn
  next to the 3-D scene; it is the same scene viewed from ecliptic north at long
  range through a narrow field, which is near-orthographic. The telescopic view
  is the same scene again with the camera at the Earth's centre. The geometry
  panel and the telescope panel therefore *cannot* disagree — they are one ray
  trace. The move between them is a dolly-zoom: the camera pulls from 2.5 AU to
  15 AU while the field narrows 41° → 11.5°, so the framed extent barely changes
  and the perspective flattens into the diagram.
- **One angular scale.** Every telescopic view in the second half — full frame,
  split panel, triptych panel — is built at a fixed **0.13″ per pixel**, and says
  so on screen. The disc grows 6.3× across the apparition for exactly one
  reason, and if the field were allowed to drift that would prove nothing.
  `tests/test_render_geometry.py` measures the rendered disc against
  `angular_radius_deg` and asserts the three panel widths agree.

**Rotation axes come from published pole orientations.** Every body's spin axis
is derived at import time from its IAU WGCCRE (2015) pole right ascension and
declination, converted into the ecliptic frame by the project's own
`equ_to_ecl`, rather than written out as a vector by hand. The hand-written
versions this replaced were wrong for three of four bodies — the Earth's had the
sign of its *y* component flipped, tilting the planet 47° off true while looking
entirely plausible on screen, and Venus' pointed at the *south* ecliptic pole
while also carrying a negative rotation period, two negatives that multiplied
into a Venus rotating **prograde**. `tests/test_render_geometry.py` now checks
each axis against an independently published number: the angle between a spin
axis and the ecliptic pole is fixed by obliquity plus orbital inclination, and
those are tabulated separately from the poles themselves (Mercury 7.0°, Venus
1.2°, Earth 23.44°, Mars 25.4°).

**The photometry declares its own limits.** Hilton's (2005) magnitude law is
fitted over phase angles 0–163.7°, and Venus passes beyond that within about six
days of inferior conjunction — which this film reaches. `venus_magnitude` clamps
there, so the quoted magnitude in that window is an extrapolation off the end of
the fit rather than a prediction. The HUD marks it with an asterisk and says so
on screen instead of presenting an extrapolated number as a measurement.

The phase itself is never set. Each body's light direction is computed from its
own position — with the Sun at the origin, `normalize(sun - body)` is just
`-normalize(pos)` — so the lit hemisphere cannot point anywhere but at the Sun.
The same test file checks this against rendered pixels: a partly lit sphere's
brightness centroid must sit off its geometric centre, sunward.

Venus' orbit is inclined 3.4°, so at the 2026 inferior conjunction it passes
*above* the Sun rather than across it. There is no transit, the minimum
elongation is 6.5° rather than zero, and that is left exactly as the ephemeris
gives it.

### Resuming a long render

`make phases` takes about half an hour, and on a loaded machine something may
stop it before it finishes. With `--resume` it becomes restartable:

```bash
python scripts/render_venus_phases.py --arch gpu --resume \
       --frames-dir out/phaseframes --out out/venus_phases.mp4
```

Frames are rendered to PNGs first and encoded in a second pass, so re-running
skips everything already on disk. Two details make that safe rather than merely
convenient. Frames are written to a temp file and `os.replace`d into position,
so a process killed mid-write leaves either the finished frame or nothing; and
resume checks that each cached frame actually ends with an `IEND` chunk rather
than trusting `os.path.exists`. Both exist because the first version did
neither, a killed run left a zero-byte frame, and the encode walked into it.

`imageio` can encode video but ships no PNG *reader* by default, so
`render/png.py` grew a `read_png` to match its hand-rolled writer rather than
taking on Pillow as a dependency.

### What is not to scale

The dynamic-range liberty described in the next section applies here too. Two
more are specific to this film, and it prints them on every frame that takes
them:

- **Body sizes in the orbital shots.** At 15 AU the Earth subtends under a
  thousandth of a degree. Planets are inflated ×900–1500 and the Sun ×14–22 —
  separately, because one factor for both would make the Sun wider than Venus'
  orbit. **Positions, orbit radii, distances and light directions are never
  scaled**, and the caption `BODIES X1500  SUN X14 - ORBITS AND LIGHT TO SCALE`
  is on screen whenever they are inflated. The telescopic shots run at true
  angular size, because that is where the argument about apparent size is made.
- **The corona** is brighter than life. The real K-corona is ~10⁻⁶ of the disc
  one solar radius out and would be invisible; here it is lifted, but made to
  die away within about three solar radii so that deep space stays black.
- **Venus' cloud banding** is held to a few percent on purpose. In visible light
  Venus really is a nearly featureless white ball — the famous swirls are
  ultraviolet. A detailed Venus would be a prettier image and a wrong one.


## Scientific validation

`make test` runs 74 checks in two files. Nothing in either is compared against
this project's own earlier output — every expected value comes from outside it.

| What | Checked against | Result |
|---|---|---|
| Julian day, ΔT, GMST/GAST | Meeus worked examples | exact |
| Obliquity and nutation | Meeus 22.a | < 0.1″ |
| Moon, truncated ELP-2000/82 | Meeus 47.a | < 0.1° |
| Earth & Venus, truncated VSOP87D | Meeus 25.b, 33.a | sub-arcsecond |
| Venus apparent geocentric RA/Dec | Meeus, end to end | < 0.0005° |
| Topocentric parallax | Meeus 40.a (Palomar) | < 2e-6 |
| Superior conjunction, greatest elongation, inferior conjunction | published 2026 phenomena (EarthSky) | < 30 min |
| Greatest elongation value | 45.9° published | < 0.4° |
| Rotation axes, all four bodies | IAU WGCCRE (2015) poles | < 0.4° |
| Venus rotates retrograde | IAU pole + sign of period | asserted |
| Occultation duration from Chennai | independent prediction | 75 ± 6 min |
| New York sees no occultation | visibility footprint | asserted |

Four more are checked against *rendered pixels* rather than against code, which
is the only way to know the picture inherited the physics:

| Invariant | Method | Result |
|---|---|---|
| The lit side faces the Sun | brightness centroid of a partly lit sphere must sit off its geometric centre, sunward | cos = 0.98 |
| The crescent points at the Sun | in the telescopic shots the Sun is off frame, so the sunward direction is reconstructed from the camera basis and compared with the measured lit centroid | cos = 0.999–1.000 |
| Annotations land on what was traced | projected Sun centre vs its rendered centroid | 0.7 px |
| One fixed angular scale | rendered disc diameter vs `angular_radius_deg`, at three panel widths | ≤ 2.3 px, identical across widths |
| Depth ordering | Venus placed behind the Sun must not punch through | asserted |
| Frame-cache integrity | PNG round-trip exact at both levels; truncated, zero-byte and missing frames detected as incomplete | asserted |

Two cross-checks fall out of the structure rather than being arranged:

- **Two independent paths to the same Venus.** The occultation simulation works
  topocentrically from a site on a rotating Earth; the phases film works from
  heliocentric vectors with no observer at all. At 2026-09-14 12:00 UT both give
  an illuminated fraction of 29.1438 % and elongation agreeing to 0.001°.
- **The disc growth really is distance.** Δ falls 1.711 → 0.273 AU, a factor of
  6.27; the apparent diameter rises 9.8″ → 61.2″, a factor of 6.24. Those agree
  to 0.5 %, which is what it means for the growth to be geometry and nothing
  else.

### Known limits, stated rather than hidden

- **Magnitude near inferior conjunction is extrapolated.** Hilton's (2005) law is
  fitted for phase angles 0–163.7°; Venus exceeds that within ~6 days of
  inferior conjunction, and `venus_magnitude` clamps. The HUD marks those frames
  with an asterisk.
- **Orbit paths and body positions use different frames.** The drawn ellipses
  come from Standish's Keplerian elements in the J2000 ecliptic; the bodies come
  from VSOP87D in the ecliptic of date. In 2026 the two frames differ by 0.37° of
  precession — but that displacement is almost entirely *along track*, and since
  an orbit is a closed curve the planet still lands on its own drawn path to
  within **0.4 px**. Measured, not assumed.
- **The quoted illuminated fraction is the geometric one**, `(1 + cos i)/2`,
  which is the standard definition and what every almanac tabulates. The
  *rendered* disc deliberately shows a little more lit area than that. Venus'
  upper haze carries sunlight past the geometric terminator — the reason its
  cusps extend so far at thin phases — and the shading models this with
  `smoothstep(-0.16, 0.22, mu)`, which puts the surface at 38 % brightness on
  the geometric terminator itself and reaches 9.2° beyond it. Measured against
  the rendered pixels: at a 90 %-of-peak threshold the lit area matches the
  ephemeris (46.7 % against 48.8 % at greatest elongation), and the excess is
  entirely in the soft fringe. The geometry is exact; the fringe is physics the
  caption does not try to summarise in one number.
- **Star positions** come from a hand-entered 106-star catalogue good to a few
  arcminutes — a backdrop, not a catalogue.
- **No transit in 2026.** Venus' orbit is inclined 3.4°, so at inferior
  conjunction it passes above the Sun. Minimum elongation is 6.5°, not zero, and
  that is left as the ephemeris gives it.


## Rendering choices, stated plainly

Two places where the images are deliberately *not* physical, both to make an
image a screen can show:

- **Dynamic range.** A sunlit lunar surface is ~10⁷ times brighter than the
  night sky background. A linear render at an exposure that shows the Moon
  leaves the sky and every star pure black. Sky radiances are therefore lifted
  from their true ratios, and the tone curve carries an explicit `compress`
  control (`= 1.0` is linear; see `SkyRenderer.tonemap`). Auto-exposure is keyed
  to the sky so the picture stays readable from daylight into darkness.
- **Earthshine** is physically ~10⁻⁴ of the sunlit surface; it is lifted to
  about 1/50 on the same reasoning.

Everything else is physical, including a detail worth calling out: bodies are
attenuated by atmospheric extinction and have the **full airlight added in
front of them**. That single term is why the dark limb of the daytime Moon is
simply *not there* in the 12:00 UT frame — you see the sky through it — while
the same limb shows earthshine once the sky darkens.

**The footprint map has no coastlines.** No shoreline dataset ships with this
project, so geography is anchored by 26 labelled cities over a 30° graticule.
It is a schematic, and it is labelled as one.

Star positions come from a hand-entered 106-star catalogue accurate to a few
arcminutes — fine as a backdrop, not a substitute for a real catalogue.

---

## Layout

```
main.py                  interactive app: three views over one clock
vmsim/                   astronomy — no rendering code anywhere in here
  timescale.py           JD, ΔT, GMST/GAST
  frames.py              vectors, obliquity, nutation, precession, alt/az
  moon.py                truncated ELP-2000/82
  vsop87.py              truncated VSOP87D for Earth and Venus
  planets.py             light-time, aberration, phase, magnitude, orrery orbits
  observer.py            sites and the topocentric transformation
  scene.py               assembles one apparent, topocentric sky state
  events.py              occultation contacts and apparition milestones
  stars.py               bright-star catalogue and B−V tinting
render/                  Taichi kernels — no astronomy anywhere in here
  sky.py                 per-pixel ray-traced sky, from the ground
  space.py               per-pixel ray-traced heliocentric scene, from anywhere
  footprint.py           global occultation solver + map
  orrery.py              schematic solar-system panels
  bridge.py              vmsim -> renderer fields (the only module that knows both)
  camera.py              python mirror of the sky camera, for overlays
  spacecam.py            python mirror of the space camera, plus line/arc drawing
  text.py                5×7 bitmap font, so PNGs can label themselves
  png.py                 dependency-free PNG writer
scripts/                 report.py, render_map.py, render_sequence.py,
                         render_frames.py, render_animation.py,
                         render_venus_phases.py, render_phase_stills.py
tests/test_ephemeris.py         reference-value regression tests
tests/test_render_geometry.py   renderer invariants, checked against pixels
```

The `vmsim` / `render` split is strict: the astronomy modules import nothing
from `render`, contain no Taichi, and are usable as a plain library.

---

## Sources

- Jean Meeus, *Astronomical Algorithms*, 2nd ed. — ELP-2000/82 and VSOP87
  truncations, nutation, parallax, and the worked examples used as tests.
- Bretagnon & Francou, VSOP87 planetary theory.
- Chapront-Touzé & Chapront, ELP-2000/82 lunar theory.
- Hilton (2005) / *Astronomical Almanac* — Venus' magnitude law, fitted for
  phase angles 0-163.7 deg.
- IAU WGCCRE (2015) report — rotation-pole right ascensions and declinations,
  from which every body's spin axis in the phases film is derived.
- Published 2026 Venus phenomena, used to check the solved beats from outside
  this project:
  [superior conjunction Jan 6](https://earthsky.org/astronomy-essentials/venus-superior-conjunction/),
  [greatest eastern elongation Aug 15, 46 deg](https://earthsky.org/astronomy-essentials/venus-after-sunset-greatest-elongation/),
  [inferior conjunction Oct 24, 04 UTC](https://earthsky.org/astronomy-essentials/inferior-conjunction-venus-between-sun-and-earth/).
  The last of these is what exposed the conjunction-definition error: solving
  the minimum of angular separation gave 14:19 UT, ten hours from the published
  time, and only the ecliptic-longitude crossing reproduces it.
- Event context: [EarthSky visible planets](https://earthsky.org/astronomy-essentials/visible-planets-tonight-mars-jupiter-venus-saturn-mercury/),
  [NASA JPL "What's Up" September 2026](https://science.nasa.gov/solar-system/skywatching/whats-up-september-2026-skywatching-tips-from-nasa/),
  [Space.com on Venus' evening apparition](https://www.space.com/stargazing/venus-is-disappearing-from-the-evening-sky-heres-how-to-see-it-while-you-can).
  These framed the brief; none of their numbers were copied — the simulation
  derives its own and agrees.
