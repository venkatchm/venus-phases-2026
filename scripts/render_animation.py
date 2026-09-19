"""Render the whole September 2026 story as a single time-ramped animation.

The encounter spans 17 days but the part that matters most lasts 90 seconds, so
the clock is deliberately not linear. Each storyboard segment carries its own
easing: the approach decelerates into first contact, the contacts themselves run
at close to real time, and the departure accelerates away again before the film
switches to one frame per evening for the rest of the month.

    python scripts/render_animation.py                # 1280x720, ~24 s
    python scripts/render_animation.py --preview      # quick low-res check
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import taichi as ti

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--site", default="Chennai")
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--samples", type=int, default=2, help="supersampling, per axis")
ap.add_argument("--arch", default="cpu")
ap.add_argument("--out", default="out/september2026.mp4")
ap.add_argument("--frames-dir", default="", help="also write individual PNG frames here")
ap.add_argument("--preview", action="store_true", help="640x360, fewer frames, 1 sample")
ap.add_argument("--scale", type=float, default=1.0, help="multiply every segment's frame count")
args = ap.parse_args()

if args.preview:
    args.width, args.height, args.samples, args.scale = 640, 360, 1, 0.34

ti.init(arch=getattr(ti, args.arch), offline_cache=True, log_level=ti.WARN)

from render.bridge import apply_state, load_star_field          # noqa: E402
from render.camera import project, splat                        # noqa: E402
from render.png import write_png                                # noqa: E402
from render.sky import SkyRenderer                              # noqa: E402
from render.text import draw_text, text_width                   # noqa: E402
from vmsim.events import evening_instant, find_occultation      # noqa: E402
from vmsim.frames import equ_to_altaz, from_spherical           # noqa: E402
from vmsim.observer import find_site                            # noqa: E402
from vmsim.scene import sky_state                               # noqa: E402
from vmsim.timescale import Instant, jd_to_utc_string           # noqa: E402

DEG = math.pi / 180.0
site = find_site(args.site)

# ---------------------------------------------------------------------------
# solve the event first: the storyboard is pinned to the real contact times
# ---------------------------------------------------------------------------
occ = find_occultation(site, Instant.from_utc(2026, 9, 14, 4, 0), hours=14.0)
if not occ.occurs:
    raise SystemExit(f"no occultation visible from {site.name}; try --site Chennai")
D = occ.disappearance.instant.jd_ut
R = occ.reappearance.instant.jd_ut
print(f"{site.name}: disappearance {jd_to_utc_string(D)}  reappearance {jd_to_utc_string(R)}")

MIN = 1.0 / 1440.0
SEC = 1.0 / 86400.0


def linear(u):
    return u


def ease_out(u):        # decelerate: simulated time crawls at the end
    return 1.0 - (1.0 - u) ** 3


def ease_in(u):         # accelerate: crawls at the start
    return u ** 3


def ease_in_out(u):
    return u * u * (3.0 - 2.0 * u)


# A segment is either a continuous span of time with its own easing, or a
# series of evenings held one at a time. Both kinds yield the same frame tuple.
#
# Continuous: (label, jd0, jd1, seconds, easing, fov0, fov1, target, ground, trail)
# Evenings:   (label, [days], seconds, fov, target, ground, trail)

APPROACH_EVENINGS = [11, 12, 13]

print("solving evening twilights...")
EVE = {d: evening_instant(site, 2026, 9, d, 30.0).jd_ut
       for d in APPROACH_EVENINGS}

CONTINUOUS = [
    ("SEPTEMBER 14 - THE DAY OF", Instant.from_utc(2026, 9, 14, 5, 30).jd_ut,
     D - 75 * MIN, 5.0, linear, 11.0, 11.0, "venus", False, True),
    ("CLOSING IN", D - 75 * MIN, D - 90 * SEC,
     6.0, ease_out, 11.0, 1.5, "venus", False, True),
    # The two contacts are the event. Everything else is approach and departure,
    # so these get the time: nine seconds each, near real speed, close in.
    ("DISAPPEARANCE", D - 90 * SEC, D + 150 * SEC,
     9.0, linear, 0.30, 0.30, "venus", False, False),
    ("HIDDEN BEHIND THE MOON", D + 150 * SEC, R - 120 * SEC,
     5.0, ease_in_out, 1.30, 1.30, "moon", False, False),
    ("REAPPEARANCE", R - 120 * SEC, R + 180 * SEC,
     9.0, linear, 0.30, 0.30, "venus", False, False),
    ("THE MOON PULLS AWAY", R + 180 * SEC, R + 65 * MIN,
     5.0, ease_in, 1.5, 9.0, "venus", True, True),
]

EVENING_FOV = 50.0


def evening_frames(label, days, seconds, fov):
    """One dusk per day, held, drifting a few minutes deeper into twilight.

    Interpolating the *instant* between consecutive evenings would put frames
    at three in the morning with nothing above the horizon, so each evening is
    held instead and the series steps from one to the next.
    """
    total = max(2, int(round(seconds * args.fps * args.scale)))
    per = max(2, total // len(days))
    for i, d in enumerate(days):
        for k in range(per):
            u = (i * per + k) / (len(days) * per - 1)
            jd = EVE[d] + (k / (per - 1)) * 7.0 * MIN
            yield label, jd, fov, "west", True, False, u


def storyboard_frames():
    """Yield (label, jd, fov, target, ground, trail, progress) per frame."""
    yield from evening_frames("NIGHT BY NIGHT, THE MOON CLOSES IN",
                              APPROACH_EVENINGS, 4.5, EVENING_FOV)
    for (label, t0, t1, seconds, ease, fov0, fov1, target, ground, trail) in CONTINUOUS:
        n = max(2, int(round(seconds * args.fps * args.scale)))
        for k in range(n):
            u = k / (n - 1)
            jd = t0 + (t1 - t0) * ease(u)
            fov = fov0 * (fov1 / fov0) ** ease_in_out(u)      # geometric: reads as a zoom
            yield label, jd, fov, target, ground, trail, u


FRAMES = list(storyboard_frames())
print(f"{len(FRAMES)} frames -> {len(FRAMES) / args.fps:.1f} s at {args.fps} fps")

# ---------------------------------------------------------------------------
# a dense track of the Moon's apparent place, for the straight-line trail
# ---------------------------------------------------------------------------
print("precomputing the Moon's track...")
TRACK_STEP = 12.0 * MIN
track_jd, track_radec = [], []
t = Instant.from_utc(2026, 9, 13, 0, 0).jd_ut
while t <= Instant.from_utc(2026, 9, 15, 12, 0).jd_ut:
    st = sky_state(Instant(t), site)
    track_jd.append(t)
    track_radec.append((st.moon.ra_deg, st.moon.dec_deg))
    t += TRACK_STEP
track_jd = np.array(track_jd)
track_radec = np.array(track_radec)


def draw_trail(img, state, cam_az, cam_alt, fov, hours_back=20.0, hours_fwd=6.0):
    """Plot where the Moon has been and is going, as one straight line.

    Past positions are converted to alt/az using the *current* sidereal time.
    That subtracts the Earth's rotation and leaves only the Moon's own motion
    against the stars -- which is the straight line the eye actually follows.
    """
    jd = state.instant.jd_ut
    lo, hi = jd - hours_back / 24.0, jd + hours_fwd / 24.0
    sel = (track_jd >= lo) & (track_jd <= hi)
    if not sel.any():
        return
    lst = state.local_sidereal_deg
    for k, (ra, dec) in zip(track_jd[sel], track_radec[sel]):
        v = from_spherical(math.radians(ra), math.radians(dec), 1.0)
        alt, az = equ_to_altaz(v, site.latitude, lst)
        d = np.array([math.cos(alt * DEG) * math.cos(az * DEG),
                      math.cos(alt * DEG) * math.sin(az * DEG),
                      math.sin(alt * DEG)])
        p = project(d, cam_az, cam_alt, fov, img.shape[1], img.shape[0])
        if p is None:
            continue
        past = k <= jd
        colour = (0.30, 0.42, 0.62) if past else (0.22, 0.24, 0.32)
        splat(img, p[0], p[1], colour, radius=1.5, alpha=0.55 if past else 0.30)


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------
renderer = SkyRenderer(args.width, args.height)
renderer.samples[None] = args.samples

import imageio.v2 as imageio                                     # noqa: E402

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
if args.frames_dir:
    os.makedirs(args.frames_dir, exist_ok=True)
writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                            quality=8, macro_block_size=1,
                            ffmpeg_params=["-pix_fmt", "yuv420p"])

scale = max(1, args.width // 640)
bar = 34 * scale // 1
last_star_key = None
for n, (label, jd, fov, target, ground, trail, u) in enumerate(FRAMES):
    st = sky_state(Instant(jd), site)

    key = int(jd * 12)
    if key != last_star_key:
        load_star_field(renderer, st)
        last_star_key = key

    if target == "venus":
        cam_az, cam_alt = st.venus.azimuth_deg, st.venus.apparent_altitude_deg
    elif target == "moon":
        cam_az, cam_alt = st.moon.azimuth_deg, st.moon.apparent_altitude_deg
    else:
        # Evening framing: centred on Venus in azimuth, dropped low enough that
        # the horizon stays in shot as Venus sinks towards it. Pointing at the
        # Sun's azimuth instead would push Venus into the corner of the frame.
        cam_az = st.venus.azimuth_deg          # drifts only ~1 deg all month
        cam_alt = 11.0                          # fixed, so Venus visibly descends

    renderer.set_camera(cam_az, cam_alt, fov)
    apply_state(renderer, st, show_ground=ground, twinkle=n * 0.07)
    img = np.transpose(renderer.frame(), (1, 0, 2))[::-1].astype(np.float32)
    img = np.ascontiguousarray(img)

    if trail:
        draw_trail(img, st, cam_az, cam_alt, fov)

    # ---- caption -------------------------------------------------------
    img[:bar] *= 0.18
    img[bar:bar + 1] = (0.30, 0.34, 0.42)
    draw_text(img, 10 * scale, 5 * scale, label, (1.0, 1.0, 1.0), 2 * scale)
    stamp = jd_to_utc_string(jd)
    draw_text(img, 10 * scale, 21 * scale, f"{stamp}   {site.name.upper()}",
              (0.70, 0.76, 0.88), 1 * scale)

    # The evening segments hold one dusk per day and step to the next, so the
    # sky jumps. Without a word of explanation that reads as a broken render
    # rather than as sixteen consecutive nights.
    if label.startswith("LATE") or label.startswith("NIGHT"):
        note = "ONE FRAME PER EVENING - SAME MOMENT AFTER SUNSET EACH NIGHT"
        draw_text(img, (args.width - text_width(note, 1 * scale)) // 2,
                  bar + 8 * scale, note, (0.62, 0.68, 0.82), 1 * scale)

    if st.is_occulted:
        right = f"VENUS HIDDEN  {abs(st.limb_distance_arcsec):.0f}\" INSIDE THE LIMB"
        rc = (1.0, 0.80, 0.42)
    elif label.startswith("LATE") or label.startswith("NIGHT"):
        right = (f"VENUS MAG {st.venus.magnitude:+.2f}  {st.venus.illuminated_fraction * 100:.0f}% LIT"
                 f"  ALT {st.venus.apparent_altitude_deg:.1f} DEG   "
                 f"MOON {st.separation_deg:.0f} DEG AWAY")
        rc = (0.85, 0.90, 1.0)
    else:
        sep = (f"{st.separation_deg:.1f} DEG" if st.separation_deg >= 1.0
               else f"{st.separation_arcsec:.0f}\"")
        right = (f"SEPARATION {sep}   MAG {st.venus.magnitude:+.2f}   "
                 f"SUN ALT {st.sun.apparent_altitude_deg:+.0f} DEG")
        rc = (0.78, 0.84, 0.95)
    draw_text(img, args.width - text_width(right, 1 * scale) - 10 * scale, 21 * scale,
              right, rc, 1 * scale)

    # progress bar across the very bottom
    done = int((n + 1) / len(FRAMES) * args.width)
    img[-3 * scale:, :done] = (0.42, 0.55, 0.80)

    rgb = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    writer.append_data(rgb)
    if args.frames_dir:
        write_png(os.path.join(args.frames_dir, f"f{n:05d}.png"), rgb)
    if n % 25 == 0 or n == len(FRAMES) - 1:
        print(f"  {n + 1:4d}/{len(FRAMES)}  {label:<30s} {stamp}")

writer.close()
size = os.path.getsize(args.out) / 1e6
print(f"wrote {args.out}  ({len(FRAMES)} frames, {len(FRAMES) / args.fps:.1f} s, {size:.1f} MB)")
