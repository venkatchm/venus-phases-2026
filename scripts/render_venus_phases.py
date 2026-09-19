"""The phases of Venus, explained by the geometry that causes them.

A ~2.5 minute film that starts outside the inner solar system, works inward to a
top-down view of the Sun-Venus-Earth triangle, and then cuts back and forth
between that triangle and the telescopic disc it produces, at the three moments
that define an apparition.

    python scripts/render_venus_phases.py --arch gpu    # 1920x1080, ~2:30
    python scripts/render_venus_phases.py --preview     # quick low-res check

Nothing in it is drawn by hand. Every position comes from the same VSOP87D
series the rest of this project uses, the phase falls out of the trace because
the Sun is the only light, and the three beats are solved before the first frame
is rendered rather than typed in as dates. Two honest departures from literal
physics, both announced on screen: in the orbital shots the bodies are inflated
so they are visible at all, and the tone curve is not linear. Neither touches
where anything is or which way the light runs.

The argument the film has to land is that a crescent Venus looks *bigger*. That
is why every telescopic shot in the second half is locked to one angular scale
in arcseconds per pixel -- if the field were allowed to drift, the growth would
prove nothing.
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
ap.add_argument("--width", type=int, default=1920)
ap.add_argument("--height", type=int, default=1080)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--samples", type=int, default=2, help="supersampling, per axis")
ap.add_argument("--arch", default="cpu")
ap.add_argument("--out", default="out/venus_phases.mp4")
ap.add_argument("--frames-dir", default="", help="also write individual PNG frames here")
ap.add_argument("--preview", action="store_true", help="960x540, fewer frames, 1 sample")
ap.add_argument("--scale", type=float, default=1.0, help="multiply every shot's frame count")
ap.add_argument("--year", type=int, default=2026, help="apparition to film")
ap.add_argument("--storyboard", default="",
                help="render one frame per shot to this directory and exit")
ap.add_argument("--resume", action="store_true",
                help="with --frames-dir, skip frames already on disk and encode "
                     "from the PNGs at the end; makes a long render restartable")
args = ap.parse_args()

if args.preview:
    args.width, args.height, args.samples = 960, 540, 1
    if args.scale == 1.0:            # respect an explicit --scale
        args.scale = 0.3

ti.init(arch=getattr(ti, args.arch), offline_cache=True, log_level=ti.WARN)

from render.bridge import (heliocentric_scene, orbit_paths,  # noqa: E402
                           point_flux, radius_au, spin_phase)
from render.png import png_complete, read_png, write_png              # noqa: E402
from render.space import SUN, VENUS, SpaceRenderer                    # noqa: E402
from render.spacecam import (arc_label_point, basis, draw_arc,        # noqa: E402
                             draw_line, project)
from render.text import draw_text, text_width                         # noqa: E402
from vmsim.events import apparition_beats, elongation_side            # noqa: E402
from vmsim.planets import geocentric_of_date, heliocentric_of_date    # noqa: E402
from vmsim.scene import venus_geometry                                # noqa: E402
from vmsim.stars import precessed_equatorial                          # noqa: E402
from vmsim.timescale import Instant, jd_to_utc_string                 # noqa: E402

DEG = math.pi / 180.0
W, H = args.width, args.height

_geom_cache: dict = {}


def geometry(jd):
    """`venus_geometry`, memoised. Each call is a full VSOP87 evaluation with
    four light-time iterations, and a frame asks for the same instant from the
    annotation code and the HUD both."""
    key = round(jd, 6)
    g = _geom_cache.get(key)
    if g is None:
        if len(_geom_cache) > 4096:
            _geom_cache.clear()
        g = _geom_cache[key] = venus_geometry(Instant(jd))
    return g


_path_cache: dict = {}


def cached_paths(instant, names):
    """Orbit ellipses, computed once.

    The elements drift by parts in ten thousand over the film's ten months --
    far below a pixel -- so recomputing 7000 points per frame bought nothing and
    cost most of the frame time. `main.py` caches these by the day for the same
    reason."""
    key = tuple(names)
    if key not in _path_cache:
        _path_cache[key] = orbit_paths(instant, names)
    return _path_cache[key]

# The invariant the second half of the film rests on. Every telescopic view --
# full frame, split panel, triptych panel -- is built at this angular scale, so
# a disc that covers more pixels is genuinely subtending a larger angle.
ARCSEC_PER_PX = 0.13


def scope_fov(width_px: int) -> float:
    return width_px * ARCSEC_PER_PX / 3600.0


# ---------------------------------------------------------------------------
# solve the apparition first: the storyboard is pinned to real instants
# ---------------------------------------------------------------------------
print("solving the apparition...")
_search = Instant.from_utc(args.year - 1, 12, 1)
BEATS = dict(apparition_beats(_search, span_days=400.0))
try:
    T_SUP = BEATS["superior conjunction"].jd_ut
    T_INF = BEATS["inferior conjunction"].jd_ut
    T_ELO = next(v.jd_ut for k, v in BEATS.items() if k.startswith("greatest"))
except (KeyError, StopIteration):
    raise SystemExit(f"could not find a full apparition starting {args.year - 1}-12")

ELO_SIDE = elongation_side(Instant(T_ELO))
# the event the rest of this project is about, if it falls inside the window
T_OCC = Instant.from_utc(2026, 9, 14, 12, 0).jd_ut
MARK_OCC = T_ELO < T_OCC < T_INF and args.year == 2026

for name, when in sorted(BEATS.items(), key=lambda kv: kv[1].jd_ut):
    g = venus_geometry(when)
    print(f"  {name:32s} {jd_to_utc_string(when.jd_ut, seconds=False)}"
          f"   elong {g.elongation_deg:5.2f} deg   illum {g.illuminated_fraction * 100:5.1f}%"
          f"   diam {g.angular_diameter_arcsec:5.1f}\"   delta {g.delta_au:.3f} AU")

_gs, _ge, _gi = (venus_geometry(Instant(t)) for t in (T_SUP, T_ELO, T_INF))
print(f"  disc grows x{_gi.angular_diameter_arcsec / _gs.angular_diameter_arcsec:.1f} "
      f"from superior to inferior conjunction, at a fixed {ARCSEC_PER_PX}\"/px")


# ---------------------------------------------------------------------------
# easing
# ---------------------------------------------------------------------------
def linear(u):
    return u


def ease_in(u):
    return u ** 3


def ease_out(u):
    return 1.0 - (1.0 - u) ** 3


def ease_in_out(u):
    return u * u * (3.0 - 2.0 * u)


def lerp(a, b, u):
    return a + (b - a) * u


def geom(a, b, u):
    """Geometric interpolation -- the one that reads as a smooth zoom or dolly."""
    return a * (b / a) ** u


# ---------------------------------------------------------------------------
# storyboard
# ---------------------------------------------------------------------------
# A shot is a dict so the table stays readable with this many knobs. Camera
# placement for the orbital shots is spherical about the Sun, with the longitude
# tied to the Earth's own: that keeps the triangle framed the same way while the
# planets move, so the only thing the viewer sees change is the geometry itself.
SHOTS = [
    dict(label="THE INNER SOLAR SYSTEM", mode="orbital", seconds=12.0,
         jd=(T_SUP - 46.0, T_SUP - 34.0), ease=linear,
         dist=(7.6, 5.2), elev=(38.0, 37.0), azim=-104.0, fov=(46.0, 46.0),
         pscale=900.0, sscale=22.0, context=True, fade_in=2.2),

    dict(label="VENUS ORBITS INSIDE US", mode="orbital", seconds=16.0,
         jd=(T_SUP - 34.0, T_SUP - 6.0), ease=linear,
         dist=(5.2, 3.3), elev=(37.0, 33.0), azim=-104.0, fov=(46.0, 43.0),
         pscale=900.0, sscale=22.0, context=True),

    dict(label="TWO ORBITS, ONE LIGHT", mode="orbital", seconds=17.0,
         jd=(T_SUP - 6.0, T_SUP + 26.0), ease=linear,
         dist=(3.3, 2.5), elev=(33.0, 28.0), azim=(-104.0, -96.0), fov=(43.0, 41.0),
         pscale=900.0, sscale=22.0),

    # A dolly-zoom: the camera pulls a long way back while the field narrows, so
    # the framed extent barely changes and the perspective flattens into the
    # orthographic view the explanatory shot needs. It is the transition from a
    # picture of the solar system to a diagram of it, done in one move.
    dict(label="FROM ECLIPTIC NORTH", mode="orbital", seconds=15.0,
         jd=(T_SUP + 26.0, T_SUP + 54.0), ease=linear,
         dist=(2.5, 15.0), elev=(28.0, 89.9), azim=(-96.0, -90.0), fov=(41.0, 11.5),
         pscale=(900.0, 1500.0), sscale=(22.0, 14.0), annotate=True),

    dict(label="THE ANGLE CHANGES", mode="orbital", seconds=25.0,
         jd=(T_SUP + 54.0, T_INF - 4.0), ease=linear,
         dist=15.0, elev=89.9, azim=-90.0, fov=11.5,
         pscale=1500.0, sscale=14.0, annotate=True, wedge=True),

    dict(label="WHAT THE EARTH SEES", mode="earthview", seconds=10.0,
         jd=(T_ELO - 6.0, T_ELO + 6.0), ease=linear, fov=(36.0, 32.0),
         tilt=31.0, standoff=2.2),

    dict(label="SUPERIOR CONJUNCTION", mode="split", seconds=12.0,
         jd=(T_SUP - 9.0, T_SUP + 9.0), ease=linear,
         dist=15.0, elev=89.9, azim=-90.0, fov=11.5,
         pscale=1500.0, sscale=14.0, annotate=True, cut_at=0.62),

    dict(label=f"GREATEST {ELO_SIDE.upper()} ELONGATION", mode="split", seconds=12.0,
         jd=(T_ELO - 9.0, T_ELO + 9.0), ease=linear,
         dist=15.0, elev=89.9, azim=-90.0, fov=11.5,
         pscale=1500.0, sscale=14.0, annotate=True, cut_at=0.62),

    dict(label="TOWARDS INFERIOR CONJUNCTION", mode="split", seconds=12.0,
         jd=(T_ELO + 20.0, T_INF - 6.0), ease=linear,
         dist=15.0, elev=89.9, azim=-90.0, fov=11.5,
         pscale=1500.0, sscale=14.0, annotate=True, cut_at=0.70),

    dict(label="SAME SCALE, SAME FIELD", mode="triptych", seconds=10.0,
         jd=(T_INF - 6.0, T_INF - 6.0), ease=linear),

    # the closing shot holds Venus and the Sun in one frame, which is only
    # possible because the elongation is small -- which is itself the point
    dict(label="WHY IT NEVER LEAVES THE SUN", mode="earthview", seconds=9.0,
         jd=(T_INF - 20.0, T_INF - 13.0), ease=linear, fov=(62.0, 56.0),
         tilt=35.0, standoff=2.2, both_in_frame=True, fade_out=2.2),
]


def _pair(v, u, how=lerp):
    return how(v[0], v[1], u) if isinstance(v, tuple) else v


def storyboard():
    """Yield one dict of resolved per-frame parameters per frame."""
    total = 0
    plan = []
    for shot in SHOTS:
        n = max(2, int(round(shot["seconds"] * args.fps * args.scale)))
        plan.append((shot, n))
        total += n
    done = 0
    for shot, n in plan:
        for k in range(n):
            u = k / (n - 1)
            e = shot["ease"](u)
            jd0, jd1 = shot["jd"]
            fade = 1.0
            # measured against this render's own frame count: at --scale 0.02 a
            # shot is a handful of frames and a fade quoted in seconds would
            # never finish, leaving the whole shot dark
            if shot.get("fade_in"):
                span = max(1.0, shot["fade_in"] * args.fps * args.scale)
                fade *= min(1.0, k / span)
            if shot.get("fade_out"):
                span = max(1.0, shot["fade_out"] * args.fps * args.scale)
                fade *= min(1.0, (n - 1 - k) / span)
            done += 1
            yield dict(
                shot=shot, label=shot["label"], mode=shot["mode"], u=u,
                jd=lerp(jd0, jd1, e), fade=fade,
                dist=_pair(shot.get("dist", 10.0), ease_in_out(u), geom),
                elev=_pair(shot.get("elev", 60.0), ease_in_out(u)),
                azim=_pair(shot.get("azim", -90.0), ease_in_out(u)),
                fov=_pair(shot.get("fov", 40.0), ease_in_out(u), geom),
                pscale=_pair(shot.get("pscale", 900.0), ease_in_out(u), geom),
                sscale=_pair(shot.get("sscale", 22.0), ease_in_out(u), geom),
                progress=done, total=total)


FRAMES = list(storyboard())
if args.storyboard:
    # one representative frame per shot, chosen where the shot has settled: the
    # midpoint for most, but past the cut for a split so the telescopic half of
    # the shot gets looked at too
    picks, seen = [], {}
    for fr in FRAMES:
        seen.setdefault(id(fr["shot"]), []).append(fr)
    for group in seen.values():
        picks.append(group[len(group) // 2])
        cut = group[0]["shot"].get("cut_at")
        if cut is not None:
            after = [f for f in group if f["u"] >= cut]
            if after:
                picks.append(after[len(after) // 2])
    FRAMES = picks
N = len(FRAMES)
print(f"{N} frames -> {N / args.fps:.1f} s at {args.fps} fps, {W}x{H}")


# ---------------------------------------------------------------------------
# cameras
# ---------------------------------------------------------------------------
def orbital_camera(jd, dist, elev_deg, azim_deg):
    """Camera on a sphere about the Sun, its longitude tied to the Earth's.

    Pinning the camera longitude to the Earth's means the Earth holds still in
    frame across a shot that spans months, so the thing that visibly moves is
    Venus and the angle between them -- which is the only thing these shots are
    about. With `azim = -90` the Earth sits at the right-hand edge.
    """
    e = heliocentric_of_date("earth", Instant(jd).jde)
    lon = math.atan2(e[1], e[0]) + azim_deg * DEG
    el = min(elev_deg, 89.9) * DEG
    return np.array([dist * math.cos(el) * math.cos(lon),
                     dist * math.cos(el) * math.sin(lon),
                     dist * math.sin(el)])


_orbit_xyz = None


def tangent_points(earth):
    """The two points on Venus' orbit that its sight lines from Earth graze.

    One per side, picked by the sign of the cross product so the pair always
    straddles the Sun rather than both landing on the same limb of the orbit.
    """
    global _orbit_xyz
    if _orbit_xyz is None:
        from vmsim.planets import orbit_ellipse_j2000
        _orbit_xyz = np.asarray(orbit_ellipse_j2000("venus", Instant(T_ELO).t, 720))
    d = _orbit_xyz - earth
    n = np.linalg.norm(d, axis=1)
    to_sun = -earth / np.linalg.norm(earth)
    cosang = (d @ to_sun) / np.maximum(n, 1e-12)
    side = d[:, 0] * to_sun[1] - d[:, 1] * to_sun[0]
    out = []
    for sel in (side >= 0.0, side < 0.0):
        if sel.any():
            idx = np.where(sel)[0][np.argmin(cosang[sel])]
            out.append(_orbit_xyz[idx])
    return out


def wedge_label(earth):
    """Half-angle of that cone in degrees, and a point to hang the label on."""
    tps = tangent_points(earth)
    if not tps:
        return 0.0, earth
    to_sun = -earth / np.linalg.norm(earth)
    angs = []
    for tp in tps:
        d = tp - earth
        angs.append(math.degrees(math.acos(
            float(np.dot(d, to_sun)) / np.linalg.norm(d))))
    reach = np.linalg.norm(tps[0] - earth)
    return max(angs), earth + to_sun * (0.30 * reach)


def venus_apparent(jd):
    """Where Venus appeared from the Earth at `jd`: light-time corrected."""
    _geo, _lt, helio = geocentric_of_date("venus", Instant(jd).jde)
    return np.array(helio)


# ---------------------------------------------------------------------------
# renderers -- Taichi field shapes are fixed at construction, so a panel of a
# different size has to be a different instance
# ---------------------------------------------------------------------------
big = SpaceRenderer(W, H)
half = SpaceRenderer(W // 2, H)
third = SpaceRenderer(W // 3, H)
for r in (big, half, third):
    r.samples[None] = args.samples

STARS = precessed_equatorial(Instant(T_ELO).t)
big.load_stars(STARS)
half.load_stars(STARS)

UI = max(1, W // 640)
BAR = 34 * UI

def in_frame(p, margin=0):
    """True if a projected point is actually on screen.

    Labels are clamped into the frame, which is right for one pinned to a body
    near the edge and wrong for one pinned to a body that is off screen
    entirely -- that produces a caption sitting against the border pointing at
    nothing.
    """
    return (p is not None and margin <= p[0] < W - margin
            and margin <= p[1] < H - margin)


def label(img, x, y, text, colour, scale=None, anchor="left", left=0, right=None):
    """Draw a label, clamped inside the frame (or inside one split panel).

    Annotations are pinned to projected body positions, which near the edge of
    frame would otherwise push the text off it -- and a caption that is half
    missing is worse than one that has been nudged. `left`/`right` bound it to a
    panel instead, so a caption in the geometry half cannot spill across the
    divider into the telescope half.
    """
    scale = UI if scale is None else scale
    right = img.shape[1] if right is None else right
    tw = text_width(text, scale)
    if anchor == "centre":
        x -= tw // 2
    x = int(min(max(x, left + 8 * UI), right - tw - 8 * UI))
    y = int(min(max(y, BAR + 3 * UI), img.shape[0] - 10 * UI))
    draw_text(img, x, y, text, colour, scale)


WARM = (1.00, 0.84, 0.52)
COOL = (0.52, 0.72, 1.00)
DIM = (0.50, 0.56, 0.68)
PALE = (0.80, 0.86, 0.98)


# ---------------------------------------------------------------------------
# annotation
# ---------------------------------------------------------------------------
def annotate(img, jd, cam, fov, w, h, *, wedge=False, x0=0):
    """Draw the Sun-Venus-Earth triangle over an already-rendered orbital view.

    Everything is projected through `render.spacecam`, which mirrors the kernel
    camera exactly, so the lines land on the bodies the tracer drew rather than
    near them.
    """
    f, r, u = basis(cam, (0.0, 0.0, 0.0))
    inst = Instant(jd)
    e = np.array(heliocentric_of_date("earth", inst.jde))
    v = np.array(heliocentric_of_date("venus", inst.jde))
    g = geometry(jd)

    def P(p):
        q = project(p, cam, f, r, u, fov, w, h)
        return None if q is None else (q[0] + x0, q[1])

    ps, pv, pe = P((0.0, 0.0, 0.0)), P(v), P(e)

    # The wedge Venus can never be seen outside of: the two sight lines from the
    # Earth that graze its orbit. Their half-angle IS the greatest elongation,
    # so this one construction answers "why does Venus never leave the Sun".
    #
    # Taken from the sampled ellipse rather than from asin(r/d) on a circle.
    # Venus' orbit is nearly circular but the Earth's is not, and the tangent
    # points are where the real greatest elongation is actually reached -- which
    # is the number the film solved for and quotes elsewhere. Deriving it two
    # different ways and having them disagree on screen would be worse than not
    # drawing it.
    if wedge and pe is not None:
        for tp in tangent_points(e):
            draw_line(img, pe, P(tp), (0.30, 0.26, 0.16),
                      radius=1.0, alpha=0.85, dash=7, gap=9)
        half_ang, anchor = wedge_label(e)
        lab = P(anchor)
        if lab is not None:
            label(img, int(lab[0]), int(lab[1]) + 16 * UI,
                  f"VENUS NEVER LEAVES THIS {half_ang:.0f} DEG CONE",
                  (0.58, 0.50, 0.34), anchor="centre", left=x0, right=x0 + w)

    draw_line(img, ps, pe, (0.20, 0.26, 0.36), radius=1.0, alpha=0.9, dash=5, gap=6)
    draw_line(img, ps, pv, (0.34, 0.28, 0.18), radius=1.0, alpha=0.9, dash=5, gap=6)
    draw_line(img, pe, pv, (0.46, 0.52, 0.64), radius=1.2, alpha=1.0)

    # Elongation, measured at the Earth; phase angle, measured at Venus. Both
    # are suppressed below a few degrees: near conjunction the arc collapses
    # onto the bodies and the reading is meaningless anyway.
    if pe is not None and ps is not None and pv is not None:
        rad, rad2 = 0.055 * w, 0.034 * w
        show_e = g.elongation_deg > 4.0
        show_v = g.phase_angle_deg > 4.0
        lab = arc_label_point(pe, ps, pv, rad * 1.24) if show_e else None
        lab2 = arc_label_point(pv, ps, pe, rad2 * 1.6) if show_v else None

        # Approaching inferior conjunction, Venus and the Earth close on each
        # other on screen and the two readouts land on the same pixels. Nudge
        # them apart vertically rather than letting them overprint into
        # something unreadable.
        if lab and lab2 and abs(lab[0] - lab2[0]) < 34 * UI \
                and abs(lab[1] - lab2[1]) < 13 * UI:
            lab = (lab[0], lab[1] - 9 * UI)
            lab2 = (lab2[0], lab2[1] + 9 * UI)

        if show_e:
            draw_arc(img, pe, ps, pv, rad, COOL, alpha=0.9)
            if lab:
                label(img, int(lab[0]), int(lab[1]) - 4 * UI,
                      f"{g.elongation_deg:.0f} DEG", COOL, anchor="centre",
                      left=x0, right=x0 + w)
        if show_v:
            draw_arc(img, pv, ps, pe, rad2, WARM, alpha=0.75, dash=3, gap=4)
            if lab2:
                label(img, int(lab2[0]), int(lab2[1]) - 4 * UI,
                      f"{g.phase_angle_deg:.0f} DEG", WARM, anchor="centre",
                      left=x0, right=x0 + w)

    if ps is not None:
        label(img, int(ps[0]) + 11 * UI, int(ps[1]) - 3 * UI, "SUN",
              (0.95, 0.85, 0.60), right=x0 + w)
        # the planet names sit radially outward from the Sun, which puts them on
        # the opposite side from the angle readouts -- those always hug the
        # sunward side, and the two used to land on top of each other
        for name, p, col in (("VENUS", pv, (0.95, 0.92, 0.82)), ("EARTH", pe, COOL)):
            if p is None:
                continue
            dx, dy = p[0] - ps[0], p[1] - ps[1]
            n = max(1e-6, math.hypot(dx, dy))
            label(img, int(p[0] + dx / n * 16 * UI) - 8 * UI,
                  int(p[1] + dy / n * 16 * UI) - 4 * UI, name, col, right=x0 + w)


# ---------------------------------------------------------------------------
# per-mode rendering
# ---------------------------------------------------------------------------
def render_orbital(fr, w=None, h=None, renderer=None, x0=0, img=None):
    r = renderer or big
    w = w or r.width
    h = h or r.height
    inst = Instant(fr["jd"])
    cam = orbital_camera(fr["jd"], fr["dist"], fr["elev"], fr["azim"])
    r.set_camera(cam, (0.0, 0.0, 0.0), fr["fov"])
    names = ("mercury", "venus", "earth", "mars") if fr["shot"].get("context") \
        else ("venus", "earth")
    heliocentric_scene(r, inst, planet_scale=fr["pscale"], sun_scale=fr["sscale"],
                       bodies=names)
    r.load_paths(cached_paths(inst, names))
    r.fade[None] = fr["fade"]
    r.exposure[None] = 1.0
    out = r.image()
    if img is None:
        img = out
    else:
        img[:, x0:x0 + w] = out
    if fr["shot"].get("annotate"):
        annotate(img, fr["jd"], cam, fr["fov"], w, h,
                 wedge=fr["shot"].get("wedge", False), x0=x0)
    return img


def render_scope(jd, r, fade=1.0):
    """Venus as it appeared, at the film's one fixed angular scale."""
    inst = Instant(jd)
    earth = np.array(heliocentric_of_date("earth", inst.jde))
    venus = venus_apparent(jd)
    r.set_camera(earth, venus, scope_fov(r.width))
    r.clear_bodies()
    # the orbit trails belong to the orbital shots; left loaded, they are still
    # depth-tested and splatted, and can drop stray points into a telescopic
    # field where an orbit line has no business being
    r.load_paths([])
    r.corona[None] = 0.0
    r.add_body(venus, radius_au("venus"), VENUS, tint=(1.00, 0.978, 0.912),
               axis=(0.0, 0.0, -1.0), spin=spin_phase("venus", jd))
    r.fade[None] = fade
    r.exposure[None] = 1.9
    img = r.image()
    r.corona[None] = 1.0
    return img


def render_earthview(fr):
    """Over the shoulder: the Earth's limb below, Venus and the Sun above it.

    The camera sits a couple of Earth radii off the surface. The view axis is
    the bisector of the directions to Venus and to the Sun, and the up vector is
    the normal of the plane containing them, which lays those two objects out
    along a horizontal line with the elongation between them read directly off
    the screen. The Earth is then tilted below that axis far enough for its limb
    to cut across the lower frame.

    Venus subtends about a minute of arc from here against a field tens of
    degrees wide, so its disc is a hundredth of a pixel. It goes in as a point
    source carrying its real magnitude -- which is exactly what it looks like to
    an eye, and the reason it is the brightest thing in that sky after the Sun.
    """
    from render.space import EARTH as _EARTH

    jd = fr["jd"]
    inst = Instant(jd)
    shot = fr["shot"]
    earth = np.array(heliocentric_of_date("earth", inst.jde))
    venus = venus_apparent(jd)
    g = geometry(jd)
    r_e = radius_au("earth")

    to_v = venus - earth
    to_v /= np.linalg.norm(to_v)
    to_s = -earth / np.linalg.norm(earth)
    up = np.cross(to_v, to_s)
    up /= np.linalg.norm(up)
    axis = (to_v + to_s) if shot.get("both_in_frame") else to_v.copy()
    axis /= np.linalg.norm(axis)

    tilt = shot.get("tilt", 33.0) * DEG
    cam = earth - (axis * math.cos(tilt) - up * math.sin(tilt)) * shot.get("standoff", 2.2) * r_e
    aim = cam + axis

    big.set_camera(cam, aim, fr["fov"], up=up)
    big.clear_bodies()
    big.add_body((0.0, 0.0, 0.0), radius_au("sun"), SUN)
    big.add_body(earth, r_e, _EARTH, axis=np.array([0.0, -0.39774, 0.91748]),
                 spin=spin_phase("earth", jd))
    big.add_glare(venus, point_flux(g.magnitude))
    big.load_paths([])
    big.fade[None] = fr["fade"]
    big.exposure[None] = 1.0
    big.spike[None] = 0.022          # the instrument PSF, so the Sun reads as a source
    img = big.image()
    big.spike[None] = 0.0

    f, rr, uu = basis(cam, aim, up=up)
    pv = project(venus, cam, f, rr, uu, fr["fov"], W, H)
    ps = project((0.0, 0.0, 0.0), cam, f, rr, uu, fr["fov"], W, H)
    if pv is not None:
        label(img, int(pv[0]) + 16 * UI, int(pv[1]) - 4 * UI, "VENUS", PALE)
        label(img, int(pv[0]) + 16 * UI, int(pv[1]) + 8 * UI,
              f"MAG {g.magnitude:+.1f}   {g.illuminated_fraction * 100:.0f}% LIT", DIM)
    if in_frame(ps, 40):
        label(img, int(ps[0]) + 16 * UI, int(ps[1]) - 4 * UI, "SUN",
              (0.95, 0.85, 0.60))
    if in_frame(pv) and in_frame(ps) and shot.get("both_in_frame"):
        draw_line(img, ps, pv, (0.34, 0.34, 0.30), radius=1.1, alpha=0.8,
                  dash=6, gap=8)
        label(img, int(0.5 * (ps[0] + pv[0])), int(0.5 * (ps[1] + pv[1])) - 16 * UI,
              f"{g.elongation_deg:.0f} DEG APART - AND NEVER MUCH MORE",
              (0.62, 0.66, 0.78), anchor="centre")
    elif pv is not None:
        label(img, int(pv[0]) - 30 * UI, int(pv[1]) + 26 * UI,
              f"{g.elongation_deg:.0f} DEG FROM THE SUN", (0.62, 0.66, 0.78))
    return img


def render_triptych(fr):
    """The three beats abreast, at the same arcseconds per pixel.

    This is the shot that has to be unimpeachable, so the panels are rendered by
    the same code path as the full-frame telescopic views with only the width
    changed -- the angular scale per pixel is identical, and the discs can be
    measured against each other with a ruler.
    """
    img = np.zeros((H, W, 3), np.float32)
    pw = W // 3
    for i, (name, jd) in enumerate((("SUPERIOR CONJUNCTION", T_SUP),
                                    (f"GREATEST {ELO_SIDE.upper()} ELONGATION", T_ELO),
                                    ("NEAR INFERIOR CONJUNCTION", T_INF - 6.0))):
        panel = render_scope(jd, third, fade=fr["fade"])
        img[:, i * pw:(i + 1) * pw] = panel
        g = geometry(jd)
        x = i * pw + 12 * UI
        y = H - 62 * UI
        draw_text(img, x, y, name, PALE, UI)
        draw_text(img, x, y + 13 * UI,
                  f"{g.angular_diameter_arcsec:.1f}\"  {g.illuminated_fraction * 100:.0f}% LIT",
                  WARM, UI)
        draw_text(img, x, y + 26 * UI, f"DISTANCE {g.delta_au:.2f} AU", DIM, UI)
        if i:
            img[:, i * pw:i * pw + 1] = (0.16, 0.18, 0.24)
    return img


def render_split(fr):
    """Geometry beside consequence, then a hard cut to the consequence alone."""
    if fr["u"] >= fr["shot"].get("cut_at", 1.1):
        return render_scope(fr["jd"], big, fade=fr["fade"]), True
    img = np.zeros((H, W, 3), np.float32)
    render_orbital(fr, w=W // 2, h=H, renderer=half, x0=0, img=img)
    img[:, W // 2:] = render_scope(fr["jd"], half, fade=fr["fade"])
    img[:, W // 2 - 1:W // 2 + 1] = (0.16, 0.18, 0.24)
    return img, False


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------
def hud(img, fr, full_scope):
    g = geometry(fr["jd"])
    img[:BAR] *= 0.18
    img[BAR:BAR + 1] = (0.30, 0.34, 0.42)
    # shrink rather than run under the readout on the right
    tscale = 2 * UI
    while tscale > UI and text_width(fr["label"], tscale) > W * 0.48:
        tscale -= 1
    draw_text(img, 10 * UI, 5 * UI, fr["label"], (1.0, 1.0, 1.0), tscale)
    where = {"orbital": "HELIOCENTRIC", "split": "GEOMETRY AND CONSEQUENCE",
             "triptych": "ONE PLANET, THREE DATES", "earthview": "BESIDE THE EARTH"}
    sub = "TELESCOPIC" if full_scope else where.get(fr["mode"], "")
    draw_text(img, 10 * UI, 21 * UI,
              f"{jd_to_utc_string(fr['jd'], seconds=False)}   {sub}", PALE, UI)

    # two lines: at UI=3 the whole readout on one line is wider than the frame
    # and runs back over the title
    top = (f"ELONG {g.elongation_deg:5.1f} DEG    PHASE {g.phase_angle_deg:5.1f} DEG"
           f"    ILLUM {g.illuminated_fraction * 100:4.1f}%")
    # Hilton's (2005) magnitude law is fitted over phase angle 0-163.7 deg and
    # `venus_magnitude` clamps there, so within about six days of inferior
    # conjunction the quoted magnitude is an extrapolation off the end of the
    # fit. The film reaches that regime, so it says so rather than presenting an
    # extrapolated number as a measurement.
    extrapolated = g.phase_angle_deg > 163.6
    bot = (f"DIAM {g.angular_diameter_arcsec:4.1f}\"    "
           f"MAG {g.magnitude:+.2f}{'*' if extrapolated else ' '}"
           f"   DISTANCE {g.delta_au:.3f} AU")
    draw_text(img, W - text_width(top, UI) - 10 * UI, 6 * UI, top, PALE, UI)
    draw_text(img, W - text_width(bot, UI) - 10 * UI, 21 * UI, bot, WARM, UI)

    # the standing note on what is and is not to scale
    mode = fr["mode"]
    if not full_scope and mode in ("orbital", "split") and fr["shot"].get("pscale"):
        note = (f"BODIES X{fr['pscale']:.0f}  SUN X{fr['sscale']:.0f} - "
                f"ORBITS AND LIGHT TO SCALE")
        draw_text(img, 10 * UI, H - 20 * UI, note, (0.42, 0.46, 0.56), UI)
    if full_scope or mode in ("split", "triptych"):
        note = f"FIXED {ARCSEC_PER_PX}\"/PIXEL - NO ZOOM"
        draw_text(img, W - text_width(note, UI) - 10 * UI, H - 20 * UI, note,
                  (0.42, 0.46, 0.56), UI)

    if extrapolated:
        # just under the readout it qualifies; the footer is already spoken for
        # by the triptych's per-panel captions
        msg = "* MAG EXTRAPOLATED PAST THE HILTON PHASE-ANGLE FIT"
        draw_text(img, W - text_width(msg, UI) - 10 * UI, BAR + 6 * UI, msg,
                  (0.52, 0.44, 0.34), UI)

    if MARK_OCC and abs(fr["jd"] - T_OCC) < 1.6:
        msg = "2026 SEPTEMBER 14 - THE MOON OCCULTS VENUS"
        draw_text(img, (W - text_width(msg, UI * 2)) // 2, BAR + 18 * UI, msg,
                  WARM, UI * 2)

    done = int(fr["progress"] / fr["total"] * W)
    img[-3 * UI:, :done] = (0.42, 0.55, 0.80)


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------
import imageio.v2 as imageio                                          # noqa: E402

if args.storyboard:
    os.makedirs(args.storyboard, exist_ok=True)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
if args.frames_dir:
    os.makedirs(args.frames_dir, exist_ok=True)
writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                            quality=8, macro_block_size=1,
                            ffmpeg_params=["-pix_fmt", "yuv420p"])

if args.resume and not args.frames_dir:
    raise SystemExit("--resume needs --frames-dir to resume from")


def frame_path(n):
    return os.path.join(args.frames_dir, f"f{n:05d}.png")


def render_frame(fr):
    """One finished RGB frame, HUD and all."""
    mode = fr["mode"]
    full_scope = False
    if mode == "orbital":
        img = render_orbital(fr)
    elif mode == "earthview":
        img = render_earthview(fr)
    elif mode == "triptych":
        img = render_triptych(fr)
    elif mode == "split":
        img, full_scope = render_split(fr)
    else:
        raise SystemExit(f"unknown shot mode {mode!r}")

    img = np.ascontiguousarray(img)
    hud(img, fr, full_scope)
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8), full_scope


def progress(n, fr):
    if n % 40 == 0 or n == N - 1:
        print(f"  {n + 1:5d}/{N}  {fr['label']:<32s} "
              f"{jd_to_utc_string(fr['jd'], seconds=False)}", flush=True)


if args.resume:
    # Two passes, so a run that dies partway through -- this machine's memory
    # reaper has taken long jobs before -- loses only the frame it was on.
    # Re-running picks up from the first PNG that isn't there yet.
    done = 0
    for n, fr in enumerate(FRAMES):
        # `exists` is not enough: a frame the previous run was killed partway
        # through is still on disk, and skipping it would silently ship a hole
        if png_complete(frame_path(n)):
            done += 1
            continue
        rgb, _ = render_frame(fr)
        # level 1: these are a scratch cache re-read once by the encoder, so
        # zlib effort here is pure cost
        write_png(frame_path(n), rgb, level=1)
        progress(n, fr)
    if done:
        print(f"  reused {done} frames already on disk")
    print("encoding...", flush=True)
    for n in range(N):
        writer.append_data(read_png(frame_path(n)))
else:
    for n, fr in enumerate(FRAMES):
        rgb, full_scope = render_frame(fr)
        if args.storyboard:
            name = fr["label"].lower().replace(" ", "_").replace(",", "")
            write_png(os.path.join(args.storyboard,
                                   f"{n:02d}_{name}{'_cut' if full_scope else ''}.png"), rgb)
            continue
        writer.append_data(rgb)
        if args.frames_dir:
            write_png(frame_path(n), rgb)
        progress(n, fr)

writer.close()
if args.storyboard:
    os.remove(args.out) if os.path.exists(args.out) else None
    raise SystemExit(f"wrote {N} storyboard frames to {args.storyboard}")
size = os.path.getsize(args.out) / 1e6
print(f"wrote {args.out}  ({N} frames, {N / args.fps:.1f} s, {size:.1f} MB)")
