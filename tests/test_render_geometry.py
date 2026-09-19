"""Invariants of the heliocentric renderer, checked numerically.

The phases film makes three claims that a viewer has to take on trust unless
something checks them: that the lit side of every body faces the Sun, that the
telescopic views share one angular scale so a bigger disc really is a bigger
angle, and that the annotations drawn over a frame land on the bodies the
tracer actually drew. Each is asserted here against rendered pixels rather than
against the code that produced them.

Run with:  python tests/test_render_geometry.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import taichi as ti

ti.init(arch=ti.cpu, offline_cache=True, log_level=ti.WARN)

from render.bridge import radius_au                                   # noqa: E402
from render.space import EARTH, SUN, VENUS, SpaceRenderer             # noqa: E402
from render.spacecam import basis, project                            # noqa: E402
from vmsim.planets import geocentric_of_date, heliocentric_of_date    # noqa: E402
from vmsim.scene import venus_geometry                                # noqa: E402
from vmsim.timescale import Instant                                   # noqa: E402

FAILURES = []
ARCSEC_PER_PX = 0.13


def check(name, got, expected, tol, unit=""):
    ok = abs(got - expected) <= tol
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name:<46s} {got:12.4f} vs {expected:12.4f} "
          f"(tol {tol:g}{unit})")
    if not ok:
        FAILURES.append(name)


def assert_true(name, ok):
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}")
    if not ok:
        FAILURES.append(name)


def centroid(img, col, row, half):
    """Brightness-weighted centroid of a window, in image coordinates."""
    c0, c1 = int(col) - half, int(col) + half
    r0, r1 = int(row) - half, int(row) + half
    win = img[r0:r1, c0:c1].sum(axis=2)
    ys, xs = np.mgrid[r0:r1, c0:c1]
    total = win.sum()
    return (xs * win).sum() / total, (ys * win).sum() / total


# ---------------------------------------------------------------------------
print("Camera parity: the overlay must project the way the kernel traced")
# `render.spacecam.basis` is a hand-copy of `SpaceRenderer.set_camera`. If they
# ever drift, every annotation drifts with them -- silently, and only by a few
# pixels, which is exactly the kind of error that survives review.
_r = SpaceRenderer(64, 64)
worst = 0.0
for elev in (0.0, 30.0, 60.0, 85.0, 89.9):
    for lon in (0.0, 1.0, 2.5, 4.0):
        e = math.radians(elev)
        pos = 8.0 * np.array([math.cos(e) * math.cos(lon),
                              math.cos(e) * math.sin(lon), math.sin(e)])
        _r.set_camera(pos, (0.0, 0.0, 0.0), 20.0)
        f, rr, uu = basis(pos, (0.0, 0.0, 0.0))
        for a, b in ((f, _r.cam_f[None].to_numpy()), (rr, _r.cam_r[None].to_numpy()),
                     (uu, _r.cam_u[None].to_numpy())):
            worst = max(worst, float(np.abs(np.asarray(a) - b).max()))
check("worst basis-vector disagreement", worst, 0.0, 1e-6)

# ---------------------------------------------------------------------------
print("\nProjection lands on what was rendered")
W, H, FOV = 960, 540, 26.0
r = SpaceRenderer(W, H)
r.samples[None] = 2
r.corona[None] = 0.0
inst = Instant.from_utc(2026, 8, 15)
earth = np.array(heliocentric_of_date("earth", inst.jde))
venus = np.array(heliocentric_of_date("venus", inst.jde))
cam = np.array([0.0, 0.0, 14.0])
r.set_camera(cam, (0.0, 0.0, 0.0), FOV)
r.clear_bodies()
r.add_body((0.0, 0.0, 0.0), radius_au("sun"), SUN, scale=25.0)
r.add_body(venus, radius_au("venus"), VENUS, scale=1500.0, tint=(1.0, 0.978, 0.912))
r.add_body(earth, radius_au("earth"), EARTH, scale=1500.0, axis=(0.4, 0.0, 0.92))
img = r.image()
f, rr, uu = basis(cam, (0.0, 0.0, 0.0))

# The Sun is uniformly emissive, so its brightness centroid is its geometric
# centre and this isolates projection error from phase.
sx, sy = project((0.0, 0.0, 0.0), cam, f, rr, uu, FOV, W, H)
cx, cy = centroid(img, sx, sy, 60)
check("Sun centroid vs projection (px)", math.hypot(cx - sx, cy - sy), 0.0, 1.5, " px")

# ---------------------------------------------------------------------------
print("\nLighting invariant: the lit side faces the Sun")
# A partly lit sphere has its brightness centroid pulled off the geometric
# centre, towards the Sun and nowhere else. This is the rendered-pixel version
# of "the illuminated side always faces the Sun".
for name, pos in (("Venus", venus), ("Earth", earth)):
    px, py = project(pos, cam, f, rr, uu, FOV, W, H)
    bx, by = centroid(img, px, py, 34)
    off = np.array([bx - px, by - py])
    sun = np.array([sx - px, sy - py])
    cos = float(off @ sun / (np.linalg.norm(off) * np.linalg.norm(sun)))
    check(f"{name}: cos(offset, sunward)", cos, 1.0, 0.06)
    assert_true(f"{name}: offset is measurable, not noise",
                float(np.linalg.norm(off)) > 0.8)

# ---------------------------------------------------------------------------
print("\nOne fixed angular scale across every telescopic panel")
# The film cuts between a full frame, a half-width split panel and a third-width
# triptych panel, and claims the disc grows only because Venus came closer. That
# is only true if all three are built at the same arcseconds per pixel.
measured = {}
for width in (1920, 960, 640):
    rr_ = SpaceRenderer(width, 1080)
    rr_.samples[None] = 2
    rr_.corona[None] = 0.0
    for label, (y, m, d) in (("superior", (2026, 1, 6)),
                             ("elongation", (2026, 8, 15)),
                             ("near inferior", (2026, 10, 18))):
        when = Instant.from_utc(y, m, d)
        e = np.array(heliocentric_of_date("earth", when.jde))
        _g, _lt, v = geocentric_of_date("venus", when.jde)
        v = np.array(v)
        rr_.set_camera(e, v, width * ARCSEC_PER_PX / 3600.0)
        rr_.clear_bodies()
        rr_.add_body(v, radius_au("venus"), VENUS, tint=(1.0, 0.978, 0.912))
        rr_.exposure[None] = 1.9
        im = rr_.image()
        ys, _xs = np.where(im.sum(axis=2) > 0.04)
        # the cusps of a crescent always reach the poles, so the lit region's
        # vertical extent is the whole disc at any phase
        measured.setdefault(label, []).append(ys.max() - ys.min() + 1)

for label, (y, m, d) in (("superior", (2026, 1, 6)), ("elongation", (2026, 8, 15)),
                         ("near inferior", (2026, 10, 18))):
    want = venus_geometry(Instant.from_utc(y, m, d)).angular_diameter_arcsec / ARCSEC_PER_PX
    got = measured[label]
    check(f"{label}: rendered diameter (px)", float(got[0]), want, 3.0, " px")
    assert_true(f"{label}: identical at 1920/960/640 px wide", len(set(got)) == 1)

grew = measured["near inferior"][0] / measured["superior"][0]
check("disc growth across the apparition", grew, 6.2, 0.4, "x")

# ---------------------------------------------------------------------------
print("\nCrescent orientation in the telescopic shots")
# The film's whole second half rests on the claim that what you see through a
# telescope follows from where Venus is. The sharpest form of that: the bright
# limb must point at the Sun -- which is off frame in these shots, so nothing in
# the rendered image could have been fitted to it. The sunward direction is
# reconstructed from the camera basis and compared with the measured offset of
# the lit centroid from the disc centre.
_scope = SpaceRenderer(960, 1080)
_scope.samples[None] = 2
_scope.corona[None] = 0.0
for _y, _m, _d in ((2026, 6, 18), (2026, 8, 15), (2026, 9, 14), (2026, 10, 18)):
    _inst = Instant.from_utc(_y, _m, _d)
    _earth = np.array(heliocentric_of_date("earth", _inst.jde))
    _gg, _lt, _venus = geocentric_of_date("venus", _inst.jde)
    _venus = np.array(_venus)
    _fov = 960 * ARCSEC_PER_PX / 3600.0
    _scope.set_camera(_earth, _venus, _fov)
    _scope.clear_bodies()
    _scope.add_body(_venus, radius_au("venus"), VENUS, tint=(1.0, 0.978, 0.912))
    _scope.exposure[None] = 1.9
    _im = _scope.image()
    _lit = _im.sum(axis=2)
    _ys, _xs = np.mgrid[0:1080, 0:960]
    _tot = _lit.sum()
    _ox = (_xs * _lit).sum() / _tot - 480.0
    _oy = (_ys * _lit).sum() / _tot - 540.0
    _f, _r, _u = basis(_earth, _venus)
    _ts = -_earth / np.linalg.norm(_earth)
    _sx, _sy = float(_ts @ _r), -float(_ts @ _u)      # image rows count downwards
    _sn = math.hypot(_sx, _sy)
    _on = math.hypot(_ox, _oy)
    # Near full phase the centroid sits within a pixel of the disc centre and
    # its direction is quantisation noise, so the angle is only meaningful once
    # the crescent is pronounced enough to move it.
    if _on < 3.0:
        continue
    _cos = (_ox * _sx / _sn + _oy * _sy / _sn) / _on
    check(f"{_y}-{_m:02d}-{_d:02d}: bright limb points sunward",
          _cos, 1.0, 0.02)

print("\nRotation axes against IAU WGCCRE values")
# Each pole is checked against an independent fact: the angle between a body's
# spin axis and the ecliptic pole is set by its obliquity combined with its
# orbital inclination, and those are published separately from the pole itself.
import math as _m

from render.bridge import IAU_POLE_RADEC, POLE, ROTATION_DAYS   # noqa: E402

for _name, _expect in (("mercury", 7.0), ("venus", 1.2), ("earth", 23.44), ("mars", 25.4)):
    _tilt = _m.degrees(_m.acos(min(1.0, abs(POLE[_name][2]))))
    check(f"{_name}: axis tilt from ecliptic pole (deg)", _tilt, _expect, 0.4, " deg")
assert_true("Earth's pole points to ecliptic longitude 90, not 270",
            POLE["earth"][1] > 0.0)
# Venus turns the other way about its IAU north pole. Both halves have to be
# right: a south-pointing axis AND a negative period cancel into prograde.
assert_true("Venus' IAU north pole is north of the ecliptic",
            POLE["venus"][2] > 0.0)
assert_true("Venus rotates retrograde", ROTATION_DAYS["venus"] < 0.0)
assert_true("every pole is derived from a published RA/Dec",
            set(POLE) == set(IAU_POLE_RADEC))

print("\nPNG round-trip and frame-cache integrity")
# The resumable render caches frames as PNGs and reads them back to encode, so
# the writer and reader have to agree exactly -- and a frame the previous run
# was killed partway through must be recognised as unfinished rather than
# skipped as done. That second half is not hypothetical: a killed render left a
# zero-byte frame, `os.path.exists` said yes, and the encode walked into it.
import os as _os                                                       # noqa: E402
import tempfile as _tf                                                 # noqa: E402

from render.png import png_complete, read_png, write_png               # noqa: E402

with _tf.TemporaryDirectory() as _d:
    _a = (np.random.rand(71, 113, 3) * 255).astype(np.uint8)
    for _lvl in (1, 6):
        _p = _os.path.join(_d, f"rt{_lvl}.png")
        write_png(_p, _a, level=_lvl)
        assert_true(f"PNG round-trip is exact at level {_lvl}",
                    np.array_equal(read_png(_p), _a))
        assert_true(f"complete PNG reports complete (level {_lvl})", png_complete(_p))
    _trunc = _os.path.join(_d, "truncated.png")
    write_png(_trunc, _a)
    _whole = open(_trunc, "rb").read()
    with open(_trunc, "wb") as _fh:
        _fh.write(_whole[:len(_whole) // 2])
    assert_true("truncated PNG is detected as incomplete", not png_complete(_trunc))
    _empty = _os.path.join(_d, "empty.png")
    open(_empty, "wb").close()
    assert_true("zero-byte PNG is detected as incomplete", not png_complete(_empty))
    assert_true("missing file is detected as incomplete",
                not png_complete(_os.path.join(_d, "nope.png")))
    # the writer must not leave its temp file behind
    assert_true("write_png leaves no .part file",
                not any(f.endswith(".part") for f in _os.listdir(_d)))

print("\nDepth ordering: nothing shows through the Sun")
# Venus placed directly behind the Sun from the camera must be hidden by it,
# the same way the occultation renderer hides Venus behind the Moon.
r2 = SpaceRenderer(200, 200)
r2.samples[None] = 1
r2.corona[None] = 0.0
r2.set_camera((0.0, -3.0, 0.0), (0.0, 0.0, 0.0), 12.0)
r2.clear_bodies()
r2.add_body((0.0, 0.0, 0.0), radius_au("sun"), SUN, scale=20.0)
behind = np.array([0.0, 0.72, 0.0])          # straight through the Sun
r2.add_body(behind, radius_au("venus"), VENUS, scale=3000.0, tint=(1.0, 0.978, 0.912))
with_sun = r2.image()
r2.clear_bodies()
r2.add_body(behind, radius_au("venus"), VENUS, scale=3000.0, tint=(1.0, 0.978, 0.912))
without_sun = r2.image()
assert_true("Venus is visible when the Sun is removed", without_sun.max() > 0.2)
mid = with_sun[95:105, 95:105].min()
assert_true("Venus does not punch through the Sun", mid > 0.9)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {', '.join(FAILURES)}")
    raise SystemExit(1)
print("all checks passed")
