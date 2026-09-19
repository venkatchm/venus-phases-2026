"""A vertical cut for Instagram Reels: 1080x1920, about a minute.

The 16:9 film does not translate to a phone held upright, so this is a separate
edit rather than a crop of it. Landscape frames are fitted to the full width and
placed high in the canvas, leaving room above for a line that says what you are
looking at and room below for the number that proves it -- which is how the
science survives being watched with the sound off on a small screen.

Composited from frames already rendered, so it costs a minute rather than an
hour and the pictures are identical to the long film.

    python scripts/build_reel.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from render.png import png_complete, read_png
from render.text import GLYPH_H, draw_text, text_width

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--phases-frames", default="out/phaseframes")
ap.add_argument("--occ-frames", default="out/occframes")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--out", default="out/venus_reel.mp4")
args = ap.parse_args()

W, H = 1080, 1920
VID_W, VID_H = 1080, 608          # a 16:9 frame fitted to full width, no upscale
VID_Y = 470                       # placed high: captions read better underneath
UI = 3

WHITE, WARM, COOL, DIM = (1, 1, 1), (1.00, 0.84, 0.52), (0.60, 0.76, 1.0), (0.55, 0.60, 0.72)


def fit(frame):
    """1920x1080 -> 1080x608 by taking every other row and column pair.

    A plain 16:9 fit, so nothing is cropped away and nothing is upscaled. The
    source is smooth gradients and discs, which survive this cleanly.
    """
    a = frame.astype(np.float32)
    a = a.reshape(1080 // 2, 2, 1920 // 2, 2, 3).mean(axis=(1, 3))   # 960x540
    # 960x540 -> 1080x608 by nearest sampling; both axes scale by ~1.125
    ys = (np.arange(VID_H) * (540 / VID_H)).astype(int).clip(0, 539)
    xs = (np.arange(VID_W) * (960 / VID_W)).astype(int).clip(0, 959)
    return a[ys][:, xs]


def centre(canvas, y, text, scale, colour, margin=40):
    """Centred text that shrinks rather than running off the edge.

    A phone is 1080 px wide and this font is 6 px per character per scale step,
    so a headline of any length overflows at the scale a headline wants to be.
    Shrinking to fit is the only option that keeps the line readable.
    """
    if not text:
        return scale
    while scale > 1 and text_width(text, scale) > W - 2 * margin:
        scale -= 1
    draw_text(canvas, (W - text_width(text, scale)) // 2, y, text, colour, scale)
    return scale


def compose(frame, headline, sub, stat):
    c = np.zeros((H, W, 3), np.float32)
    c[VID_Y:VID_Y + VID_H] = fit(frame) / 255.0
    centre(c, 250, headline, 3 * UI, WHITE)
    for i, line in enumerate(sub):
        centre(c, VID_Y + VID_H + 90 + i * (GLYPH_H * 2 * UI + 22), line, 2 * UI, COOL)
    if stat:
        centre(c, H - 330, stat, 2 * UI, WARM)
    centre(c, H - 150, "github.com/venkatchm/venus-phases-2026", UI, DIM)
    return (np.clip(c, 0, 1) * 255).astype(np.uint8)


def card(lines, seconds, fade=0.5):
    n = int(seconds * args.fps)
    base = np.zeros((H, W, 3), np.float32)
    gap = 11 * UI
    fitted = []
    for text, sc, col in lines:
        while sc > 1 and text and text_width(text, sc) > W - 80:
            sc -= 1
        fitted.append((text, sc, col))
    height = sum(GLYPH_H * sc for _, sc, _ in fitted) + gap * (len(fitted) - 1)
    y = (H - height) // 2
    for text, sc, col in fitted:
        centre(base, y, text, sc, col)
        y += GLYPH_H * sc + gap
    nf = max(1, int(fade * args.fps))
    for i in range(n):
        k = min(1.0, min(i, n - 1 - i) / nf) if n > 2 * nf else 1.0
        yield (np.clip(base * k, 0, 1) * 255).astype(np.uint8)


def clip(folder, lo, hi, headline, sub, stat):
    names = sorted(f for f in os.listdir(folder) if f.endswith(".png"))
    for name in names[lo:hi]:
        p = os.path.join(folder, name)
        if png_complete(p):
            yield compose(read_png(p), headline, sub, stat)


P, O = args.phases_frames, args.occ_frames

SEGMENTS = [
    # Open on the payoff. A phone viewer decides in two seconds.
    ("clip", O, 1000, 1120, "THE MOON MOVED",
     ["VENUS COMES BACK OUT", "FROM BEHIND THE MOON"], "2026 SEPTEMBER 14"),
    ("card", [("I MISSED IT.", 4 * UI, WHITE), ("", 1, WHITE),
              ("SO I SIMULATED IT.", 4 * UI, WARM)], 2.6),
    ("clip", P, 1900, 2200, "WHY VENUS HAS PHASES",
     ["VENUS ORBITS INSIDE US,", "SO WE SEE IT LIT FROM THE SIDE"],
     "THE SUN IS THE ONLY LIGHT"),
    ("clip", P, 3960, 4230, "THE CRESCENT IS BIGGEST",
     ["SAME SCALE IN ALL THREE.", "IT IS NOT GROWING - IT IS CLOSER"],
     "1.71 AU  ->  0.28 AU"),
    ("clip", O, 500, 720, "DISAPPEARANCE",
     ["VENUS GOES BEHIND THE", "MOON'S DARK LIMB - IN DAYLIGHT"], "12:00:30 UT"),
    ("clip", O, 940, 1140, "REAPPEARANCE",
     ["75 MINUTES LATER", "IT COMES BACK"], "13:15:33 UT"),
    ("card", [("NOTHING HERE IS DRAWN", 3 * UI, WHITE),
              ("", 1, WHITE),
              ("EVERY POSITION SOLVED", 2 * UI, COOL),
              ("FROM VSOP87D AT RUN TIME", 2 * UI, COOL),
              ("", 1, WHITE),
              ("74 CHECKS AGAINST", 2 * UI, DIM),
              ("PUBLISHED VALUES", 2 * UI, DIM),
              ("", 1, WHITE),
              ("PYTHON + TAICHI", 2 * UI, WARM)], 4.5),
]

import imageio.v2 as imageio                                          # noqa: E402

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                            quality=8, macro_block_size=1,
                            ffmpeg_params=["-pix_fmt", "yuv420p"])
n = 0
for seg in SEGMENTS:
    if seg[0] == "card":
        frames = card(seg[1], seg[2])
        what = "card"
    else:
        _, folder, lo, hi, head, sub, stat = seg
        frames = clip(folder, lo, hi, head, sub, stat)
        what = head
    c = 0
    for f in frames:
        writer.append_data(f)
        n += 1
        c += 1
    print(f"  {what:28s} {c:4d} frames", flush=True)
writer.close()
print(f"\nwrote {args.out}")
print(f"  {W}x{H} vertical, {n} frames, {n / args.fps:.0f} s, "
      f"{os.path.getsize(args.out) / 1e6:.1f} MB")
