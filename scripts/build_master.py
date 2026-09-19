"""Assemble the one film: title cards, both parts, one file, one resolution.

Four separate videos is four things a viewer has to be told about before they
watch any of them. This stitches the two renders into a single narrative with
cards that say what is coming and why it matters, which is what an upload needs.

It composites already-rendered frames rather than re-rendering, so the pictures
are bit-identical to the standalone films and building the master costs minutes
instead of an hour.

    python scripts/render_venus_phases.py --arch gpu --resume \
           --frames-dir out/phaseframes --out out/venus_phases.mp4
    python scripts/render_animation.py --arch gpu --width 1920 --height 1080 \
           --frames-dir out/occframes --out out/september2026.mp4
    python scripts/build_master.py
"""
import argparse
import math
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
ap.add_argument("--out", default="out/venus_the_whole_story.mp4")
args = ap.parse_args()

W, H = 1920, 1080
UI = max(1, W // 640)


def card(lines, seconds, fade=0.6):
    """A title card: (text, scale, colour) lines, centred as a block."""
    n = int(seconds * args.fps)
    # the bitmap font is GLYPH_H rows tall, so a line occupies 7 * scale pixels;
    # measuring it as anything else leaves the block visibly off-centre
    block = [(t, sc, col) for t, sc, col in lines]
    gap = 9 * UI
    height = sum(GLYPH_H * sc for _, sc, _ in block) + gap * (len(block) - 1)
    base = np.zeros((H, W, 3), np.float32)
    y = (H - height) // 2
    for text, scale, colour in block:
        if text:
            draw_text(base, (W - text_width(text, scale)) // 2, y, text, colour, scale)
        y += GLYPH_H * scale + gap
    nf = max(1, int(fade * args.fps))
    for i in range(n):
        k = min(1.0, min(i, n - 1 - i) / nf) if n > 2 * nf else 1.0
        yield (np.clip(base * k, 0, 1) * 255).astype(np.uint8)


def frames_from(folder):
    names = sorted(f for f in os.listdir(folder) if f.endswith(".png"))
    for name in names:
        path = os.path.join(folder, name)
        if png_complete(path):
            yield read_png(path)


WHITE, WARM, COOL, DIM = (1, 1, 1), (1.00, 0.84, 0.52), (0.60, 0.76, 1.0), (0.52, 0.58, 0.70)

OPENING = [
    ("THE PHASES OF VENUS", 5 * UI, WHITE),
    ("", 1, WHITE),
    ("WHY VENUS SHOWS PHASES, WHY IT NEVER LEAVES THE SUN,", 2 * UI, COOL),
    ("AND WHY THE THIN CRESCENT LOOKS BIGGEST", 2 * UI, COOL),
    ("", 1, WHITE),
    ("EVERY POSITION SOLVED FROM VSOP87D AT RUN TIME", UI, DIM),
    ("NOTHING IN THIS FILM IS DRAWN BY HAND", UI, DIM),
]
PART1 = [
    ("PART ONE", 3 * UI, WARM),
    ("", 1, WHITE),
    ("WHY VENUS SHOWS PHASES", 4 * UI, WHITE),
    ("", 1, WHITE),
    ("THE SUN IS THE ONLY LIGHT IN THE SCENE, AND EACH BODY'S", UI, DIM),
    ("LIGHT DIRECTION COMES FROM ITS OWN POSITION - SO THE", UI, DIM),
    ("PHASE IS AN OUTPUT OF THE RAY TRACE, NEVER AN INPUT", UI, DIM),
]
PART2 = [
    ("PART TWO", 3 * UI, WARM),
    ("", 1, WHITE),
    ("2026 SEPTEMBER 14", 4 * UI, WHITE),
    ("THE MOON PASSES IN FRONT OF VENUS", 3 * UI, WHITE),
    ("", 1, WHITE),
    ("SEEN FROM CHENNAI, INDIA - IN DAYLIGHT", 2 * UI, COOL),
    ("", 1, WHITE),
    ("VENUS DISAPPEARS BECAUSE THE MOON'S SPHERE IS NEARER TO", UI, DIM),
    ("THE OBSERVER AND GETS IN THE WAY - THE SAME REASON IT", UI, DIM),
    ("HAPPENS IN THE SKY", UI, DIM),
]
CLOSING = [
    ("SOLVED, NOT DRAWN", 4 * UI, WHITE),
    ("", 1, WHITE),
    ("SUPERIOR CONJUNCTION      2026-01-06 16:01 UT    76 s FROM PUBLISHED", UI, COOL),
    ("GREATEST EASTERN ELONGATION  2026-08-15 06:30 UT   30 min FROM PUBLISHED", UI, COOL),
    ("INFERIOR CONJUNCTION      2026-10-24 03:49 UT    10 min FROM PUBLISHED", UI, COOL),
    ("", 1, WHITE),
    ("74 AUTOMATED CHECKS AGAINST MEEUS, THE IAU POLE TABLES", UI, DIM),
    ("AND PUBLISHED 2026 PHENOMENA - FOUR OF THEM MEASURED", UI, DIM),
    ("AGAINST RENDERED PIXELS RATHER THAN AGAINST CODE", UI, DIM),
    ("", 1, WHITE),
    ("PYTHON + TAICHI    NO TEXTURES, NO IMAGE ASSETS, NO EPHEMERIS FILES", UI, WARM),
]

import imageio.v2 as imageio                                          # noqa: E402

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                            quality=8, macro_block_size=1,
                            ffmpeg_params=["-pix_fmt", "yuv420p"])

count = 0


def emit(frames, what):
    """Write frames out, reporting progress so a long build is legible."""
    global count
    first = count
    for f in frames:
        if f.shape[0] != H or f.shape[1] != W:
            raise SystemExit(f"{what}: expected {W}x{H}, got {f.shape[1]}x{f.shape[0]} "
                             f"-- re-render that part at 1920x1080")
        writer.append_data(f)
        count += 1
        if count % 300 == 0:
            print(f"  {count:5d} frames  ({what})", flush=True)
    print(f"  {what}: {count - first} frames", flush=True)


print(f"assembling {args.out}")
emit(card(OPENING, 6.0), "opening card")
emit(card(PART1, 5.0), "part one card")
emit(frames_from(args.phases_frames), "part one: the phases of Venus")
emit(card(PART2, 6.0), "part two card")
emit(frames_from(args.occ_frames), "part two: the occultation")
emit(card(CLOSING, 8.0), "closing card")
writer.close()

size = os.path.getsize(args.out) / 1e6
print(f"\nwrote {args.out}")
print(f"  {count} frames, {count / args.fps / 60:.0f}m{count / args.fps % 60:02.0f}s, {size:.1f} MB")
