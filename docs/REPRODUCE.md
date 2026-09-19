# How every file in `media/` was made

Nothing here is hand-edited, retouched, or assembled in a video editor. Each
artifact is the direct output of one command. There are no image assets in the
project at all — no textures, no maps, no downloaded data — so a fresh checkout
plus these commands reproduces the media bit for bit, apart from the h.264
encoder's own nondeterminism.

## Everything at once

```bash
cd source
./run.sh --selftest        # first run only: builds the Python 3.12 venv
./rebuild.sh               # regenerates every file in media/
```

`rebuild.sh` prints what it is doing and roughly how long each step takes.

## Or one at a time

Run these from `source/`. `--arch gpu` selects Metal on macOS and
Vulkan/CUDA elsewhere; drop it to run on CPU, which works but is much slower.

| Output | Command | Time (M5 Pro, GPU) |
|---|---|---|
| `venus_phases.mp4` | `make phases` | ~8 min |
| `venus_phases_preview.mp4` | `make phases-preview` | ~2 min |
| `venus_phases_sheet.png` | `make phases-sheet` | ~1 min |
| `september2026.mp4` | `make animation` | ~1 min |
| `sequence_wide.png`, `sequence_eyepiece.png` | `make sequence` | ~4 min (CPU) |
| `chennai_*.png` | `make stills` | ~1 min |
| `footprint.png` | `make map` | ~1 min (**CPU only**) |
| `report.txt` | `make report` | seconds |
| `selftest/*.png` | `python main.py --selftest` | ~1 min |

The exact invocations behind the three films:

```bash
python scripts/render_venus_phases.py --arch gpu --out out/venus_phases.mp4
python scripts/render_venus_phases.py --preview --arch gpu --out out/venus_phases_preview.mp4
python scripts/render_animation.py   --arch gpu --out out/september2026.mp4
```

## Two things that will bite you

**The footprint map must run on CPU.** `FootprintSolver` holds its geocentric
vectors in explicit f64 fields, and Metal has no f64 — `make map` therefore does
not pass `--arch gpu`. Forcing it aborts with a `bind_pipeline` assertion.

**The first run compiles shaders for 2-3 minutes.** Taichi JITs the kernels, and
the lunar surface shader is large (cellular crater field, 18 named features,
per-pixel normal perturbation). It is cached afterwards (`offline_cache=True`),
so later runs start in seconds. Warm the cache before demoing on a fresh machine.

## Resuming a long render

The Venus film is 4500 frames. If something interrupts it:

```bash
python scripts/render_venus_phases.py --arch gpu --resume \
       --frames-dir out/phaseframes --out out/venus_phases.mp4
```

Frames are rendered to PNGs first and encoded in a second pass, so re-running
skips whatever is already on disk. Frames are written to a temp file and
`os.replace`d into position, and resume verifies each cached frame actually ends
with an `IEND` chunk rather than trusting that the file exists — a killed run
otherwise leaves a zero-byte frame that the encoder walks straight into.

## Checking the result rather than trusting it

```bash
make test        # 74 checks: Meeus worked examples, IAU pole tables,
                 # published 2026 phenomena, and four that measure rendered pixels
```

`make all` rebuilds everything including the 30-minute film — usually not what
you want.
