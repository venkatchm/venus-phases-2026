# YouTube title & description

## Title  (pick one)

    I Missed the Moon Covering Venus - So I Simulated It

    Why Venus Has Phases - Simulated From Scratch, Nothing Drawn

    The Night the Moon Covered Venus, and Why Venus Has Phases

## Description  (paste as-is; the timestamps are YouTube chapters)

On 14 September 2026 the Moon passed in front of Venus. From Chennai it happened
in broad daylight - Venus vanished behind the Moon's dark limb at 12:00:30 UT and
came back out 75 minutes later.

I missed it. So I wrote a simulation that puts it back, and then kept going: a
film that explains why Venus looks the way it does at all.

Nothing here is drawn by hand. There are no textures, no planet maps and no
ephemeris files anywhere in the project - every frame is computed. Positions come
from truncated VSOP87D and ELP-2000/82 series evaluated at run time, with
light-time correction, aberration, nutation and the topocentric transformation
for an observer on a rotating Earth.

The phase of Venus is never set. Each body's light direction is derived from its
own position, so the lit hemisphere cannot point anywhere except at the Sun - the
crescent is an output of the ray trace, not an input. Venus disappears behind the
Moon for the same reason it does in the sky: the Moon's sphere is nearer to the
observer and gets in the way.

The three moments the film is built around are solved, not typed in. The code
scans the elongation curve, finds the turning points and names each from the
geometry:

  Superior conjunction        2026-01-06 16:01 UT    76 seconds from published
  Greatest eastern elongation 2026-08-15 06:30 UT    30 minutes from published
  Inferior conjunction        2026-10-24 03:49 UT    10 minutes from published

74 automated checks back it up, against Meeus' worked examples, the IAU pole
tables and published 2026 phenomena. Four of them measure rendered pixels rather
than code - including one that checks the bright limb points at a Sun that is
off-screen, so nothing in the image could have been fitted to it.

Writing those tests found three errors that looked perfectly fine on screen: the
Earth's rotation axis 46.9 degrees off true, Venus spinning prograde when it
famously turns retrograde, and "inferior conjunction" computed by the wrong
definition - ten hours out, caught only by checking against a published time.

Code, full write-up and reproduction steps:
https://github.com/venkatchm/venus-phases-2026

Built with Python 3.12 and Taichi.

CHAPTERS
0:00 Introduction
0:11 The inner solar system
0:38 Two orbits, one light
1:11 The angle that makes the phase
1:36 What the Earth sees
1:46 Superior conjunction - full, and tiny
1:58 Greatest elongation - half lit
2:10 Towards inferior conjunction - the crescent grows
2:22 Same scale, same field: the proof
2:41 14 September 2026 - the Moon covers Venus
3:02 Disappearance
3:16 Reappearance

#astronomy #venus #simulation #python #spacevisualization

---

# Instagram Reel

`media/venus_reel.mp4` - 1080x1920 vertical, 44 s, 6.2 MB. Built by
`scripts/build_reel.py` from the same rendered frames as the long film.

## Caption

I missed it.

On 14 September 2026 the Moon passed in front of Venus. From Chennai it happened
in broad daylight — Venus slid behind the Moon's dark limb and came back out 75
minutes later. I wasn't looking.

So I built a simulation to see what I'd missed. Then I got curious about
something else: why does Venus have phases at all, and why does the thin crescent
look bigger than the full disc?

It isn't growing. It's closer — 1.71 AU down to 0.28.

The phases aren't drawn in. The Sun is the only light in the scene, so the lit
side can only ever face the Sun. And Venus disappears because the Moon is nearer
to us and gets in the way. Same reasons as the real sky.

Code and the full film in bio.

#astronomy #venus #space #python #simulation #astrophotography #science
#solarsystem #moon #occultation #creativecoding #dataviz
