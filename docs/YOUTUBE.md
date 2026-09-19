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

So I built a simulation to see what I'd missed.

Then I got something wrong. Venus shows up as a crescent in this, and I assumed
that was a shadow falling on it. It isn't. Nothing is casting a shadow on Venus.
Not the Earth, not the Moon.

Venus is always exactly half lit. It's a ball with the Sun off to one side, so
one half is in daylight and the other is in night — permanently, that never
changes. What changes is how much of the lit half is turned towards us.

Far side of the Sun, we're looking at its day side: full, and tiny.
Off to one side: half lit.
Swinging round between us and the Sun, we're looking mostly at its night side: a
thin crescent.

That day the Sun–Venus–Earth angle was 115°, so 29% of the face we could see was
in sunlight. It would have been a crescent whether or not the Moon happened to
pass in front of it.

And the part I still find strange: the crescent is the biggest phase, not the
smallest. The same position that turns Venus's night side towards us is the one
that brings it closest — 1.71 AU down to 0.28.

The Moon and Venus weren't actually near each other either. The Moon was 389,000
km away. Venus was 68 million. They just happened to line up from where I was
standing.

Code and the full film in bio.

#astronomy #venus #space #python #simulation #astrophotography #science
#solarsystem #moon #occultation #creativecoding #dataviz
