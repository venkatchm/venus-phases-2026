# Phases are not shadows

The single most common misconception in astronomy, and worth stating plainly
because this project is built entirely on the thing it gets wrong.

**Venus' phases are not caused by any shadow.** Not the Earth's, not the Moon's,
not Venus' own. The same is true of the Moon's phases, which is where the
confusion usually starts.

## Why a shadow cannot be the explanation

Take the moment in this simulation when Venus was hidden behind the Moon, on
2026 September 14 at 12:30 UT from Chennai:

| | |
|---|---|
| Moon distance | 388,842 km |
| Venus distance | **67,893,722 km** |
| | Venus was **175× farther away** |

The Earth's shadow is a cone that tapers to nothing roughly 1,400,000 km behind
us. Venus was **48 times beyond the tip of it** — and on the *sunward* side of
the Earth in any case, 41° from the Sun. It was never anywhere near our shadow
and never can be.

The Moon's shadow is shorter still. It only just reaches the Earth, which is
exactly why a total solar eclipse is visible from such a narrow strip. It cannot
reach something 67 million km away.

The Moon and Venus were close only in *appearance*. They lined up along one
sight line from southern India, the way a coin at arm's length covers a distant
building. In space they were 67 million km apart.

## What actually causes them

**Venus is always exactly half lit.** It is a sphere with the Sun off to one
side, so one hemisphere is in daylight and the other is in night, permanently.
That never changes.

What changes is **how much of the lit half is turned towards us**:

| Where Venus is | What we see |
|---|---|
| Far side of the Sun (superior conjunction) | its day side, face on — **full**, and tiny |
| Off to one side (greatest elongation) | day and night side-on — **half lit** |
| Between us and the Sun (inferior conjunction) | mostly its night side — a thin **crescent**, and large |

On 2026 September 14 the Sun–Venus–Earth angle was **114.7°**, putting **29.1 %**
of the visible disc in sunlight. Venus was moving towards inferior conjunction,
coming round to our side of the Sun. It would have shown that crescent that day
whether or not the Moon happened to pass in front of it.

This is also why the crescent is the *biggest* phase rather than the smallest:
the geometry that turns Venus' night side towards us is the same geometry that
brings it closest. Lit fraction and apparent size move in opposite directions,
and both are consequences of one orbital position.

## How the code settles it

There is no shadow-casting code anywhere in this project. No shadow rays, no
occlusion test against a light source, no shadow maps. A grep for them returns
nothing.

Two honest footnotes, so nobody catches this overclaiming. The lunar craters
*do* darken on the side facing away from the Sun, and that reads as shadow — but
it is shading, not a cast shadow: the crater height field perturbs the surface
normal, and a normal tilted away from the Sun simply receives less light. No ray
is traced towards the Sun to ask whether anything is in the way. And `render/text.py`
contains the word `shadow`, which is the drop shadow behind HUD lettering and
has nothing to do with lighting at all.

The phase falls out of a single dot product in `render/space.py`:

```python
mu = ti.math.dot(n, light)                 # surface normal against sunward direction
lit = ti.math.smoothstep(-0.16, 0.22, mu)  # facing the Sun -> lit; facing away -> dark
```

and the light direction itself is derived from the body's own position in
`render/bridge.py` — with the Sun at the origin, `normalize(sun - body)` is just
`-normalize(pos)`. The lit hemisphere therefore *cannot* point anywhere except
at the Sun. The phase is an output of the trace and there is no way to set it by
hand even if you wanted to.

`tests/test_render_geometry.py` checks this against rendered pixels rather than
against the code: the brightness centroid of a partly lit sphere must sit off
its geometric centre, sunward, and the bright limb must point at the Sun in the
telescopic shots — where the Sun is off frame, so nothing in the image could have
been fitted to it. Measured at r = 0.999–1.000.

## The one thing that *is* a blocking event

Venus disappearing behind the Moon is not a shadow either. It is an
**occultation**: the Moon's solid body physically got between the observer and
Venus.

In the renderer that happens for the same reason it happens in the sky — depth
ordering. Both spheres are traced, and the nearer one wins:

```python
if t_v > 0.0 and t_v * self.venus_dist[None] < depth:
```

Which is also why it was a local event. An occultation depends on exactly where
you stand, because the Moon is only half a degree wide and 60 Earth radii away.
Chennai saw it. New York saw nothing, and
`tests/test_ephemeris.py` asserts that too.

## In one line

> Phases are about **which way a lit sphere is facing**. Shadows are about
> **something getting in the way**. Venus does the first every day of its orbit;
> the Moon did the second for 75 minutes on 2026 September 14.
