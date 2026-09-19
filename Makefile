PY := .venv/bin/python

.PHONY: all run test report map sequence stills animation preview clean \
        phases phases-preview phases-sheet phases-storyboard

run:                    ## launch the interactive simulation
	$(PY) main.py

test:                   ## ephemeris regression tests, and renderer invariants
	$(PY) tests/test_ephemeris.py
	$(PY) tests/test_render_geometry.py

report:                 ## text timeline for the whole of September 2026
	$(PY) scripts/report.py --site Chennai --save out/report.txt

map:                    ## global occultation visibility footprint
	$(PY) scripts/render_map.py

sequence:               ## contact sheets of the occultation
	$(PY) scripts/render_sequence.py --site Chennai --out out/sequence_wide.png
	$(PY) scripts/render_sequence.py --site Chennai --centre venus --fov 0.11 \
	      --tile 480 --out out/sequence_eyepiece.png

stills:                 ## a few individual sky frames
	$(PY) scripts/render_frames.py --site Chennai --fov 2.0

animation:              ## the full September story as an MP4 (~30 s)
	$(PY) scripts/render_animation.py --out out/september2026.mp4

preview:                ## fast low-res version of the animation
	$(PY) scripts/render_animation.py --preview --out out/preview.mp4

phases:                 ## the cinematic Venus-phases film (~2:30)
	$(PY) scripts/render_venus_phases.py --arch gpu --out out/venus_phases.mp4

phases-preview:         ## fast low-res check of the phases film
	$(PY) scripts/render_venus_phases.py --preview --arch gpu \
	      --out out/venus_phases_preview.mp4

phases-sheet:           ## contact sheet of Venus through the 2026 apparition
	$(PY) scripts/render_phase_stills.py --arch gpu

phases-storyboard:      ## one still per shot of the phases film, for review
	$(PY) scripts/render_venus_phases.py --arch gpu --storyboard out/storyboard

all: test report map sequence stills animation phases-sheet phases

clean:
	rm -rf out __pycache__ */__pycache__
