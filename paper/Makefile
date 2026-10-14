# Regenerate paper figures from their saved views. Each figure's source is a
# .pxview.json + .pxview.data pair checked in beside this Makefile; running
# `make fig_gui.png` restores the view in the real app and re-photographs it,
# so figures track UI changes instead of going stale.
#
# Override PYTHON for a different interpreter; the default assumes a conda env
# named "pxviewer" (or pxviewer on PATH via `conda run`).

PYTHON ?= /opt/homebrew/Caskroom/miniforge/base/envs/pxviewer/bin/python

FIGURES := fig_gui.png

figures: $(FIGURES)

# FORCE: the pixels reflect the current UI, which mtime deps cannot see —
# `make figures` always re-photographs, never reports "up to date".
fig_gui.png: figure1.pxview.json regen_figure.py FORCE
	$(PYTHON) regen_figure.py figure1.pxview.json $@ --controls

FORCE:

pxviewer.pdf: pxviewer.tex $(FIGURES)
	tectonic pxviewer.tex

clean:
	rm -f pxviewer.pdf pxviewer.aux pxviewer.out

.PHONY: figures clean
