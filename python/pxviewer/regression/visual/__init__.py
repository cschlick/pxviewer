"""Visual passes: the scripted half of VISUAL_TESTS.md.

Each ``tst_visual_*.py`` here drives the real desktop app — real QtWebEngine
viewport, real Mol* render — through one of the chained sequences the document
describes, asserting on the state each step leaves behind and writing viewport
screenshots to the run directory for review. ``harness.py`` is the shared
driver; it is not a test.
"""
