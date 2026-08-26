"""The frontend's localres grid helpers, run under node against independent references.

``cropGrid`` and ``decimateGrid`` (frontend/src/live.ts) are what make a level change on
a colour-by-resolution map cheap: contouring visits only the sub-box holding voxels at or
above the level (12.7x fewer voxels at EMD-53478's deposited level), and slider drags
preview from a 2x-decimated copy (8x fewer even when the crop cannot help). An indexing
mistake in either would not fail loudly -- it would draw the surface subtly wrong or in
the wrong place -- so their arithmetic is asserted in frontend/src/localres-grids.test.ts
against independently-written traversals, and this script runs it the only way frontend
code can run: bundled by the vendored esbuild, executed by node.

Skips cleanly when the frontend toolchain is absent (an installed package, or a checkout
that never ran `npm ci`), same as the GUI tests skip without Qt.
"""

from __future__ import absolute_import, division, print_function

import os
import shutil
import subprocess
import sys
import tempfile

from pxviewer.regression.tst_utils import skip

_here = os.path.dirname(os.path.abspath(__file__))
_frontend = os.path.abspath(os.path.join(_here, os.pardir, os.pardir, os.pardir, "frontend"))
_esbuild = os.path.join(_frontend, "node_modules", "@esbuild", "darwin-arm64", "bin", "esbuild")
if not os.path.isdir(_frontend):
    skip("no frontend/ in this tree (installed package)")
if not os.path.isfile(os.path.join(_frontend, "src", "localres-grids.test.ts")):
    skip("frontend grid test not present")
node = shutil.which("node")
if node is None:
    skip("node not on PATH (environment.yml provides it)")
if not os.path.isfile(_esbuild):
    # The vendored binary is platform-specific; fall back to any esbuild on PATH.
    _esbuild = shutil.which("esbuild") or ""
    if not _esbuild:
        skip("no esbuild (run `npm ci` in frontend/)")


def exercise_the_grid_helpers_match_their_references():
    with tempfile.TemporaryDirectory() as work:
        bundle = os.path.join(work, "grids.test.js")
        subprocess.run(
            [_esbuild, os.path.join(_frontend, "src", "localres-grids.test.ts"),
             "--bundle", "--platform=node", "--outfile=" + bundle],
            check=True, capture_output=True, cwd=_frontend)
        result = subprocess.run([node, bundle], capture_output=True, text=True)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        assert result.returncode == 0, "grid helper assertions failed under node"
        assert "OK (grid helpers)" in result.stdout


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
