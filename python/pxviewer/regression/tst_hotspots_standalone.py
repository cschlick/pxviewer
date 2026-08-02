"""``hotspots/`` must run without the viewer.

It is kept separable so it can be upstreamed, and it is meant to run headless on a compute
node for corpus work. Both stop being true the moment something there imports pxviewer or Qt
-- and it would keep working in a full development environment, so the breakage would only
show up on the machine that matters. Hence the subprocess: a fresh interpreter is the only
way to see what actually got imported.
"""

from __future__ import absolute_import, division, print_function

import json
import os
import subprocess
import sys

from pxviewer.regression.tst_utils import data_path, have, skip, tmp_dir

if not have("mmtbx"):
    skip("mmtbx not available")

#: Importing any of these means the directory is no longer standalone. rdkit is deliberately
#: absent: cctbx pulls it in itself, so its presence says nothing about hotspots.
VIEWER_ONLY = ("pxviewer", "PySide6", "PyQt5", "websockets", "molviewspec", "qtconsole",
               "ipykernel")


def _repo_root():
    import pxviewer

    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(pxviewer.__file__))))


def _hotspots_dir():
    path = os.path.join(_repo_root(), "hotspots")
    return path if os.path.isdir(path) else None


def _run(code):
    """Run code in a fresh interpreter, from the hotspots directory."""
    return subprocess.run([sys.executable, "-c", code], cwd=_hotspots_dir(),
                          capture_output=True, text=True, timeout=1800)


def exercise_runs_without_the_viewer():
    """Generate a small field in a subprocess and assert no viewer module was imported."""
    out = _run("""
import json, sys, tempfile
sys.path.insert(0, "hotspots")
from make_concern_maps import generate
manifest = generate(%r, tempfile.mkdtemp(), spacing=3.0, heavy_atom_clashes=True)
leaked = sorted({m.split('.')[0] for m in sys.modules} & set(%r))
print(json.dumps({"leaked": leaked, "metrics": sorted(manifest["outputs"])}))
""" % (data_path("1tec.pdb"), VIEWER_ONLY))
    assert out.returncode == 0, "generation failed:\n%s" % out.stderr[-3000:]
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["leaked"] == [], \
        "hotspots/ imported viewer-side modules: %s" % result["leaked"]
    assert "combined" in result["metrics"]


def exercise_finds_the_single_shared_extractor():
    """There is one validation_events.py and hotspots resolves it by path.

    If this breaks, the likely cause is a second copy appearing -- the drift the shared file
    exists to prevent -- or the canonical one moving.
    """
    out = _run("""
import sys
sys.path.insert(0, "hotspots")
from events import _load_shared
print(_load_shared().__file__)
""")
    assert out.returncode == 0, out.stderr[-2000:]
    resolved = os.path.realpath(out.stdout.strip().splitlines()[-1])
    expected = os.path.realpath(
        os.path.join(_repo_root(), "python", "pxviewer", "validation_events.py"))
    assert resolved == expected, "resolved %s, expected %s" % (resolved, expected)


def exercise_corpus_runner_survives_a_bad_model():
    """A corpus run must not stop on one bad model, and must QC what it produced.

    'It finished without raising' is not the same as 'the fields are right', which is why the
    runner checks the field back against the validation it came from.
    """
    with tmp_dir() as work:
        listing = os.path.join(work, "models.txt")
        broken = os.path.join(work, "not_a_model.pdb")
        with open(broken, "w") as fh:
            fh.write("this is not a PDB file\n")
        with open(listing, "w") as fh:
            fh.write("%s\n%s\n" % (data_path("1tec.pdb"), broken))
        out_dir = os.path.join(work, "out")

        out = _run("""
import sys
sys.path.insert(0, "hotspots")
sys.argv = ["run_corpus.py", %r, %r, "--output-pixel-size", "2.0",
            "--heavy-atom-clashes"]
import run_corpus
run_corpus.main()
""" % (listing, out_dir))
        assert out.returncode == 0, \
            "the runner should survive a bad model:\n%s" % out.stderr[-3000:]

        with open(os.path.join(out_dir, "corpus_results.jsonl")) as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        by_status = dict((r["stem"], r["status"]) for r in records)
        assert by_status.get("1tec") == "ok"
        assert by_status.get("not_a_model") == "failed"    # recorded, not fatal

        good = [r for r in records if r["stem"] == "1tec"][0]
        # At 2.0 A the field must still mark every flagged outlier; see the pixel-size
        # table in hotspots/README.md for what happens when this is relaxed.
        assert good["agreement"]["rama"]["recall"] == 1.0


def run():
    if _hotspots_dir() is None:
        skip("hotspots/ has been split back out")
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
