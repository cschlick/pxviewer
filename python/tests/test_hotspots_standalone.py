"""`hotspots/` must run without the viewer.

It is kept separable so it can be upstreamed, and it is meant to run headless on a compute
node for corpus work. Both of those stop being true the moment something there imports
pxviewer or Qt — and it would keep working here, where everything is installed, so the
breakage would only show up on the machine that matters.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mmtbx")

_HOTSPOTS = Path("hotspots")
_MODEL = Path("python/pxviewer/data/1tec.pdb")

#: Importing any of these means the directory is no longer standalone. rdkit is deliberately
#: absent: cctbx pulls it in itself, so its presence says nothing about hotspots.
_VIEWER_ONLY = ("pxviewer", "PySide6", "PyQt5", "websockets", "molviewspec", "qtconsole",
                "ipykernel")


@pytest.fixture(scope="module")
def hotspots_dir():
    if not _HOTSPOTS.is_dir():
        pytest.skip("hotspots/ has been split back out")
    return _HOTSPOTS


def _run(hotspots_dir, code):
    """Run code in a fresh interpreter with only hotspots' own package importable."""
    return subprocess.run([sys.executable, "-c", code], cwd=hotspots_dir,
                          capture_output=True, text=True, timeout=900)


def test_hotspots_runs_without_the_viewer(hotspots_dir):
    """Generate a small field in a subprocess and assert no viewer module was imported."""
    out = _run(hotspots_dir, f"""
import json, sys, tempfile
sys.path.insert(0, "hotspots")
from make_concern_maps import generate
manifest = generate({str(_MODEL.resolve())!r}, tempfile.mkdtemp(),
                    spacing=3.0, heavy_atom_clashes=True)
leaked = sorted({{m.split('.')[0] for m in sys.modules}} & set({_VIEWER_ONLY!r}))
print(json.dumps({{"leaked": leaked, "metrics": sorted(manifest["outputs"])}}))
""")
    assert out.returncode == 0, f"generation failed:\n{out.stderr[-3000:]}"
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["leaked"] == [], (
        f"hotspots/ imported viewer-side modules: {result['leaked']}")
    assert "combined" in result["metrics"]


def test_hotspots_finds_the_single_shared_extractor(hotspots_dir):
    """There is one validation_events.py and hotspots resolves it by path.

    If this breaks, the likely cause is someone dropping a second copy in — which is the
    drift the shared file exists to prevent — or moving the canonical one.
    """
    out = _run(hotspots_dir, """
import sys
sys.path.insert(0, "hotspots")
from events import _load_shared
print(_load_shared().__file__)
""")
    assert out.returncode == 0, out.stderr[-2000:]
    resolved = Path(out.stdout.strip().splitlines()[-1]).resolve()
    assert resolved == Path("python/pxviewer/validation_events.py").resolve()


def test_the_corpus_runner_reports_recall_and_survives_a_bad_model(hotspots_dir, tmp_path):
    """A corpus run must not stop on one bad model, and must QC what it produced.

    'It finished without raising' is not the same as 'the fields are right', which is the
    whole reason the runner checks the field back against the validation it came from.
    """
    listing = tmp_path / "models.txt"
    broken = tmp_path / "not_a_model.pdb"
    broken.write_text("this is not a PDB file\n")
    listing.write_text(f"{_MODEL.resolve()}\n{broken}\n")

    out = _run(hotspots_dir, f"""
import sys
sys.path.insert(0, "hotspots")
sys.argv = ["run_corpus.py", {str(listing)!r}, {str(tmp_path / 'out')!r},
            "--output-pixel-size", "2.0", "--heavy-atom-clashes"]
import run_corpus
run_corpus.main()
""")
    assert out.returncode == 0, f"the runner should survive a bad model:\n{out.stderr[-3000:]}"

    records = [json.loads(line) for line in
               (tmp_path / "out" / "corpus_results.jsonl").read_text().splitlines() if line]
    by_status = {r["stem"]: r["status"] for r in records}
    assert by_status.get("1tec") == "ok"
    assert by_status.get("not_a_model") == "failed"   # recorded, not fatal

    good = next(r for r in records if r["stem"] == "1tec")
    # At 2.0 A the field must still mark every flagged outlier; see the pixel-size table
    # in hotspots/README.md for what happens when this is relaxed.
    assert good["agreement"]["rama"]["recall"] == 1.0
