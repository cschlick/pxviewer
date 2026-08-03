"""File-kind routing and the browser-staging path for volumes (pxviewer.loader)."""

from __future__ import absolute_import, division, print_function

import json
import os
import sys

from libtbx.test_utils import raises

from pxviewer.regression.tst_utils import have, skip, tmp_dir

if not have("numpy"):
    skip("numpy not available")

import numpy as np                                   # noqa: E402

from pxviewer.loader import (FILE_DIALOG_FILTER, MODEL_FORMATS,  # noqa: E402
                             SAMPLE_STRUCTURE, VOLUME_FORMATS,
                             create_volume_file_view, file_kind,
                             sample_structure_path)
from pxviewer.volume import write_volume            # noqa: E402


def exercise_suffixes_classify_by_kind():
    """Every declared suffix routes to its kind. A loop rather than a parametrized case
    per suffix: the list is the point, and one failure naming the suffix is enough."""
    for suffix in sorted(MODEL_FORMATS):
        assert file_kind("/some/where/model%s" % suffix) == "model", suffix
    for suffix in sorted(VOLUME_FORMATS):
        assert file_kind("/some/where/map%s" % suffix) == "volume", suffix


def exercise_file_kind_is_case_insensitive():
    assert file_kind("MODEL.PDB") == "model"
    assert file_kind("MAP.MRC") == "volume"


def exercise_unsupported_suffix_names_the_supported_ones():
    with raises(ValueError) as e:
        file_kind("notes.txt")
    assert ".pdb" in str(e.value)


def exercise_dialog_filter_offers_combined_and_all_files():
    """The first entry accepts everything pxviewer reads, so Open just works; the per-kind
    entries and All files follow."""
    assert FILE_DIALOG_FILTER.startswith("All supported (")
    for suffix in ("*.pdb", "*.mrc", "*.mtz"):     # a model, a map, reflections
        assert suffix in FILE_DIALOG_FILTER.split(";;")[0]
    assert "Reflections (*.mtz)" in FILE_DIALOG_FILTER
    assert FILE_DIALOG_FILTER.endswith("All files (*)")


def exercise_a_missing_file_raises():
    with tmp_dir() as work:
        with raises(FileNotFoundError):
            create_volume_file_view(os.path.join(work, "absent.mrc"),
                                    out_dir=os.path.join(work, "out"))


def exercise_a_volume_is_copied_and_the_scene_points_at_the_copy():
    with tmp_dir() as work:
        src = os.path.join(work, "density.mrc")
        write_volume(np.zeros((4, 4, 4), dtype=np.float32), src, data_order="xyz")

        out = os.path.join(work, "served")
        mvsj_path = create_volume_file_view(src, out_dir=out)

        assert os.path.isfile(os.path.join(out, "density.mrc"))   # the copy served
        # The scene must use a bare filename so it resolves next to itself when served.
        with open(str(mvsj_path)) as fh:
            assert "density.mrc" in json.dumps(json.load(fh))


def exercise_model_files_are_rejected_here():
    """Models load through cctbx, not this browser-staging path."""
    with tmp_dir() as work:
        src = os.path.join(work, "model.pdb")
        with open(src, "w") as fh:
            fh.write("REMARK dummy\n")
        with raises(ValueError) as e:
            create_volume_file_view(src, out_dir=os.path.join(work, "out"))
        assert "loaded via cctbx" in str(e.value)


def exercise_the_bundled_sample_is_present_and_is_a_model():
    sample = sample_structure_path()
    assert sample is not None, "the bundled sample model is missing from pxviewer/data"
    assert sample.name == SAMPLE_STRUCTURE[0]
    assert file_kind(sample) == "model"


def exercise_each_load_can_target_a_fresh_directory():
    """Loading twice into separate dirs keeps the scenes independent (no cache bleed)."""
    with tmp_dir() as work:
        a = os.path.join(work, "a.mrc")
        b = os.path.join(work, "b.mrc")
        write_volume(np.zeros((4, 4, 4), dtype=np.float32), a, data_order="xyz")
        write_volume(np.zeros((4, 4, 4), dtype=np.float32), b, data_order="xyz")

        first = create_volume_file_view(a, out_dir=os.path.join(work, "served", "1"))
        second = create_volume_file_view(b, out_dir=os.path.join(work, "served", "2"))

        assert "a.mrc" in first.read_text()
        assert "b.mrc" in second.read_text()
        assert "a.mrc" not in second.read_text()


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
