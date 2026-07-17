import json

import numpy as np
import pytest

from pxviewer.loader import (
    FILE_DIALOG_FILTER,
    MODEL_FORMATS,
    SAMPLE_STRUCTURE,
    VOLUME_FORMATS,
    create_volume_file_view,
    file_kind,
    sample_structure_path,
)
from pxviewer.volume import write_volume


@pytest.mark.parametrize("suffix", sorted(MODEL_FORMATS))
def test_model_suffixes_classify_as_model(suffix):
    assert file_kind(f"/some/where/model{suffix}") == "model"


@pytest.mark.parametrize("suffix", sorted(VOLUME_FORMATS))
def test_volume_suffixes_classify_as_volume(suffix):
    assert file_kind(f"/some/where/map{suffix}") == "volume"


def test_file_kind_is_case_insensitive():
    assert file_kind("MODEL.PDB") == "model"
    assert file_kind("MAP.MRC") == "volume"


def test_unsupported_suffix_names_the_supported_ones():
    with pytest.raises(ValueError, match=r"\.pdb"):
        file_kind("notes.txt")


def test_dialog_filter_offers_a_combined_and_an_all_files_entry():
    """The first entry accepts everything pxviewer reads, so Open just works; the
    per-kind entries and All files follow."""
    assert FILE_DIALOG_FILTER.startswith("All supported (")
    for suffix in ("*.pdb", "*.mrc", "*.mtz"):  # a model, a map, reflections
        assert suffix in FILE_DIALOG_FILTER.split(";;")[0]
    assert "Reflections (*.mtz)" in FILE_DIALOG_FILTER
    assert FILE_DIALOG_FILTER.endswith("All files (*)")


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        create_volume_file_view(tmp_path / "absent.mrc", out_dir=tmp_path / "out")


def test_volume_file_is_copied_and_scene_points_at_the_copy(tmp_path):
    src = tmp_path / "density.mrc"
    write_volume(np.zeros((4, 4, 4), dtype=np.float32), src, data_order="xyz")

    out = tmp_path / "served"
    mvsj_path = create_volume_file_view(src, out_dir=out)

    assert (out / "density.mrc").is_file()  # the copy the frontend will fetch
    # The scene must use a bare filename so it resolves next to itself when served.
    assert "density.mrc" in json.dumps(json.loads(mvsj_path.read_text()))


def test_model_files_are_rejected_here(tmp_path):
    """Models load through cctbx, not this browser-staging path."""
    src = tmp_path / "model.pdb"
    src.write_text("REMARK dummy\n")
    with pytest.raises(ValueError, match="loaded via cctbx"):
        create_volume_file_view(src, out_dir=tmp_path / "out")


def test_bundled_sample_is_present_and_is_a_model():
    sample = sample_structure_path()
    assert sample is not None, "the bundled sample model is missing from pxviewer/data"
    assert sample.name == SAMPLE_STRUCTURE[0]
    assert file_kind(sample) == "model"


def test_each_volume_load_can_target_a_fresh_directory(tmp_path):
    """Loading twice into separate dirs keeps the scenes independent (no cache bleed)."""
    a = tmp_path / "a.mrc"
    b = tmp_path / "b.mrc"
    write_volume(np.zeros((4, 4, 4), dtype=np.float32), a, data_order="xyz")
    write_volume(np.zeros((4, 4, 4), dtype=np.float32), b, data_order="xyz")

    first = create_volume_file_view(a, out_dir=tmp_path / "served" / "1")
    second = create_volume_file_view(b, out_dir=tmp_path / "served" / "2")

    assert "a.mrc" in first.read_text()
    assert "b.mrc" in second.read_text()
    assert "a.mrc" not in second.read_text()
