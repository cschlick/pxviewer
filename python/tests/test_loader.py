import json

import numpy as np
import pytest

from pxviewer.data import Atom, write_bcif
from pxviewer.loader import (
    FILE_DIALOG_FILTER,
    SAMPLE_STRUCTURE,
    STRUCTURE_FORMATS,
    VOLUME_FORMATS,
    create_file_view,
    file_kind,
    sample_structure_path,
    structure_format,
)
from pxviewer.volume import write_volume


@pytest.mark.parametrize("suffix", sorted(STRUCTURE_FORMATS))
def test_structure_suffixes_classify_as_structure(suffix):
    assert file_kind(f"/some/where/model{suffix}") == "structure"


@pytest.mark.parametrize("suffix", sorted(VOLUME_FORMATS))
def test_volume_suffixes_classify_as_volume(suffix):
    assert file_kind(f"/some/where/map{suffix}") == "volume"


def test_file_kind_is_case_insensitive():
    assert file_kind("MODEL.PDB") == "structure"
    assert file_kind("MAP.MRC") == "volume"


def test_unsupported_suffix_names_the_supported_ones():
    with pytest.raises(ValueError, match=r"\.pdb"):
        file_kind("notes.txt")


def test_structure_format_maps_to_parser_names():
    assert structure_format("a.pdb") == "pdb"
    assert structure_format("a.cif") == "mmcif"
    assert structure_format("a.bcif") == "bcif"
    with pytest.raises(ValueError):
        structure_format("a.mrc")


def test_dialog_filter_offers_a_combined_and_an_all_files_entry():
    assert FILE_DIALOG_FILTER.startswith("Structures and volumes (")
    assert "*.pdb" in FILE_DIALOG_FILTER and "*.mrc" in FILE_DIALOG_FILTER
    assert FILE_DIALOG_FILTER.endswith("All files (*)")


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        create_file_view(tmp_path / "absent.pdb", out_dir=tmp_path / "out")


def test_volume_file_is_copied_and_scene_points_at_the_copy(tmp_path):
    src = tmp_path / "density.mrc"
    write_volume(np.zeros((4, 4, 4), dtype=np.float32), src, data_order="xyz")

    out = tmp_path / "served"
    mvsj_path, kind = create_file_view(src, out_dir=out)

    assert kind == "volume"
    assert (out / "density.mrc").is_file()  # the copy the frontend will fetch
    # The scene must use a bare filename so it resolves next to itself when served.
    assert "density.mrc" in json.dumps(json.loads(mvsj_path.read_text()))


def test_structure_file_is_copied_and_parsed_in_its_own_format(tmp_path):
    src = tmp_path / "frag.bcif"
    write_bcif([Atom(id=1, element="C", name="C", resname="UNL", resseq=1, chain="A", x=0, y=0, z=0)], src)

    out = tmp_path / "served"
    mvsj_path, kind = create_file_view(src, out_dir=out)

    assert kind == "structure"
    assert (out / "frag.bcif").is_file()
    scene = mvsj_path.read_text()
    assert "frag.bcif" in scene
    assert "bcif" in scene  # parsed as bcif, not guessed as mmcif


def test_bundled_lysozyme_sample_is_present_and_is_a_structure():
    sample = sample_structure_path()
    assert sample is not None, "the bundled lysozyme sample is missing from tests/data"
    assert sample.name == SAMPLE_STRUCTURE[0]
    assert file_kind(sample) == "structure"

    text = sample.read_text()
    assert "LYSOZYME" in text.upper()
    assert sum(line.startswith("ATOM") for line in text.splitlines()) > 500


def test_lysozyme_sample_loads_into_a_scene(tmp_path):
    sample = sample_structure_path()
    mvsj_path, kind = create_file_view(sample, out_dir=tmp_path / "served")

    assert kind == "structure"
    assert (tmp_path / "served" / sample.name).is_file()

    scene = json.loads(mvsj_path.read_text())
    blob = json.dumps(scene)
    assert sample.name in blob
    assert '"pdb"' in blob  # parsed as PDB, not mis-guessed as mmCIF
    assert "cartoon" in blob  # the polymer actually gets a representation


def test_each_load_can_target_a_fresh_directory(tmp_path):
    """Loading twice into separate dirs keeps the scenes independent (no cache bleed)."""
    a = tmp_path / "a.mrc"
    b = tmp_path / "b.mrc"
    write_volume(np.zeros((4, 4, 4), dtype=np.float32), a, data_order="xyz")
    write_volume(np.zeros((4, 4, 4), dtype=np.float32), b, data_order="xyz")

    first, _ = create_file_view(a, out_dir=tmp_path / "served" / "1")
    second, _ = create_file_view(b, out_dir=tmp_path / "served" / "2")

    assert "a.mrc" in first.read_text()
    assert "b.mrc" in second.read_text()
    assert "a.mrc" not in second.read_text()
