"""Generate local sample files for manually testing the desktop Load-file dialog.

Writes into ``tests/data`` (git-ignored) so you have real files on disk to browse
to when exercising the three load paths:

  * a model            -> Load file… -> 1ubq.pdb        (single model)
  * a map              -> Load file… -> 1ubq_map.mrc    (single volume)
  * a map + model group-> Load file… -> select both     (map_model_manager group)

The map is computed from the bundled model with cctbx (no download), so nothing
here needs to be committed. Run:

    python scripts/make_dev_fixtures.py
"""

import shutil
from pathlib import Path

from iotbx.data_manager import DataManager
from iotbx.map_model_manager import map_model_manager

from pxviewer.loader import sample_structure_path

OUT = Path(__file__).resolve().parents[1] / "tests" / "data"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    sample = sample_structure_path()
    if sample is None:
        raise SystemExit("bundled sample model not found (pxviewer/data)")

    # 1) the model, copied out where it's easy to browse to
    model_path = OUT / sample.name
    shutil.copyfile(sample, model_path)

    # 2) a density map computed from that model, via cctbx
    dm = DataManager()
    dm.process_model_file(str(sample))
    mmm = map_model_manager(model=dm.get_model())
    mmm.generate_map(d_min=3.0)
    map_path = OUT / f"{sample.stem}_map.mrc"
    mmm.map_manager().write_map(str(map_path))

    print(f"wrote:\n  {model_path}\n  {map_path}")
    print("\nIn the desktop app:")
    print("  Load file…  -> the .pdb           (model)")
    print("  Load file…  -> the .mrc           (volume)")
    print("  Load file…  -> select both files  (map+model group)")


if __name__ == "__main__":
    main()
