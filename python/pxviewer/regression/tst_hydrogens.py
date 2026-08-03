"""reduce2-based hydrogen placement."""

from __future__ import absolute_import, division, print_function

import sys

from pxviewer.regression.tst_utils import data_path, have, skip

if not have("iotbx.data_manager", "mmtbx.programs.reduce2"):
    skip("iotbx data_manager / reduce2 not available")


def exercise_add_hydrogens_on_ubiquitin():
    from pxviewer.hydrogens import add_hydrogens, hydrogens_available

    if not hydrogens_available():
        print("  skipping: no monomer library "
              "(set MMTBX_CCP4_MONOMER_LIB to a geostd checkout)")
        return

    from iotbx.data_manager import DataManager

    dm = DataManager()
    dm.process_model_file(data_path("1ubq.pdb"))
    model = dm.get_model()
    before = list(model.get_hierarchy().atoms().extract_element())
    assert not any(e.strip().upper() == "H" for e in before)   # starts H-less

    h_model = add_hydrogens(model)
    after = list(h_model.get_hierarchy().atoms().extract_element())
    n_h = sum(1 for e in after if e.strip().upper() == "H")
    assert n_h > 500                # ubiquitin gains hundreds of hydrogens
    assert len(after) > len(before)


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
