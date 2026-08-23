#!/usr/bin/env bash
# Build the validation caches from the data the chem_data package ships: rotamer/
# Ramachandran and CaBLAM ship as *.data but are loaded from *.pickle, which must be
# built once (mmtbx.rebuild_*_cache) or the Validation tools exit with "missing pickle".
#
# The monomer library needs nothing here. cctbx finds both directories chem_data ships
# (geostd and mon_lib) through its own repository cascade, and pxviewer's import checks
# that on the way in (pxviewer.geometry.configure_monomer_library).
#
# This script used to also write an activate.d hook exporting MMTBX_CCP4_MONOMER_LIB at
# geostd. That was worse than redundant: the variable is a single-directory redirect
# consulted *before* cctbx's cascade, so it pinned the search to geostd and made every
# monomer carried only by mon_lib -- HEM among them -- stop resolving, while the other
# ~54k kept working. It also hard-coded a python3.12 path that a version bump would
# invalidate. pxviewer now ignores such a redirect; see tst_monomer_library.py. If an
# old hook is still in your env, delete it:
#
#   rm -f "$CONDA_PREFIX/etc/conda/activate.d/pxviewer-monomer-lib.sh"
#
# Run once after `conda env create`, with the pxviewer env active:
#
#   conda activate pxviewer
#   ./scripts/setup_chem_data.sh
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "no active conda env — run: conda activate pxviewer" >&2
  exit 1
fi

# Resolve chem_data by importing it rather than guessing a site-packages path, which is
# what pxviewer itself does -- no python-version string to go stale.
if ! python -c "import chem_data" 2>/dev/null; then
  echo "chem_data is not importable in this env" >&2
  echo "install it first: conda install -c chem_data chem_data" >&2
  exit 1
fi

# Remove the activation hook older versions of this script wrote: it pins cctbx to
# geostd and hides mon_lib. Harmless to run when there is nothing to remove.
hook="$CONDA_PREFIX/etc/conda/activate.d/pxviewer-monomer-lib.sh"
if [[ -f "$hook" ]]; then
  rm -f "$hook"
  echo "removed stale monomer-library hook: $hook"
  echo "  (it pinned cctbx to geostd, hiding the mon_lib monomers such as HEM)"
fi

# Validation caches — build the pickles the rotamer/Ramachandran and CaBLAM
# analyses load. One-off and idempotent (a rebuild just re-converts the *.data).
echo "building rotamer/Ramachandran cache (mmtbx.rebuild_rotarama_cache)…"
mmtbx.rebuild_rotarama_cache
echo "building CaBLAM cache (mmtbx.rebuild_cablam_cache)…"
mmtbx.rebuild_cablam_cache

echo "done."
