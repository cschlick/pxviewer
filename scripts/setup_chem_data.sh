#!/usr/bin/env bash
# Configure the active conda env to use everything the chem_data package ships, so
# cctbx's restraints and validation work without separately-managed data. Two things
# the package cannot do for itself:
#
#   1. The monomer library (geostd): cctbx finds it through MMTBX_CCP4_MONOMER_LIB,
#      which a package cannot set, so this writes an activate.d hook pointing it at
#      geostd inside the env. Needed by minimization, tugging, the geometry tables.
#   2. The validation caches: rotamer/Ramachandran and CaBLAM ship as *.data and are
#      loaded from *.pickle that must be built once (mmtbx.rebuild_*_cache), or the
#      Validation tools exit with "missing pickle" errors.
#
# Run once after `conda env create`, with the pxviewer env active:
#
#   conda activate pxviewer
#   ./scripts/setup_chem_data.sh
#   conda deactivate && conda activate pxviewer   # pick up the monomer-lib hook
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "no active conda env — run: conda activate pxviewer" >&2
  exit 1
fi

geostd="$CONDA_PREFIX/lib/python3.12/site-packages/chem_data/geostd"
if [[ ! -f "$geostd/a/data_ALA.cif" ]]; then
  echo "chem_data's geostd is not at $geostd" >&2
  echo "install it first: conda install -c chem_data chem_data" >&2
  exit 1
fi

# 1. Monomer library.
hook_dir="$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$hook_dir"
cat > "$hook_dir/pxviewer-monomer-lib.sh" <<'EOF'
# pxviewer: point cctbx at the monomer library shipped by the chem_data package.
# Resolved from $CONDA_PREFIX so it stays correct if the env is recreated elsewhere.
export MMTBX_CCP4_MONOMER_LIB="$CONDA_PREFIX/lib/python3.12/site-packages/chem_data/geostd"
EOF
echo "wrote $hook_dir/pxviewer-monomer-lib.sh"

# 2. Validation caches — build the pickles the rotamer/Ramachandran and CaBLAM
#    analyses load. One-off and idempotent (a rebuild just re-converts the *.data).
echo "building rotamer/Ramachandran cache (mmtbx.rebuild_rotarama_cache)…"
mmtbx.rebuild_rotarama_cache
echo "building CaBLAM cache (mmtbx.rebuild_cablam_cache)…"
mmtbx.rebuild_cablam_cache

echo "done. re-activate the env to apply the monomer-lib hook:"
echo "  conda deactivate && conda activate pxviewer"
