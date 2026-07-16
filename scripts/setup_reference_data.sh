#!/usr/bin/env bash
# Fetch + wire the Richardson-lab (Top8000) validation reference data that MolProbity
# validators need: rotamers (rotalyze), CaBLAM, and Rama-Z. (Ramachandran, cis/twisted
# peptides and Cbeta deviation work without it.)
#
# cctbx locates this data via libtbx.env.find_in_repositories("chem_data/..."), which
# only searches under $CONDA_PREFIX — there is no env-var override like geostd's
# MMTBX_CCP4_MONOMER_LIB. So we keep a git-ignored checkout in ./reference_data and
# symlink $CONDA_PREFIX/chem_data at it. Idempotent: safe to re-run (e.g. after the
# conda env is recreated, which drops the symlink).
#
# Usage:  conda activate pxviewer && scripts/setup_reference_data.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
RD="$REPO/reference_data"
REMOTE="https://github.com/rlabduke/reference_data.git"

if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "error: activate the conda env first (conda activate pxviewer)" >&2
  exit 1
fi

# 1) Sparse checkout — only the four Top8000 grid dirs we need (~264 MB), not the
#    whole ~148 MB-packed repo history.
if [ ! -d "$RD/.git" ]; then
  echo ">> cloning reference_data (sparse)…"
  git clone --filter=tree:0 --no-checkout --depth 1 "$REMOTE" "$RD"
  git -C "$RD" sparse-checkout init --cone
  git -C "$RD" sparse-checkout set \
    Top8000/Top8000_rotamer_pct_contour_grids \
    Top8000/Top8000_ramachandran_pct_contour_grids \
    Top8000/Top8000_cablam_pct_contour_grids \
    Top8000/rama_z
  git -C "$RD" checkout master
else
  echo ">> reference_data already checked out"
fi

# 2) Assemble the chem_data/ layout cctbx expects (symlinks — no duplication).
echo ">> assembling chem_data/…"
mkdir -p "$RD/chem_data/rotarama_data"
ln -sfn "$RD/Top8000/Top8000_cablam_pct_contour_grids" "$RD/chem_data/cablam_data"
ln -sfn "$RD/Top8000/rama_z" "$RD/chem_data/rama_z"
# rotarama_data = rotamer grids + ramachandran grids merged
ln -sf "$RD"/Top8000/Top8000_rotamer_pct_contour_grids/*.data "$RD/chem_data/rotarama_data/"
ln -sf "$RD"/Top8000/Top8000_ramachandran_pct_contour_grids/*.data "$RD/chem_data/rotarama_data/"

# 3) Wire into the conda env — the only place cctbx searches.
echo ">> linking \$CONDA_PREFIX/chem_data -> reference_data/chem_data"
ln -sfn "$RD/chem_data" "$CONDA_PREFIX/chem_data"

# 4) Build the .pickle caches cctbx wants (rotalyze + cablam read caches, not the
#    raw .data/.stat grids). Fast; skips work already done via a .dlite index.
echo ">> building rotarama + cablam caches…"
mmtbx.rebuild_rotarama_cache >/dev/null
mmtbx.rebuild_cablam_cache   >/dev/null

echo ">> done. All six MolProbity validators can now run."
