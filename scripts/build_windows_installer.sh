#!/usr/bin/env bash
# Build pxviewer-<version>-windows-x86_64-setup.exe: a complete conda
# environment plus an NSIS installer — same "ship the env" model as the macOS
# app. Runs under Git Bash (on the windows-latest CI runner it is the default
# `run:` shell; locally it needs Git Bash + conda + NSIS).
#
#   bash scripts/build_windows_installer.sh
#
# Output: build/windows-installer/pxviewer-<version>-windows-x86_64-setup.exe
#
# Unlike macOS there is no launcher script — the Start Menu shortcut execs
# env\pythonw.exe directly, so the install-time step runs prefix_fixup.py to
# rewrite the env's embedded build prefix (qt.conf, kernelspecs, conda-meta).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[[ "$(uname -s)" == MINGW* || "$(uname -s)" == MSYS* || "$(uname -s)" == CYGWIN* ]] \
  || { echo "build_windows_installer.sh is for Windows only" >&2; exit 1; }
ARCH=x86_64
VERSION="$(sed -n 's/^  version: "\(.*\)"$/\1/p' conda-recipe/recipe.yaml | head -1)"
[[ -n "$VERSION" ]] || { echo "could not read version from conda-recipe/recipe.yaml" >&2; exit 1; }

BUILD="$ROOT/build/windows-installer"
ENV_STAGING="$BUILD/env"
STAGE="$BUILD/stage"
SETUP="$BUILD/pxviewer-$VERSION-windows-$ARCH-setup.exe"

to_win() { cygpath -w "$1"; }
to_posix() { cygpath -u "$1"; }

# -- conda binary ------------------------------------------------------------
# setup-miniconda puts condabin on PATH (`conda` resolves to a .bat that Git
# Bash executes fine). Fall back to CONDA_EXE / usual install dirs.
CONDA="$(command -v conda || true)"
if [[ -z "$CONDA" && -n "${CONDA_EXE:-}" ]]; then CONDA="$CONDA_EXE"; fi
if [[ -z "$CONDA" ]]; then
  for cand in "$USERPROFILE/miniconda3/Scripts/conda.exe" \
              /c/Miniconda3/Scripts/conda.exe \
              /c/tools/miniconda3/Scripts/conda.exe; do
    if [[ -x "$cand" ]]; then CONDA="$cand"; break; fi
  done
fi
[[ -n "$CONDA" ]] || { echo "conda not found" >&2; exit 1; }
echo "==> conda: $CONDA"

# -- frontend bundle ---------------------------------------------------------
if [[ ! -f frontend/build/index.js ]]; then
  if [[ -d frontend/node_modules/molstar ]]; then
    echo "==> frontend bundle missing — building"
    bash scripts/build_frontend.sh
  else
    echo "frontend/build/index.js missing and frontend/node_modules is not populated." >&2
    echo "  cd frontend && npm ci   # once" >&2
    exit 1
  fi
fi

# -- build tools (rattler-build), isolated from base --------------------------
# rattler-build is a Rust binary, not a Python entry point, so it does not land
# at Scripts/ where conda puts shims — locate it rather than assume the dir.
TOOLS="$BUILD/tools"
mkdir -p "$TOOLS"   # find below must not fail under pipefail on a fresh build dir
RATTLER="$(find "$TOOLS" -name 'rattler-build*' -type f 2>/dev/null | head -1)"
if [[ -z "$RATTLER" ]]; then
  echo "==> creating build-tools env (rattler-build)"
  "$CONDA" create -y -p "$(to_win "$TOOLS")" -c conda-forge rattler-build
  RATTLER="$(find "$TOOLS" -name 'rattler-build*' -type f | head -1)"
fi
[[ -n "$RATTLER" ]] || { echo "rattler-build not found in tools env" >&2; exit 1; }

# -- build the conda package (the recipe is the single source of run deps) ---
echo "==> building pxviewer conda package"
"$RATTLER" build --recipe "$(to_win conda-recipe/recipe.yaml)" \
  -c conda-forge -c chem_data --output-dir "$(to_win "$BUILD/pkg")"
PKG="$(ls "$BUILD"/pkg/noarch/pxviewer-*.conda | head -1)"
echo "    package: $(basename "$PKG")"

# -- the app env: package + runtime extras the recipe can't express ----------
# Same reasoning as scripts/build_macos_app.sh: scipy named explicitly,
# qtconsole via pip to avoid conda-forge's PyQt5 (GPL) pin.
echo "==> creating the app environment"
# A bare path is a valid channel; file://<win path> misparses ("d" becomes the
# URL host) — do not prepend the scheme on Windows.
"$CONDA" create -y -p "$(to_win "$ENV_STAGING")" --override-channels \
  -c "$(to_win "$BUILD/pkg")" -c conda-forge -c chem_data \
  "pxviewer=$VERSION" scipy pip
"$ENV_STAGING/python.exe" -m pip install --quiet "qtconsole>=5.5"

# Validation caches — pickles land inside the env, so they ship in the bundle.
echo "==> building validation caches (rotamer/CaBLAM)"
ENV_POSIX="$(to_posix "$ENV_STAGING")"
PATH="$ENV_POSIX:$ENV_POSIX/Scripts:$ENV_POSIX/Library/bin:$PATH" \
  CONDA_PREFIX="$ENV_POSIX" bash scripts/setup_chem_data.sh

# Trim chem_data datasets nothing in the app can consume — same list and
# reasoning as build_macos_app.sh (eLBOW is not shipped; segment_lib feeds an
# unused CLI tool). ~1.9 GB that would otherwise bloat the installer.
echo "==> trimming unused chem_data (ligand_lib, segment_lib)"
rm -rf "$ENV_STAGING"/Lib/site-packages/chem_data/ligand_lib \
       "$ENV_STAGING"/Lib/site-packages/chem_data/segment_lib

# -- sanity: the env actually imports the app ---------------------------------
"$ENV_STAGING/python.exe" -c "import pxviewer.desktop"

# -- staging: env + fixup + icon + license ------------------------------------
echo "==> staging install payload"
rm -rf "$STAGE"
mkdir -p "$STAGE"
cp -r "$ENV_STAGING" "$STAGE/env"
cp packaging/windows/prefix_fixup.py "$STAGE/"
cp LICENSE "$STAGE/"
"$ENV_STAGING/python.exe" - "$ROOT" "$(to_win "$STAGE")" <<'PY'
import sys
from PIL import Image
root, stage = sys.argv[1], sys.argv[2]
Image.open(root + "/assets/icon.png").save(
    stage + "/pxviewer.ico",
    sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
)
PY

# -- NSIS ---------------------------------------------------------------------
MAKENSIS="$(command -v makensis || true)"
if [[ -z "$MAKENSIS" ]]; then
  for cand in "/c/Program Files (x86)/NSIS/makensis.exe" "/c/Program Files/NSIS/makensis.exe"; do
    if [[ -x "$cand" ]]; then MAKENSIS="$cand"; break; fi
  done
fi
[[ -n "$MAKENSIS" ]] || { echo "makensis not found — install NSIS (choco install nsis)" >&2; exit 1; }
echo "==> makensis: $MAKENSIS"

# -D defines, not /D: an argument starting with / gets path-mangled by MSYS2
# ("/DVERSION=..." becomes "C:/Program Files/Git/DVERSION=...").
rm -f "$SETUP"
"$MAKENSIS" "-DVERSION=$VERSION" \
  "-DSTAGEDIR=$(to_win "$STAGE")" \
  "-DBUILDPREFIX=$(to_win "$ENV_STAGING")" \
  "-DOUTFILE=$(to_win "$SETUP")" \
  packaging/windows/installer.nsi

echo
echo "==> built: $SETUP ($(du -h "$SETUP" | cut -f1))"
echo "    unsigned: first run trips SmartScreen -> 'More info' -> 'Run anyway'"
