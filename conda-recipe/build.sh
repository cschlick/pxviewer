#!/usr/bin/env bash
# conda-build script for pxviewer (Linux/macOS). Bundles the built Mol* frontend
# inside the Python package, then installs it. Runs from the recipe's copy of the
# repo ($SRC_DIR).
set -euo pipefail

# 1. Ensure the frontend bundle exists. It is normally pre-built with
#    scripts/build_frontend.sh before `conda build`; build it here if it is missing
#    and node_modules is available (skip with PXVIEWER_SKIP_FRONTEND_BUILD=1).
if [[ ! -f "$SRC_DIR/frontend/build/index.js" ]]; then
  if [[ "${PXVIEWER_SKIP_FRONTEND_BUILD:-0}" == "1" ]]; then
    echo "frontend/build/index.js missing and PXVIEWER_SKIP_FRONTEND_BUILD=1" >&2
    exit 1
  fi
  bash "$SRC_DIR/scripts/build_frontend.sh"
fi

# 2. Copy the frontend runtime files into the package so the install is self-contained
#    (find_frontend_dir looks for pxviewer/frontend/ first). Only the served subset —
#    not node_modules, src or the sourcemap.
pkg_fe="$SRC_DIR/python/pxviewer/frontend"
mkdir -p "$pkg_fe/build"
cp "$SRC_DIR/frontend/index.html"     "$pkg_fe/"
cp "$SRC_DIR/frontend/app.html"       "$pkg_fe/"
cp "$SRC_DIR/frontend/favicon.png"    "$pkg_fe/"
cp "$SRC_DIR/frontend/build/index.js" "$pkg_fe/build/"

# 3. Install the package (hatchling picks up pxviewer/frontend/ via the `artifacts` glob).
$PYTHON -m pip install ./python --no-deps --no-build-isolation -vv
