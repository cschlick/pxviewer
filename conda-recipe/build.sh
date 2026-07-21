#!/usr/bin/env bash
# Build script for pxviewer, shared by conda-build and rattler-build. Both run it from
# the source root with the host environment on PATH, so it uses cwd-relative paths and
# falls back to `python` when $PYTHON is unset. Bundles the built Mol* frontend inside
# the Python package, then installs it.
set -euo pipefail

# 1. Ensure the frontend bundle exists. It is normally pre-built with
#    scripts/build_frontend.sh before the package build; build it here if it is missing
#    (that script errors clearly if node_modules has not been populated).
if [[ ! -f "frontend/build/index.js" ]]; then
  bash scripts/build_frontend.sh
fi

# 2. Copy the frontend runtime files into the package so the install is self-contained
#    (find_frontend_dir looks for pxviewer/frontend/ first). Only the served subset —
#    not node_modules, src or the sourcemap.
pkg_fe="python/pxviewer/frontend"
mkdir -p "$pkg_fe/build"
cp frontend/index.html     "$pkg_fe/"
cp frontend/app.html       "$pkg_fe/"
cp frontend/favicon.png    "$pkg_fe/"
cp frontend/build/index.js "$pkg_fe/build/"

# 3. Install the package (hatchling picks up pxviewer/frontend/ via the `artifacts` glob).
"${PYTHON:-python}" -m pip install ./python --no-deps --no-build-isolation -vv
