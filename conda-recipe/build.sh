#!/usr/bin/env bash
# Build script for pxviewer, shared by conda-build and rattler-build. Both run it from
# the source root with the host environment on PATH, so it uses cwd-relative paths and
# falls back to `python` when $PYTHON is unset. Bundles the built Mol* frontend inside
# the Python package, then installs it.
set -euo pipefail

# 1. Ensure the frontend bundle exists. It is normally pre-built with
#    scripts/build_frontend.sh before the package build (see the recipe header).
#
#    A source tree that already has frontend/node_modules can rebuild it here from the
#    vendored esbuild binary — recipe.yaml copies node_modules in via `use_gitignore:
#    false`. That is a convenience, not a declared build dependency: nodejs is
#    deliberately absent from the recipe's requirements, because the package ships the
#    bundle prebuilt and end users never compile it. On a clean checkout there is no
#    node_modules to fall back to, so fail here with the fix rather than deeper inside
#    build_frontend.sh, whose "run npm ci" advice does not apply mid-build. Mirrors the
#    same guard in bld.bat.
if [[ ! -f "frontend/build/index.js" ]]; then
  if [[ -d "frontend/node_modules/molstar" ]]; then
    echo "frontend/build/index.js missing — rebuilding from the vendored esbuild binary."
    bash scripts/build_frontend.sh
  else
    echo "frontend/build/index.js is missing, and frontend/node_modules is not populated" >&2
    echo "so it cannot be built here (nodejs is not a build requirement of this recipe)." >&2
    echo >&2
    echo "Build the bundle in the source tree first, then re-run the package build:" >&2
    echo "    cd frontend && npm ci        # once; needs npm (conda-forge: nodejs)" >&2
    echo "    ./scripts/build_frontend.sh  # -> frontend/build/index.js" >&2
    exit 1
  fi
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
