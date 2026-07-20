#!/usr/bin/env bash
# Build the Mol* frontend bundle (frontend/build/index.js) that the desktop app and
# the live web server serve. This is what the conda recipe copies into the package.
#
# The esbuild npm package ships a statically-linked native binary, already vendored
# under frontend/node_modules, so no node runtime is needed — but node_modules must be
# populated (npm's dependency tree: molstar, react). If it is missing, run `npm ci`
# in frontend/ once on a machine that has node.
#
#   ./scripts/build_frontend.sh
#
# Output: frontend/build/index.js (minified, ~3 MB). Re-run after editing frontend/src/*.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fe="$here/frontend"

# Prefer a real node/npm if present; otherwise use the vendored esbuild binary directly.
if command -v npm >/dev/null 2>&1 && [[ -d "$fe/node_modules/molstar" ]]; then
  ( cd "$fe" && npm run build )
  exit 0
fi

# Locate the vendored, statically-linked esbuild binary for this platform.
esbuild=""
for cand in \
  "$fe/node_modules/@esbuild"/*/bin/esbuild \
  "$fe/node_modules/esbuild/bin/esbuild"; do
  if [[ -x "$cand" ]]; then esbuild="$cand"; break; fi
done

if [[ -z "$esbuild" ]]; then
  echo "no esbuild found. Populate frontend/node_modules first:" >&2
  echo "    cd frontend && npm ci" >&2
  exit 1
fi
if [[ ! -d "$fe/node_modules/molstar" ]]; then
  echo "frontend/node_modules is missing its dependencies (molstar, react)." >&2
  echo "Populate them once: cd frontend && npm ci" >&2
  exit 1
fi

mkdir -p "$fe/build"
"$esbuild" "$fe/src/index.tsx" \
  --bundle --outfile="$fe/build/index.js" --minify --sourcemap
echo "built $fe/build/index.js"
