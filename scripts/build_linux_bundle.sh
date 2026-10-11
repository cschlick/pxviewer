#!/usr/bin/env bash
# Build a self-contained Linux tarball: a complete conda environment plus a
# launcher — the same "ship the whole env" model as the macOS dmg and the
# Windows setup.exe, packaged for distros that have no installer convention.
# Users untar anywhere and run ./pxviewer; an optional install.sh adds a
# desktop-menu entry. Works fully offline — nothing is downloaded at run time.
#
#   ./scripts/build_linux_bundle.sh
#
# Output: build/linux-bundle/pxviewer-<version>-linux-<arch>.tar.xz
#
# Requirements: Linux, a conda binary (miniforge is fine), and the frontend
# bundle already built (./scripts/build_frontend.sh — needs frontend/node_modules,
# populated by `npm ci` in frontend/). Build tools (rattler-build, conda-pack)
# are bootstrapped into build/linux-bundle/tools, so the user's base env is
# never touched.
#
# Portability: every binary comes from conda-forge packages, which target an
# old glibc sysroot — the tarball runs on distributions older than the build
# host (the reverse is not true, so build on the oldest runner available).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[[ "$(uname -s)" == "Linux" ]] || { echo "build_linux_bundle.sh is for Linux only" >&2; exit 1; }
ARCH="$(uname -m)"
VERSION="$(sed -n 's/^  version: "\(.*\)"$/\1/p' conda-recipe/recipe.yaml | head -1)"
[[ -n "$VERSION" ]] || { echo "could not read version from conda-recipe/recipe.yaml" >&2; exit 1; }

BUILD="$ROOT/build/linux-bundle"
NAME="pxviewer-$VERSION-linux-$ARCH"
STAGE="$BUILD/$NAME"
ENV_STAGING="$BUILD/env"                          # env created here, packed below
ENV_APP="$STAGE/env"                              # final location inside the bundle
TARBALL="$BUILD/$NAME.tar.xz"

# -- conda binary ------------------------------------------------------------
CONDA="$(command -v conda || true)"
if [[ -z "$CONDA" ]]; then
  for prefix in "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/miniconda3" /opt/miniforge3; do
    if [[ -x "$prefix/bin/conda" ]]; then CONDA="$prefix/bin/conda"; break; fi
  done
fi
[[ -n "$CONDA" ]] || { echo "conda not found — install Miniforge first" >&2; exit 1; }
echo "==> conda: $CONDA"

# -- frontend bundle ----------------------------------------------------------
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

# -- build tools (rattler-build + conda-pack), isolated from base --------------
TOOLS="$BUILD/tools"
if [[ ! -x "$TOOLS/bin/rattler-build" || ! -x "$TOOLS/bin/conda-pack" ]]; then
  echo "==> creating build-tools env (rattler-build + conda-pack)"
  "$CONDA" create -y -p "$TOOLS" -c conda-forge rattler-build conda-pack
fi

# -- build the conda package (the recipe is the single source of run deps) -----
echo "==> building pxviewer conda package"
"$TOOLS/bin/rattler-build" build --recipe conda-recipe/recipe.yaml \
  -c conda-forge -c chem_data --output-dir "$BUILD/pkg"
PKG="$(ls "$BUILD"/pkg/noarch/pxviewer-*.conda | head -1)"
echo "    package: $(basename "$PKG")"

# -- the app env: package + runtime extras the recipe can't express ------------
# scipy: hotspots uses scipy.ndimage; the recipe inherits it transitively, but
# name it here so the shipped env never depends on a transitive detail.
# qtconsole: conda-forge's qtconsole pins PyQt5 (GPL); pip's is binding-neutral
# and drives PySide6 via qtpy. pip-installed here exactly as environment.yml does.
echo "==> creating the app environment"
"$CONDA" create -y -p "$ENV_STAGING" --override-channels \
  -c "file://$BUILD/pkg" -c conda-forge -c chem_data \
  "pxviewer=$VERSION" scipy pip
"$ENV_STAGING/bin/python" -m pip install --quiet "qtconsole>=5.5"

# Validation caches — pickles land inside the env, so they ship in the bundle.
echo "==> building validation caches (rotamer/CaBLAM)"
PATH="$ENV_STAGING/bin:$PATH" CONDA_PREFIX="$ENV_STAGING" \
  bash scripts/setup_chem_data.sh

# Trim chem_data datasets nothing in the app can consume. Done post-pull so the
# recipe keeps depending on the upstream chem_data package unchanged — this is
# packaging-only, and the fat is what pushes the dmg past GitHub's 2 GiB asset
# cap. ligand_lib (1.5 GB): BLAST databases for eLBOW, which cctbx-base does not
# ship (nothing can reach it; only iotbx.local_blast references the path and
# pxviewer never calls that). segment_lib (357 MB): feeds only the standalone
# mmtbx.secondary_structure.regularize_from_pdb CLI tool, which pxviewer does
# not invoke. Everything else is load-bearing — geostd (geometry restraints),
# chemical_components (novel ligands), mon_lib (HEM etc.), and the
# rotamer/rotarama/cablam data behind the Validation tab.
echo "==> trimming unused chem_data (ligand_lib, segment_lib)"
rm -rf "$ENV_STAGING"/lib/python*/site-packages/chem_data/ligand_lib \
       "$ENV_STAGING"/lib/python*/site-packages/chem_data/segment_lib

# -- sanity: the env actually imports the app ----------------------------------
"$ENV_STAGING/bin/python" -c "import pxviewer.desktop"

# -- pack the env and unpack it inside the bundle -------------------------------
echo "==> conda-pack -> $ENV_APP"
rm -rf "$STAGE"                        # rebuilds start clean — no stale leftovers
mkdir -p "$ENV_APP"
"$TOOLS/bin/conda-pack" -p "$ENV_STAGING" -o "$BUILD/env.tar.gz" \
  --ignore-missing-files
tar -xzf "$BUILD/env.tar.gz" -C "$ENV_APP"
# conda-unpack rewrites embedded prefixes (shebangs, conda-meta, configs) to the
# *current* location — here the staging path; the launcher re-prefixes if the
# user later moves the unpacked tree.
"$ENV_APP/bin/python" "$ENV_APP/bin/conda-unpack"
echo "$ENV_APP" > "$ENV_APP/.pxviewer-prefix"

# -- launcher -------------------------------------------------------------------
# Same repair-in-background logic as the macOS launcher, with GNU sed syntax.
cat > "$STAGE/pxviewer" <<'LAUNCHER'
#!/usr/bin/env bash
# Launch pxviewer from the conda environment bundled beside this script.
HERE="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR="$HERE/env"

# Prefix-embedded files (bin/ shebangs, conda-meta, build configs) were
# rewritten to the build-time staging path by conda-unpack. If the tree has
# been moved since, re-prefix it in the background — nothing at runtime reads
# those records, and blocking here would stall first launch for minutes. The
# mkdir lock keeps a second launch from racing the first repair.
MARKER="$ENV_DIR/.pxviewer-prefix"
LOCK="$ENV_DIR/.prefix-repair.lock"
STORED="$(cat "$MARKER" 2>/dev/null || true)"
if [[ -w "$ENV_DIR" && "$STORED" != "$ENV_DIR" ]]; then
  # a lock older than an hour was orphaned by a killed repair — clear it
  find "$LOCK" -maxdepth 0 -type d -mmin +60 -exec rmdir {} \; 2>/dev/null || true
fi
if [[ -w "$ENV_DIR" && "$STORED" != "$ENV_DIR" ]] \
  && mkdir "$LOCK" 2>/dev/null; then
  (
    trap 'rmdir "$LOCK"' EXIT
    if [[ -n "$STORED" ]]; then
      find "$ENV_DIR" -type f -print0 2>/dev/null \
        | xargs -0 grep -IlF "$STORED" 2>/dev/null \
        | while IFS= read -r f; do
            sed -i "s|$STORED|$ENV_DIR|g" "$f"
          done
    else
      "$ENV_DIR/bin/python" "$ENV_DIR/bin/conda-unpack"
    fi
    echo "$ENV_DIR" > "$MARKER"
  ) &>/dev/null &
  disown
fi

export PATH="$ENV_DIR/bin:$PATH"
exec "$ENV_DIR/bin/python" -m pxviewer desktop "$@"
LAUNCHER
chmod +x "$STAGE/pxviewer"

# -- desktop-menu integration (optional, run once by the user) ------------------
cat > "$STAGE/install.sh" <<'INSTALL'
#!/usr/bin/env bash
# Install a freedesktop menu entry + icon for this pxviewer bundle.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
apps="$HOME/.local/share/applications"
icons="$HOME/.local/share/icons/hicolor/256x256/apps"
mkdir -p "$apps" "$icons"
cp "$HERE/icon.png" "$icons/pxviewer.png"
cat > "$apps/pxviewer.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=pxviewer
Comment=Mol*-based viewer for cctbx objects
Exec=$HERE/pxviewer
Icon=pxviewer
Terminal=false
Categories=Science;Education;
EOF
command -v update-desktop-database >/dev/null 2>&1 \
  && update-desktop-database "$apps" || true
echo "installed: $apps/pxviewer.desktop"
INSTALL
chmod +x "$STAGE/install.sh"

cp assets/icon.png "$STAGE/icon.png"
cp LICENSE "$STAGE/"

# -- tarball ----------------------------------------------------------------------
rm -f "$TARBALL"
echo "==> packing $TARBALL"
XZ_OPT=-T0 tar -cJf "$TARBALL" -C "$BUILD" "$NAME"

echo
echo "==> built: $TARBALL ($(du -h "$TARBALL" | cut -f1))"
echo "    usage: tar -xf $(basename "$TARBALL") && ./$NAME/pxviewer"
