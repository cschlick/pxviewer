#!/usr/bin/env bash
# Build pxviewer.app + a distributable .dmg: a complete conda environment packed
# inside a real app bundle — the same pattern Phenix/ChimeraX ship. Users drag
# the app into /Applications; no conda install, no terminal.
#
#   ./scripts/build_macos_app.sh
#
# Output: build/macos-app/pxviewer-<version>-macos-<arch>.dmg
#
# Requirements: macOS, a conda binary (miniforge is fine), and the frontend
# bundle already built (./scripts/build_frontend.sh — needs frontend/node_modules,
# populated by `npm ci` in frontend/). Build tools (rattler-build, conda-pack)
# are bootstrapped into build/macos-app/tools, so the user's base env is never
# touched.
#
# The .app layout:
#   pxviewer.app/Contents/MacOS/pxviewer        launcher -> env python
#   pxviewer.app/Contents/Resources/env/        conda-packed environment
#   pxviewer.app/Contents/Info.plist            bundle metadata + icon ref
#
# Signing: the bundle is ad-hoc signed, which is all arm64 requires to RUN
# locally — but a downloaded unsigned app is quarantined by Gatekeeper, so the
# first launch needs right-click -> Open (or `xattr -dr com.apple.quarantine`).
# With a Developer ID identity this becomes a one-flag change: export
# SIGNING_IDENTITY="Developer ID Application: <name>" and re-run; notarization
# (`xcrun notarytool`) is then the remaining step for a warning-free install.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[[ "$(uname -s)" == "Darwin" ]] || { echo "build_macos_app.sh is for macOS only" >&2; exit 1; }
ARCH="$(uname -m)"
VERSION="$(sed -n 's/^  version: "\(.*\)"$/\1/p' conda-recipe/recipe.yaml | head -1)"
[[ -n "$VERSION" ]] || { echo "could not read version from conda-recipe/recipe.yaml" >&2; exit 1; }

BUILD="$ROOT/build/macos-app"
STAGE="$BUILD/dmg-root"
APP="$STAGE/pxviewer.app"
ENV_STAGING="$BUILD/env"                          # env created here, packed below
ENV_APP="$APP/Contents/Resources/env"             # final location inside the bundle
DMG="$BUILD/pxviewer-$VERSION-macos-$ARCH.dmg"

# -- conda binary -----------------------------------------------------------
# No activation needed anywhere below — only `conda create` — so the plain
# binary suffices; check PATH, then the usual prefixes.
CONDA="$(command -v conda || true)"
if [[ -z "$CONDA" ]]; then
  for cand in "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/miniconda3" \
              /opt/miniforge3 /opt/homebrew/Caskroom/miniforge/base; do
    if [[ -x "$cand/bin/conda" ]]; then CONDA="$cand/bin/conda"; break; fi
  done
fi
[[ -n "$CONDA" ]] || { echo "conda not found — install Miniforge first" >&2; exit 1; }
echo "==> conda: $CONDA"

# -- frontend bundle --------------------------------------------------------
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

# -- build tools (rattler-build + conda-pack), isolated from base ------------
TOOLS="$BUILD/tools"
if [[ ! -x "$TOOLS/bin/rattler-build" || ! -x "$TOOLS/bin/conda-pack" ]]; then
  echo "==> creating build-tools env (rattler-build + conda-pack)"
  "$CONDA" create -y -p "$TOOLS" -c conda-forge rattler-build conda-pack
fi

# -- build the conda package (the recipe is the single source of run deps) ---
echo "==> building pxviewer conda package"
"$TOOLS/bin/rattler-build" build --recipe conda-recipe/recipe.yaml \
  -c conda-forge -c chem_data --output-dir "$BUILD/pkg"
PKG="$(ls "$BUILD"/pkg/noarch/pxviewer-*.conda | head -1)"
echo "    package: $(basename "$PKG")"

# -- the app env: package + runtime extras the recipe can't express ----------
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

# -- sanity: the env actually imports the app --------------------------------
"$ENV_STAGING/bin/python" -c "import pxviewer.desktop"

# -- pack the env and unpack it inside the .app -------------------------------
echo "==> conda-pack -> $ENV_APP"
# --ignore-missing-files: conda's python.app package gets its launcher files
# rewritten by its own post-link, so its manifest never matches disk. The
# launcher is unused (we exec env/bin/python), but conda-pack aborts on the
# mismatch without this flag.
"$TOOLS/bin/conda-pack" -p "$ENV_STAGING" -o "$BUILD/env.tar.gz" \
  --ignore-missing-files
rm -rf "$STAGE"                        # rebuilds start clean — no stale leftovers
mkdir -p "$ENV_APP"
tar -xzf "$BUILD/env.tar.gz" -C "$ENV_APP"
# conda-unpack rewrites embedded prefixes (shebangs, conda-meta, configs) to the
# *current* location — here the staging path; the launcher re-prefixes if the
# user later moves the app.
"$ENV_APP/bin/python" "$ENV_APP/bin/conda-unpack"
echo "$ENV_APP" > "$ENV_APP/.pxviewer-prefix"

# -- launcher ----------------------------------------------------------------
# LaunchServices refuses apps whose CFBundleExecutable is a script (error
# -10669 since macOS 11), so Contents/MacOS/pxviewer is a tiny compiled stub
# that execs this bash launcher. All logic lives in the script — the stub is
# fixed glue.
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cat > "$APP/Contents/Resources/launcher.sh" <<'LAUNCHER'
#!/bin/bash
# Launch pxviewer from the conda environment bundled inside this .app.
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ENV_DIR="$HERE/Resources/env"

# Prefix-embedded files (bin/ shebangs, conda-meta, build configs) were
# rewritten to the build-time staging path by conda-unpack. If the app has been
# moved since, re-prefix them to the current location — scan once, then record
# the location so later launches skip the scan.
MARKER="$ENV_DIR/.pxviewer-prefix"
STORED="$(cat "$MARKER" 2>/dev/null || true)"
if [[ "$STORED" != "$ENV_DIR" ]]; then
  if [[ -n "$STORED" ]]; then
    find "$ENV_DIR" -type f -print0 2>/dev/null \
      | xargs -0 grep -IlF "$STORED" 2>/dev/null \
      | while IFS= read -r f; do
          sed -i '' "s|$STORED|$ENV_DIR|g" "$f"
        done
  else
    "$ENV_DIR/bin/python" "$ENV_DIR/bin/conda-unpack" >/dev/null
  fi
  echo "$ENV_DIR" > "$MARKER"
fi

export PATH="$ENV_DIR/bin:$PATH"
exec "$ENV_DIR/bin/python" -m pxviewer desktop "$@"
LAUNCHER
chmod +x "$APP/Contents/Resources/launcher.sh"

command -v clang >/dev/null 2>&1 || {
  echo "clang not found — install the Xcode command line tools (xcode-select --install)" >&2
  exit 1
}
cat > "$BUILD/launcher.c" <<'EOF'
/* CFBundleExecutable stub: exec the bash launcher beside us in Resources/. */
#include <unistd.h>
#include <libgen.h>
#include <limits.h>
#include <stdio.h>
#include <mach-o/dyld.h>

int main(void) {
    char path[PATH_MAX];
    uint32_t n = sizeof(path);
    if (_NSGetExecutablePath(path, &n) != 0) return 1;
    char *macos_dir = dirname(path);
    char launcher[PATH_MAX];
    snprintf(launcher, sizeof(launcher), "%s/../Resources/launcher.sh", macos_dir);
    execv("/bin/bash", (char *[]){"bash", launcher, NULL});
    perror("execv");
    return 1;
}
EOF
clang -O2 -o "$APP/Contents/MacOS/pxviewer" "$BUILD/launcher.c"

# -- Info.plist ---------------------------------------------------------------
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>            <string>pxviewer</string>
  <key>CFBundleDisplayName</key>     <string>pxviewer</string>
  <key>CFBundleIdentifier</key>      <string>io.github.cschlick.pxviewer</string>
  <key>CFBundleVersion</key>         <string>$VERSION</string>
  <key>CFBundleShortVersionString</key> <string>$VERSION</string>
  <key>CFBundleExecutable</key>      <string>pxviewer</string>
  <key>CFBundlePackageType</key>     <string>APPL</string>
  <key>CFBundleIconFile</key>        <string>icon</string>
  <key>LSMinimumSystemVersion</key>  <string>12.0</string>
  <key>NSHighResolutionCapable</key> <true/>
  <key>NSSupportsAutomaticGraphicsSwitching</key> <true/>
</dict>
</plist>
PLIST

# -- icon ---------------------------------------------------------------------
ICONSET="$BUILD/icon.iconset"
rm -rf "$ICONSET"; mkdir -p "$ICONSET"
for s in 16 32 128 256 512; do
  sips -z "$s" "$s" assets/icon.png --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
  sips -z "$((2*s))" "$((2*s))" assets/icon.png \
    --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns -o "$APP/Contents/Resources/icon.icns" "$ICONSET"

# -- ad-hoc sign (arm64 requires *some* signature; Developer ID is the real fix)
codesign --deep --force --sign "${SIGNING_IDENTITY:--}" "$APP"

# -- dmg ----------------------------------------------------------------------
ln -sfn /Applications "$STAGE/Applications"
rm -f "$DMG"
# ULFO = lzfse: several times faster than UDZO/zlib on a 7 GB env at about the
# same ratio; requires macOS 10.11+, and the plist already floors at 12.0.
hdiutil create -volname pxviewer -srcfolder "$STAGE" -ov -format ULFO "$DMG" >/dev/null

echo
echo "==> built: $DMG ($(du -h "$DMG" | cut -f1) dmg)"
echo "    unsigned: first launch needs right-click -> Open, or:"
echo "      xattr -dr com.apple.quarantine /Applications/pxviewer.app"
