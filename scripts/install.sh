#!/usr/bin/env bash
# One-shot pxviewer development setup for macOS and Linux — safe to run from a
# checkout or straight from curl:
#
#   curl -fsSL https://raw.githubusercontent.com/cschlick/pxviewer/main/scripts/install.sh | bash
#   ./scripts/install.sh            asks before installing git/Miniforge
#   ./scripts/install.sh --yes      no prompts (e.g. a fresh box over ssh)
#   ./scripts/install.sh --dev      track main instead of the latest release
#   ./scripts/install.sh --ref v0.1.0   pin an explicit tag or branch
#
# What it does, in order:
#
#   conda?    — install Miniforge into ~/miniforge3 if no conda is found
#   git?      — `conda install git` into base if missing; used by full path so
#               it works even with auto_activate_base: false; no sudo or Xcode
#               CLT needed
#   checkout  — clone https://github.com/cschlick/pxviewer if not already in one
#               (newest v* tag by default; --dev/--ref change that)
#   conda env create -f environment.yml   (or update, if the env exists)
#   pip install -e ./python --no-deps
#   ./scripts/setup_chem_data.sh          validation caches; Windows viewport fix
#   (cd frontend && npm ci)               vendor molstar/react/esbuild
#   ./scripts/build_frontend.sh           -> frontend/build/index.js
#
# Afterwards:
#
#   conda activate pxviewer
#   python -m pxviewer desktop
set -euo pipefail

yes=0
dev=0
ref=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--yes) yes=1 ;;
    --dev) dev=1 ;;
    --ref) ref="${2:?--ref needs a value}"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

uname_s="$(uname -s)"
case "$uname_s" in
  Darwin|Linux) ;;
  *) echo "this script supports macOS and Linux; on Windows use Git Bash" >&2
     echo "(see the README's Windows section)." >&2; exit 1 ;;
esac

confirm() { # $1 = prompt; returns 0 for yes
  [[ "$yes" == 1 ]] && return 0
  local reply
  # `curl ... | bash` runs with the script itself on stdin — a bare `read`
  # would consume upcoming script lines (or EOF) as the "answer", and every
  # prompt silently resolves to no. Ask the terminal on its own fd instead.
  # A /dev/tty node exists everywhere but refuses to open without a real
  # controlling terminal (CI, cron, containers) — and there the one-line
  # install is itself the consent, so proceed.
  if { exec 9<>/dev/tty; } 2>/dev/null; then
    read -r -p "$1 [y/N] " reply <&9
    exec 9>&-
    [[ "$reply" =~ ^[Yy] ]]
  else
    echo "$1 — yes (no terminal to ask)"
    return 0
  fi
}

# --- conda -------------------------------------------------------------------
# An existing conda is used as-is; nothing is installed next to it. Miniforge is
# the fallback because it defaults to conda-forge, which is where every pxviewer
# dependency lives — a defaults-first distribution (Anaconda's Miniconda) would
# solve the same env but pulls the channels explicitly anyway.
conda_was_on_path=1
if ! command -v conda >/dev/null 2>&1; then
  conda_was_on_path=0
  for prefix in "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/miniconda3"; do
    if [[ -x "$prefix/bin/conda" ]]; then
      eval "$("$prefix/bin/conda" shell.bash hook)"
      break
    fi
  done
fi

installed_fresh=0
if ! command -v conda >/dev/null 2>&1; then
  if ! confirm "conda not found — install Miniforge into \$HOME/miniforge3?"; then
    echo "conda is required: https://github.com/conda-forge/miniforge" >&2
    exit 1
  fi
  installer="Miniforge3-${uname_s}-$(uname -m).sh"
  url="https://github.com/conda-forge/miniforge/releases/latest/download/${installer}"
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  echo "downloading $url"
  if command -v curl >/dev/null 2>&1; then
    curl -fL "$url" -o "$tmp/$installer"
  elif command -v wget >/dev/null 2>&1; then
    wget -q "$url" -O "$tmp/$installer"
  else
    echo "neither curl nor wget found — install one, or install Miniforge by hand" >&2
    exit 1
  fi
  bash "$tmp/$installer" -b -p "$HOME/miniforge3"
  eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
  # Batch install skips the interactive "conda init?" step — do it here so new
  # shells get conda without a manual follow-up (and this one is covered by the
  # hook eval above).
  "$HOME/miniforge3/bin/conda" init "$(basename "${SHELL:-bash}")"
  installed_fresh=1
  echo "installed Miniforge to $HOME/miniforge3"
fi

# `conda activate` is a shell function, not a binary call: load the hook in this
# script even when conda was already on PATH, or the activate below fails.
eval "$(conda shell.bash hook)"
echo "using: $(command -v conda)  ($(conda --version))"

# --- git ---------------------------------------------------------------------
# Only needed to clone. conda-forge's git works on every platform, needs no
# sudo, and skips macOS's Xcode CLT dialog entirely — so it is the bootstrap
# rather than the system package manager.
GIT=git
if ! command -v git >/dev/null 2>&1; then
  GIT="$(conda info --base)/bin/git"
  if [[ ! -x "$GIT" ]]; then
    if ! confirm "git not found — install it into the base conda env?"; then
      echo "git is required to fetch the pxviewer checkout." >&2
      exit 1
    fi
    conda install -y -n base -c conda-forge git
    if [[ ! -x "$GIT" ]]; then
      echo "installed git into base but $GIT is missing" >&2
      exit 1
    fi
  fi
  # Used by full path from here on: whether base activates onto PATH or not
  # (auto_activate_base: false) cannot matter, and an existing base git is
  # never touched.
fi

# --- checkout ------------------------------------------------------------------
# Running inside a pxviewer checkout (the environment.yml + python/pxviewer
# signature) uses it in place; piped-from-curl runs clone beside the cwd.
in_checkout=0
if [[ -f environment.yml && -d python/pxviewer ]]; then
  in_checkout=1
elif [[ -f pxviewer/environment.yml && -d pxviewer/python/pxviewer ]]; then
  cd pxviewer
  in_checkout=1
fi

if [[ "$in_checkout" == 0 ]]; then
  # Ref precedence: --ref wins, --dev tracks main, and the default is the
  # newest v* tag so `curl | bash` installs a release, not HEAD.
  target="$ref"
  if [[ -z "$target" && "$dev" == 0 ]]; then
    target="$("$GIT" ls-remote --tags https://github.com/cschlick/pxviewer 'v*' \
      | sed -n 's|.*refs/tags/\([^^]*\)|\1|p' | sort -V | tail -1)"
    [[ -n "$target" ]] && echo "latest release: $target"
  fi
  [[ -z "$target" ]] && target=main
  if ! confirm "no pxviewer checkout here — clone $target into ./pxviewer?"; then
    echo "a pxviewer checkout is required." >&2
    exit 1
  fi
  # A tag only needs the one commit; dev gets a full clone so branches work.
  if [[ "$dev" == 1 && -z "$ref" ]]; then
    "$GIT" clone https://github.com/cschlick/pxviewer
  else
    "$GIT" clone --depth 1 --branch "$target" https://github.com/cschlick/pxviewer
  fi
  cd pxviewer
elif [[ -n "$ref" ]]; then
  # An existing checkout is used as-is unless --ref was given explicitly.
  "$GIT" fetch --tags --quiet
  "$GIT" checkout "$ref"
fi

echo "checkout: $(pwd)  $("$GIT" describe --tags --always 2>/dev/null || echo unknown)"

# --- environment -------------------------------------------------------------
# The env name comes from environment.yml itself; an existing env is updated
# rather than recreated so a re-run is a refresh, not a rebuild.
if conda env list | grep -qE '^pxviewer[[:space:]]'; then
  echo "env 'pxviewer' exists — updating"
  conda env update -f environment.yml --prune
else
  conda env create -f environment.yml
fi
conda activate pxviewer

# Headless rendering: on a display-less box pxviewer hosts the Mol* page in
# Chromium itself — no Qt, no X11/Xvfb (the GUI-agnostic core path; see
# pxviewer.appcore/pxviewer.browser). playwright (in environment.yml) is the
# driver; this fetches the browser binary, which is not a package. The
# headless shell alone is enough — it renders WebGL via SwANGLE.
if [[ "$uname_s" == "Linux" ]]; then
  python -m playwright install chromium --only-shell \
    || echo "warning: chromium headless shell did not install — headless mode will have no viewport (load/select still work)" >&2

  # The shell is a foreign binary — no rpath into the env — so the shared
  # libraries playwright expects from the OS (libnspr4, libnss3, …) must come
  # from conda and be found via LD_LIBRARY_PATH (pxviewer.browser sets it at
  # launch). Ask ldd what is actually missing and conda-install the matching
  # packages rather than guessing a list a distro update can shift.
  shell_bin=$(find "$(python -c 'import os; print(os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or os.path.expanduser("~/.cache/ms-playwright"))')" \
                  -name "chrome-headless-shell" -type f 2>/dev/null | head -1)
  if [[ -n "$shell_bin" ]]; then
    declare -A conda_for_lib=(
      [libnspr4.so]=nspr [libplc4.so]=nspr [libplds4.so]=nspr
      [libnss3.so]=nss [libnssutil3.so]=nss [libsmime3.so]=nss
      [libatk-1.0.so.0]=atk-1.0 [libatk-bridge-2.0.so.0]=at-spi2-atk
      [libatspi.so.0]=at-spi2-core
      [libcairo.so.2]=cairo [libpango-1.0.so.0]=pango
      [libpangocairo-1.0.so.0]=pango [libpangoft2-1.0.so.0]=pango
      [libcups.so.2]=cups [libdbus-1.so.3]=dbus [libdrm.so.2]=libdrm
      [libgbm.so.1]=libgbm
      [libglib-2.0.so.0]=glib [libgobject-2.0.so.0]=glib [libgio-2.0.so.0]=glib
      [libwayland-client.so.0]=wayland
      [libX11.so.6]=xorg-libx11 [libX11-xcb.so.1]=xorg-libx11
      [libxcb.so.1]=xorg-libxcb
      [libXcomposite.so.1]=xorg-libxcomposite [libXdamage.so.1]=xorg-libxdamage
      [libXext.so.6]=xorg-libxext [libXfixes.so.3]=xorg-libxfixes
      [libxkbcommon.so.0]=libxkbcommon [libXrandr.so.2]=xorg-libxrandr
      [libXrender.so.1]=xorg-libxrender [libXi.so.6]=xorg-libxi
      [libXtst.so.6]=xorg-libxtst [libXcursor.so.1]=xorg-libxcursor
      [libasound.so.2]=alsa-lib [libexpat.so.1]=expat
      [libfontconfig.so.1]=fontconfig [libfreetype.so.6]=freetype
    )
    missing=$(ldd "$shell_bin" 2>/dev/null | awk '/not found/ {print $1}' | sort -u)
    pkgs=()
    unmapped=()
    while read -r so; do
      [[ -z "$so" ]] && continue
      base="${so%%.*}"          # libnspr4.so -> libnspr4 (versions like .so.0 too)
      pkg="${conda_for_lib[$so]:-${conda_for_lib[$base]:-}}"
      if [[ -n "$pkg" ]]; then pkgs+=("$pkg"); else unmapped+=("$so"); fi
    done <<< "$missing"
    if [[ ${#pkgs[@]} -gt 0 ]]; then
      # shellcheck disable=SC2068 # word-splitting is the point: one arg per package
      mapfile -t pkgs < <(printf '%s\n' ${pkgs[@]} | sort -u)
      conda install -y -n pxviewer -c conda-forge ${pkgs[@]} \
        || echo "warning: browser shared libs did not install — headless screenshots may fail" >&2
    fi
    if [[ ${#unmapped[@]} -gt 0 ]]; then
      echo "warning: no conda mapping for missing libs: ${unmapped[*]} — add them to install.sh" >&2
    fi
    # Verify: anything still unresolved is a real failure the user should see.
    still=$(ldd "$shell_bin" 2>/dev/null | awk '/not found/ {print $1}' | sort -u | tr '\n' ' ')
    [[ -z "$still" ]] || echo "warning: chromium still missing: $still" >&2
  fi
fi

# --- pxviewer ----------------------------------------------------------------
# --no-deps keeps pip from pulling PyPI wheels over conda-managed packages —
# most damagingly numpy, which cctbx is compiled against.
pip install -e ./python --no-deps
./scripts/setup_chem_data.sh
( cd frontend && npm ci )
./scripts/build_frontend.sh

# A script cannot activate the caller's shell. Fresh installs got conda init
# (future shells just work); prefix-found or on-path condas are assumed set up.
# Either way, THIS shell needs the hook sourced once if conda wasn't on PATH.
if [[ "$installed_fresh" == 1 ]]; then
  cat <<EOF

done. 'conda init' ran, so every NEW shell has conda. For THIS shell, once:

  eval "\$("$HOME/miniforge3/bin/conda" shell.bash hook)"

then:

  conda activate pxviewer
  python -m pxviewer desktop            # the app
  libtbx.python -m pxviewer.run_tests   # the headless suite

EOF
elif [[ "$conda_was_on_path" == 0 ]]; then
  cat <<EOF

done. conda was found at a prefix but is not on this shell's PATH. For THIS
shell, once:

  eval "\$($(command -v conda) shell.bash hook)"

(or permanently: '$(command -v conda) init bash', then open a new shell)

then:

  conda activate pxviewer
  python -m pxviewer desktop            # the app
  libtbx.python -m pxviewer.run_tests   # the headless suite

EOF
else
  cat <<'EOF'

done. To run:

  conda activate pxviewer
  python -m pxviewer desktop            # the app
  libtbx.python -m pxviewer.run_tests   # the headless suite

EOF
fi
