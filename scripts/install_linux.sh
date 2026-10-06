#!/usr/bin/env bash
# One-shot Linux setup: install conda (Miniforge) if missing, then build the
# pxviewer development environment from this checkout — the same steps the
# README lists, run in order:
#
#   conda env create -f environment.yml   (or update, if the env exists)
#   pip install -e ./python --no-deps
#   ./scripts/setup_chem_data.sh          validation caches
#   (cd frontend && npm ci)               vendor molstar/react/esbuild
#   ./scripts/build_frontend.sh           -> frontend/build/index.js
#
#   ./scripts/install_linux.sh            asks before installing Miniforge
#   ./scripts/install_linux.sh --yes      no prompts (e.g. a fresh box over ssh)
#
# Afterwards:
#
#   conda activate pxviewer
#   python -m pxviewer desktop
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

yes=0
for arg in "$@"; do
  case "$arg" in
    -y|--yes) yes=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "this script is for Linux; on macOS follow the README steps directly." >&2
  exit 1
fi

confirm() { # $1 = prompt; returns 0 for yes
  [[ "$yes" == 1 ]] && return 0
  read -r -p "$1 [y/N] " reply
  [[ "$reply" =~ ^[Yy] ]]
}

# --- conda -------------------------------------------------------------------
# An existing conda is used as-is; nothing is installed next to it. Miniforge is
# the fallback because it defaults to conda-forge, which is where every pxviewer
# dependency lives — a defaults-first distribution (Anaconda's Miniconda) would
# solve the same env but pulls the channels explicitly anyway.
if ! command -v conda >/dev/null 2>&1; then
  for prefix in "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/miniconda3"; do
    if [[ -x "$prefix/bin/conda" ]]; then
      eval "$("$prefix/bin/conda" shell.bash hook)"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  if ! confirm "conda not found — install Miniforge into \$HOME/miniforge3?"; then
    echo "conda is required: https://github.com/conda-forge/miniforge" >&2
    exit 1
  fi
  arch="$(uname -m)"
  installer="Miniforge3-Linux-${arch}.sh"
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
  echo "installed Miniforge to $HOME/miniforge3"
  echo "(new shells need: source \$HOME/miniforge3/etc/profile.d/conda.sh"
  echo " or run: $HOME/miniforge3/bin/conda init bash)"
fi

# `conda activate` is a shell function, not a binary call: load the hook in this
# script even when conda was already on PATH, or the activate below fails.
eval "$(conda shell.bash hook)"
echo "using: $(command -v conda)  ($(conda --version))"

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

# --- pxviewer ----------------------------------------------------------------
# --no-deps keeps pip from pulling PyPI wheels over conda-managed packages —
# most damagingly numpy, which cctbx is compiled against.
pip install -e ./python --no-deps
./scripts/setup_chem_data.sh
( cd frontend && npm ci )
./scripts/build_frontend.sh

cat <<'EOF'

done. To run:

  conda activate pxviewer
  python -m pxviewer desktop          # the app
  libtbx.python -m pxviewer.run_tests # the headless suite

EOF
