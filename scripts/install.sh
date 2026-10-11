#!/usr/bin/env bash
# One-shot pxviewer development setup for macOS and Linux — safe to run from a
# checkout or straight from curl:
#
#   curl -fsSL https://raw.githubusercontent.com/cschlick/pxviewer/main/scripts/install.sh | bash
#   ./scripts/install.sh            asks before installing git/Miniforge
#   ./scripts/install.sh --yes      no prompts (e.g. a fresh box over ssh)
#
# What it does, in order:
#
#   git?      — install Xcode CLT (macOS) or the system package (Linux) if missing
#   conda?    — install Miniforge into ~/miniforge3 if no conda is found
#   checkout  — clone https://github.com/cschlick/pxviewer if not already in one
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
for arg in "$@"; do
  case "$arg" in
    -y|--yes) yes=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

uname_s="$(uname -s)"
case "$uname_s" in
  Darwin|Linux) ;;
  *) echo "this script supports macOS and Linux; on Windows use Git Bash" >&2
     echo "(see the README's Windows section)." >&2; exit 1 ;;
esac

confirm() { # $1 = prompt; returns 0 for yes
  [[ "$yes" == 1 ]] && return 0
  read -r -p "$1 [y/N] " reply
  [[ "$reply" =~ ^[Yy] ]]
}

# --- git ---------------------------------------------------------------------
# Needed to clone when the script runs outside a checkout. macOS gets git from
# the Xcode Command Line Tools — xcode-select pops a dialog the user must
# accept, so we trigger it and ask for a re-run rather than wait.
if ! command -v git >/dev/null 2>&1; then
  if [[ "$uname_s" == "Darwin" ]]; then
    echo "git not found — triggering the Xcode Command Line Tools installer."
    xcode-select --install 2>/dev/null || true
    echo "finish the CLT install, then re-run this script." >&2
    exit 1
  fi
  if confirm "git not found — install it with the system package manager?"; then
    if command -v apt-get >/dev/null 2>&1; then
      sudo apt-get update && sudo apt-get install -y git
    elif command -v dnf >/dev/null 2>&1; then
      sudo dnf install -y git
    elif command -v pacman >/dev/null 2>&1; then
      sudo pacman -S --noconfirm git
    else
      echo "no apt/dnf/pacman found — install git by hand and re-run." >&2
      exit 1
    fi
  else
    echo "git is required to fetch the pxviewer checkout." >&2
    exit 1
  fi
fi

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

# --- checkout ------------------------------------------------------------------
# Running inside a pxviewer checkout (the environment.yml + python/pxviewer
# signature) uses it in place; piped-from-curl runs clone beside the cwd.
if [[ -f environment.yml && -d python/pxviewer ]]; then
  :
elif [[ -f pxviewer/environment.yml && -d pxviewer/python/pxviewer ]]; then
  cd pxviewer
else
  if ! confirm "no pxviewer checkout here — clone https://github.com/cschlick/pxviewer into ./pxviewer?"; then
    echo "a pxviewer checkout is required." >&2
    exit 1
  fi
  git clone https://github.com/cschlick/pxviewer
  cd pxviewer
fi
echo "checkout: $(pwd)"

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
