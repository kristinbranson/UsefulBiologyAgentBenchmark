#!/bin/bash
# playfmf.sh
#
# Launch the playfmf.py video player (UFMF, FMF, SBFMF, AVI) in the conda
# environment that has its dependencies. All arguments are passed through to
# playfmf.py, e.g.
#   playfmf.sh /path/to/movie.ufmf
#   playfmf.sh --zero-based /path/to/movie.ufmf
#   playfmf.sh --help
#
# The script can be symlinked into a directory on PATH (e.g. ~/bin); it
# follows the symlink to find playfmf.py.
#
# Optional environment variables:
#   PLAYFMF_CONDA_ROOT  conda installation to use (default: ~/miniforge3)
#   PLAYFMF_CONDA_ENV   conda environment to activate (default: scibench)

set -euo pipefail

conda_root="${PLAYFMF_CONDA_ROOT:-$HOME/miniforge3}"
conda_env="${PLAYFMF_CONDA_ENV:-scibench}"

# Directory containing the real playfmf.sh (and playfmf.py), after resolving symlinks.
script_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

# conda's activation scripts read variables that may be unset, so allow that
# while activating.
set +u
source "$conda_root/etc/profile.d/conda.sh"
conda activate "$conda_env"
set -u

exec python "$script_dir/playfmf.py" "$@"
