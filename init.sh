#!/bin/bash

# One-shot environment setup for dnabind. Safe to re-run.
#
#   1. create/update the conda env from the platform's environment file
#      (no torch inside it)
#   2. install the platform-appropriate PyTorch build (kept out of the yml)
#   3. install the dnabind package itself in editable mode
#
# The platform (linux/macos) is auto-detected from `uname`. Override it by
# passing an arg or setting DNABIND_PLATFORM, e.g. for a CPU-only Linux box:
#
#   bash init.sh linux            (or: DNABIND_PLATFORM=linux bash init.sh)
#
# Run from the repository root:  bash init.sh   (or: make init)

set -e  # exit on first error

ENV_NAME="dnabind"                 # conda environment name
REQUIREMENTS_FILE="requirements.txt"  # optional extra pip deps (may be absent)

# Pick the platform: explicit arg > env var > auto-detect via uname.
# This selects both the conda yml and the PyTorch build (macOS has no CUDA).
PLATFORM="${1:-${DNABIND_PLATFORM:-}}"
if [ -z "$PLATFORM" ]; then
    case "$(uname -s)" in
        Darwin) PLATFORM="macos" ;;
        Linux)  PLATFORM="linux" ;;
        *) echo "ERROR: unsupported OS '$(uname -s)'. Pass 'linux' or 'macos' explicitly."; exit 1 ;;
    esac
fi

case "$PLATFORM" in
    linux|macos) ;;
    *) echo "ERROR: unknown platform '$PLATFORM'. Use 'linux' or 'macos'."; exit 1 ;;
esac

YML_FILE="environment.${PLATFORM}.yml"  # conda deps (python + scientific stack)

echo "--- Starting environment setup for '$ENV_NAME' (platform: $PLATFORM) ---"

# 1. Check that Conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: Conda not found. Install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# 2. Create the env, or update it if it already exists (makes re-runs safe)
if [ ! -f "$YML_FILE" ]; then
    echo "ERROR: '$YML_FILE' not found. Run this from the repository root."
    exit 1
fi
conda env create -f "$YML_FILE" --name "$ENV_NAME" || \
conda env update -f "$YML_FILE" --name "$ENV_NAME"

# 3. Activate the env for the rest of this script
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# 4. Install PyTorch.
#    Tested build: PyTorch 2.6.0. CUDA wheels are hardware-specific, which is why
#    torch is installed here instead of in the yml. On Linux we pull the CUDA 12.6
#    build; on macOS we use the default wheel (CPU/MPS — there is no CUDA on Mac).
if [ "$PLATFORM" = "macos" ]; then
    echo "--- Installing PyTorch (macOS: CPU/MPS build) ---"
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
else
    echo "--- Installing PyTorch (CUDA 12.6 build) ---"
    pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126 \
        --index-url https://download.pytorch.org/whl/cu126
fi

# 5. Install any extra pip packages (optional; skipped if the file is absent)
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "--- Installing extra pip packages from $REQUIREMENTS_FILE ---"
    pip install -r "$REQUIREMENTS_FILE"
fi

# 6. Install the dnabind package itself (torch is already satisfied above)
echo "--- Installing dnabind (editable) ---"
pip install -e .

echo "--- Setup complete ---"
echo "Activate the environment in future sessions with:"
echo "   conda activate $ENV_NAME"
