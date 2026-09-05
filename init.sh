#!/bin/bash

# One-shot environment setup for dnabind. Safe to re-run.
#
#   1. create/update the conda env from environment.yml (no torch inside it)
#   2. install the exact PyTorch build (CUDA-specific, so kept out of the yml)
#   3. install the dnabind package itself in editable mode
#
# Run from the repository root:  bash init.sh   (or: make init)

set -e  # exit on first error

ENV_NAME="dnabind"                 # conda environment name
YML_FILE="environment.yml"         # conda deps (python + scientific stack)
REQUIREMENTS_FILE="requirements.txt"  # optional extra pip deps (may be absent)

echo "--- Starting environment setup for '$ENV_NAME' ---"

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
#    Tested build: PyTorch 2.6.0 + CUDA 12.6. CUDA wheels are hardware-specific,
#    which is exactly why torch is installed here instead of in environment.yml.
#    On a CPU-only machine, comment the CUDA block and uncomment the CPU line.
echo "--- Installing PyTorch (CUDA 12.6 build) ---"
pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126
# CPU-only alternative (works on macOS / Windows / Linux without a GPU):
# pip install torch==2.6.0

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
