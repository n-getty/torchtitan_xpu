#!/bin/bash
# Checkpoint Verification Launch Script

# Load system modules
. /usr/share/lmod/lmod/init/bash
module purge
module load frameworks
module load mpich
module load py-mpi4py

AURORA_INF=/home/ngetty/proj/vllm_gpt-oss/Aurora-Inferencing
FW_SITES=/opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.2.0/lib/python3.10/site-packages
# Only add project paths and FW sites.
export PYTHONPATH=$(pwd):$(pwd)/..:$(pwd)/../..:$AURORA_INF:$FW_SITES:$PYTHONPATH
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export PYTHONUNBUFFERED=1
export ONECCL_BINDINGS_FOR_PYTORCH_ENV_VERBOSE=1
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export TORCH_EXTENSIONS_DIR="./torch_extensions"
mkdir -p $TORCH_EXTENSIONS_DIR

PYTHON_EXE=$(which python3)
echo "Using Python: $PYTHON_EXE"

# Run the verification script
# We run as a single process for simplicity, simulating rank 0
export RANK=0
export WORLD_SIZE=1
$PYTHON_EXE scripts/verify_checkpoint.py
