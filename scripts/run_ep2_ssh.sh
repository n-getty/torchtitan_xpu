#!/bin/bash
# Initialize modules
. /usr/share/lmod/lmod/init/bash
module load py-torch/2.9.0.dev20250804 py-ipex/xpu-main

# Paths
SPACK_TORCH=/opt/aurora/25.190.0/spack/unified/0.10.1/install/linux-sles15-x86_64/gcc-13.3.0/py-torch-2.9.0.dev20250804-56izcu7/lib/python3.10/site-packages
SPACK_IPEX=/opt/aurora/25.190.0/spack/unified/0.10.1/install/linux-sles15-x86_64/gcc-13.3.0/py-ipex-xpu-main-7u26l7n/lib/python3.10/site-packages
LOCAL_TRITON=/home/ngetty/.local/lib/python3.10/site-packages
SYSTEM_SITES=/usr/lib/python3.10/site-packages
AURORA_INF=/home/ngetty/proj/vllm_gpt-oss/Aurora-Inferencing

# Prioritize SPACK Torch/IPEX and LOCAL Triton
export PYTHONPATH=$(pwd):$AURORA_INF:$SPACK_TORCH:$SPACK_IPEX:$LOCAL_TRITON:$PYTHONPATH:$SYSTEM_SITES

export ZE_FLAT_DEVICE_HIERARCHY=FLAT

echo "Using PYTHONPATH: $PYTHONPATH"

PYTHON_EXE=/opt/aurora/25.190.0/spack/unified/0.10.1/install/linux-sles15-x86_64/gcc-13.3.0/python-venv-1.0-bdxpu45/bin/python3
MPIEXEC=/opt/cray/pals/1.8/bin/mpiexec

BASE_DIR=/home/ngetty/proj/vllm_gpt-oss/Aurora-Inferencing/torchtitan_xpu/torchtitan

MASTER_ADDR=$(hostname)
MASTER_PORT=29517

$MPIEXEC -n 2 -ppn 2 \
  -env MASTER_ADDR $MASTER_ADDR \
  -env MASTER_PORT $MASTER_PORT \
  -env CCL_PROCESS_LAUNCHER=none \
  -env CCL_KVS_MODE=mpi \
  -env FI_PROVIDER=psm3 \
  -env FI_PSM3_HALO=1 \
  -env CCL_BLOCKING_WAIT=1 \
  -env PYTHONPATH "$PYTHONPATH" \
  $PYTHON_EXE -u $BASE_DIR/launch_wrapper.py $BASE_DIR/torchtitan/train.py --job.config_file $BASE_DIR/torchtitan/models/llama4/train_configs/llama4_ep2_triton_xpu.toml --training.steps=5
