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
export MASTER_ADDR=localhost
export MASTER_PORT=29525

# oneCCL settings - use pmi mode for torchrun
export CCL_PROCESS_LAUNCHER=none
export CCL_KVS_MODE=pmi

# Python settings
export PYTHONUNBUFFERED=1
PYTHON_EXE=/opt/aurora/25.190.0/spack/unified/0.10.1/install/linux-sles15-x86_64/gcc-13.3.0/python-venv-1.0-bdxpu45/bin/python3
BASE_DIR=/home/ngetty/proj/vllm_gpt-oss/Aurora-Inferencing/torchtitan_xpu/torchtitan

$PYTHON_EXE -m torch.distributed.run \
    --nproc_per_node=2 \
    --nnodes=1 \
    --rdzv_id=101 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    $BASE_DIR/torchtitan/train.py --job.config_file $BASE_DIR/torchtitan/models/llama4/train_configs/llama4_ep2_triton_xpu.toml --training.steps=5
