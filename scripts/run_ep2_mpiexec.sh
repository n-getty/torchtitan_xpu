#!/bin/bash
# EP2 Benchmark with mpiexec launch following ALCF pattern

# Initialize modules
. /usr/share/lmod/lmod/init/bash
module load py-torch/2.9.0.dev20250804 py-ipex/xpu-main py-mpi4py/4.0.1

# Paths
SPACK_TORCH=/opt/aurora/25.190.0/spack/unified/0.10.1/install/linux-sles15-x86_64/gcc-13.3.0/py-torch-2.9.0.dev20250804-56izcu7/lib/python3.10/site-packages
SPACK_IPEX=/opt/aurora/25.190.0/spack/unified/0.10.1/install/linux-sles15-x86_64/gcc-13.3.0/py-ipex-xpu-main-7u26l7n/lib/python3.10/site-packages
LOCAL_TRITON=/home/ngetty/.local/lib/python3.10/site-packages
SYSTEM_SITES=/usr/lib/python3.10/site-packages
AURORA_INF=/home/ngetty/proj/vllm_gpt-oss/Aurora-Inferencing

# Prioritize SPACK Torch/IPEX and LOCAL Triton
export PYTHONPATH=$(pwd):$AURORA_INF:$SPACK_TORCH:$SPACK_IPEX:$LOCAL_TRITON:$PYTHONPATH:$SYSTEM_SITES

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export PYTHONUNBUFFERED=1

PYTHON_EXE=/opt/aurora/25.190.0/spack/unified/0.10.1/install/linux-sles15-x86_64/gcc-13.3.0/python-venv-1.0-bdxpu45/bin/python3
MPIEXEC=/opt/cray/pals/1.8/bin/mpiexec
BASE_DIR=/home/ngetty/proj/vllm_gpt-oss/Aurora-Inferencing/torchtitan_xpu/torchtitan

# Run with mpiexec: 2 processes, 2 per node (single node)
$MPIEXEC -n 2 -ppn 2 $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
    --job.config_file $BASE_DIR/torchtitan/models/llama4/train_configs/llama4_ep2_triton_xpu.toml \
    --training.steps=5
