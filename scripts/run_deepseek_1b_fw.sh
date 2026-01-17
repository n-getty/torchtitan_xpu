#!/bin/bash
# DeepSeek-V3 1B Benchmark Launch Script
# Usage: ./run_deepseek_1b.sh <moe_noep|ep12>

. /usr/share/lmod/lmod/init/bash
module load frameworks

SPACK_TORCH=/opt/aurora/25.190.0/spack/unified/0.10.1/install/linux-sles15-x86_64/gcc-13.3.0/py-torch-2.9.0.dev20250804-56izcu7/lib/python3.10/site-packages
SPACK_IPEX=/opt/aurora/25.190.0/spack/unified/0.10.1/install/linux-sles15-x86_64/gcc-13.3.0/py-ipex-xpu-main-7u26l7n/lib/python3.10/site-packages
LOCAL_TRITON=/home/ngetty/.local/lib/python3.10/site-packages
SYSTEM_SITES=/usr/lib/python3.10/site-packages
AURORA_INF=/home/ngetty/proj/vllm_gpt-oss/Aurora-Inferencing

export PYTHONPATH=$(pwd):$(pwd)/..:$(pwd)/../..:$AURORA_INF:$SPACK_TORCH:$SPACK_IPEX:$LOCAL_TRITON:$PYTHONPATH:$SYSTEM_SITES
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export PYTHONUNBUFFERED=1
export ONECCL_BINDINGS_FOR_PYTORCH_ENV_VERBOSE=1
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export TORCH_EXTENSIONS_DIR="./torch_extensions"
mkdir -p $TORCH_EXTENSIONS_DIR

PYTHON_EXE=/opt/aurora/25.190.0/spack/unified/0.10.1/install/linux-sles15-x86_64/gcc-13.3.0/python-venv-1.0-bdxpu45/bin/python3
MPIEXEC=/opt/cray/pals/1.8/bin/mpiexec
BASE_DIR=/home/ngetty/proj/vllm_gpt-oss/Aurora-Inferencing/torchtitan_xpu/torchtitan

CONFIG=$1
if [ -z "$CONFIG" ]; then
    echo "Usage: $0 <moe_noep|ep12>"
    exit 1
fi

case $CONFIG in
    moe_noep)
        echo "Running DeepSeek-V3 1B MoE (No EP) benchmark..."
        $MPIEXEC -n 12 -ppn 12 $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/deepseek_v3/train_configs/deepseek_1b_moe_noep_xpu.toml
        ;;
    ep12)
        echo "Running DeepSeek-V3 1B EP=12 benchmark..."
        $MPIEXEC -n 12 -ppn 12 $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/deepseek_v3/train_configs/deepseek_1b_ep12_xpu.toml
        ;;
    ep12_ddp)
        echo "Running DeepSeek-V3 1B EP=12 DDP benchmark..."
        $MPIEXEC -n 12 -ppn 12 $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/deepseek_v3/train_configs/deepseek_1b_ep12_ddp_xpu.toml
        ;;
    *)
        echo "Unknown config: $CONFIG"
        exit 1
        ;;
esac
