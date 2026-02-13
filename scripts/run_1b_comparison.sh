#!/bin/bash
# Llama 4 1B Throughput Comparison Launch Script
# Usage: ./run_1b_comparison.sh <config_key>

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
MPIEXEC=/opt/cray/pals/1.8/bin/mpiexec
BASE_DIR=/home/ngetty/proj/vllm_gpt-oss/Aurora-Inferencing/torchtitan_xpu/torchtitan

CONFIG=$1
if [ -z "$CONFIG" ]; then
    echo "Usage: $0 {dense|moe_noep|moe_ep12|moe_hsdp_ep6|profile|compile|dense_compile|dense_compile_bs16}"
    exit 1
fi

case $CONFIG in
    dense)
        echo "Running Llama 4 1B Dense benchmark..."
        $MPIEXEC -n 12 -ppn 12 --envall $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/llama4/train_configs/llama4_1b_dense_xpu.toml
        ;;
    moe_noep)
        echo "Running Llama 4 1B MoE (No EP) benchmark..."
        $MPIEXEC -n 12 -ppn 12 --envall $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/llama4/train_configs/llama4_1b_moe_noep_xpu.toml
        ;;
    moe_ep12)
        echo "Running Llama 4 1B MoE (EP=12) benchmark..."
        $MPIEXEC -n 12 -ppn 12 --envall $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/llama4/train_configs/llama4_1b_moe_ep12_xpu.toml
        ;;
    moe_hsdp)
        echo "Running Llama 4 1B MoE (HSDP EP=6) benchmark..."
        $MPIEXEC -n 12 -ppn 12 --envall $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/llama4/train_configs/llama4_1b_moe_hsdp_ep6_xpu.toml
        ;;
    profile)
        echo "Running Llama 4 1B MoE (HSDP EP=6) Profiling..."
        $MPIEXEC -n 12 -ppn 12 --envall $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/llama4/train_configs/llama4_1b_moe_hsdp_ep6_xpu_profile.toml
        ;;
    compile)
        echo "Running Llama 4 1B MoE (HSDP EP=6) Compiled..."
        $MPIEXEC -n 12 -ppn 12 --envall $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/llama4/train_configs/llama4_1b_moe_hsdp_ep6_xpu_compile.toml
        ;;
    dense_compile)
        echo "Running Llama 4 1B Dense Compiled..."
        $MPIEXEC -n 12 -ppn 12 --envall $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/llama4/train_configs/llama4_1b_dense_xpu_compile.toml
        ;;
    dense_compile_bs16)
        echo "Running Llama 4 1B Dense Compiled (BS=16)..."
        $MPIEXEC -n 12 -ppn 12 --envall $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/llama4/train_configs/llama4_1b_dense_xpu_compile_bs16.toml
        ;;
    *)
        echo "Unknown config: $CONFIG"
        echo "Available options: dense, moe_noep, moe_ep12, moe_hsdp, profile, compile"
        exit 1
        ;;
esac
