#!/bin/bash
# DeepSeek-V3 1B Benchmark Launch Script
# Usage: ./run_deepseek_1b.sh <moe_noep|ep12>

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
    echo "Usage: $0 <moe_noep|ep12>"
    exit 1
fi

case $CONFIG in
    moe_noep)
        echo "Running DeepSeek-V3 1B MoE (No EP) benchmark..."
        $MPIEXEC -n 12 -ppn 12 --envall $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/deepseek_v3/train_configs/deepseek_1b_moe_noep_xpu.toml
        ;;
    bigbatch)
        echo "Running DeepSeek-V3 1B MoE (No EP) BigBatch benchmark..."
        $MPIEXEC -n 12 -ppn 12 --envall $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/deepseek_v3/train_configs/deepseek_1b_moe_bigbatch_xpu.toml
        ;;
    ep12)
        echo "Running DeepSeek-V3 1B EP=12 benchmark..."
        $MPIEXEC -n 12 -ppn 12 --envall $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/deepseek_v3/train_configs/deepseek_1b_ep12_xpu.toml
        ;;
    ep12_bigbatch)
        echo "Running DeepSeek-V3 1B EP=12 BigBatch benchmark..."
        $MPIEXEC -n 12 -ppn 12 --envall $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/deepseek_v3/train_configs/deepseek_1b_ep12_bigbatch_xpu.toml
        ;;
    ep12_ddp)
        echo "Running DeepSeek-V3 1B EP=12 DDP benchmark..."
        $MPIEXEC -n 12 -ppn 12 --envall $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/deepseek_v3/train_configs/deepseek_1b_ep12_ddp_xpu.toml
        ;;
    16b_noep)
        echo "Running DeepSeek-V3 16B MoE (No EP) benchmark..."
        $MPIEXEC -n 12 -ppn 12 --envall $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/deepseek_v3/train_configs/deepseek_16b_moe_noep_xpu.toml
        ;;
    16b_ep12)
        echo "Running DeepSeek-V3 16B EP=12 benchmark..."
        $MPIEXEC -n 12 -ppn 12 --envall $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/deepseek_v3/train_configs/deepseek_16b_ep12_xpu.toml
        ;;
    llama4_noep)
        echo "Running Llama4 Large MoE (No EP) benchmark..."
        $MPIEXEC -n 12 -ppn 12 --envall $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/llama4/train_configs/llama4_large_moe_noep_xpu.toml
        ;;
    llama4_ep12)
        echo "Running Llama4 Large EP=12 benchmark..."
        $MPIEXEC -n 12 -ppn 12 --envall $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/llama4/train_configs/llama4_large_ep12_xpu.toml
        ;;
    llama4_ep12_bigbatch)
        echo "Running Llama4 Large EP=12 Big Batch benchmark..."
        $MPIEXEC -n 12 -ppn 12 --envall $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/llama4/train_configs/llama4_large_ep12_bigbatch_xpu.toml
        ;;
    llama4_ep12_maxbatch)
        echo "Running Llama4 Large EP=12 Max Batch (Batch=8) benchmark..."
        $MPIEXEC -n 12 -ppn 12 --envall $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/llama4/train_configs/llama4_large_ep12_maxbatch_xpu.toml
        ;;
    llama4_dense_equivalent)
        echo "Running Llama4 Large Dense Equivalent benchmark..."
        $MPIEXEC -n 12 -ppn 12 --envall $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/llama4/train_configs/llama4_large_dense_equivalent_xpu.toml
        ;;
    *)
        echo "Unknown config: $CONFIG"
        exit 1
        ;;
esac
