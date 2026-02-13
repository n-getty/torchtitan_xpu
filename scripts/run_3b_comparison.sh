#!/bin/bash
# Llama 4 3B Throughput Comparison Launch Script
# Usage: ./run_3b_comparison.sh <config_key>

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
    echo "Usage: $0 <dense|moe_noep|moe_ep12|moe_hsdp>"
    exit 1
fi

case $CONFIG in
    dense)
        echo "Running Llama 4 3B Dense benchmark..."
        $MPIEXEC -n 12 -ppn 12 --envall $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/llama4/train_configs/llama4_3b_dense_xpu.toml
        ;;
    moe_noep)
        echo "Running Llama 4 3B MoE (No EP) benchmark..."
        $MPIEXEC -n 12 -ppn 12 --envall $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/llama4/train_configs/llama4_3b_moe_noep_xpu.toml
        ;;
    moe_ep12)
        echo "Running Llama 4 3B MoE (EP=12) benchmark..."
        $MPIEXEC -n 12 -ppn 12 --envall $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/llama4/train_configs/llama4_3b_moe_ep12_xpu.toml
        ;;
    moe_hsdp)
        echo "Running Llama 4 3B MoE (HSDP EP=6) benchmark..."
        $MPIEXEC -n 12 -ppn 12 --envall $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/llama4/train_configs/llama4_3b_moe_hsdp_ep6_xpu.toml
        ;;
    dense_compile)
        echo "Running Llama 4 3B Dense Compiled..."
        $MPIEXEC -n 12 -ppn 12 --envall $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/llama4/train_configs/llama4_3b_dense_xpu_compile.toml
        ;;
    hsdp_compile)
        echo "Running Llama 4 3B MoE (HSDP EP=6) Compiled..."
        $MPIEXEC -n 12 -ppn 12 --envall $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/llama4/train_configs/llama4_3b_moe_hsdp_ep6_xpu_compile.toml
        ;;
    ep12_compile)
        echo "Running Llama 4 3B MoE (EP=12) Compiled..."
        $MPIEXEC -n 12 -ppn 12 --envall $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/llama4/train_configs/llama4_3b_moe_ep12_xpu_compile.toml
        ;;
    hsdp_compile_bs8)
        echo "Running Llama 4 3B MoE (HSDP EP=6) Compiled (BS=8)..."
        $MPIEXEC -n 12 -ppn 12 --envall $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/llama4/train_configs/llama4_3b_moe_hsdp_ep6_xpu_compile_bs8.toml
        ;;
    dense_compile_bs12)
        echo "Running Llama 4 3B Dense Compiled (BS=12)..."
        $MPIEXEC -n 12 -ppn 12 --envall $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/llama4/train_configs/llama4_3b_dense_xpu_compile_bs12.toml
        ;;
    dense_compile_bs8)
        echo "Running Llama 4 3B Dense Compiled (BS=8)..."
        $MPIEXEC -n 12 -ppn 12 --envall $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/llama4/train_configs/llama4_3b_dense_xpu_compile_bs8.toml
        ;;
    hsdp_compile_3b_total)
        echo "Running Llama 4 3B Total (Iso-Param) MoE (HSDP EP=6) Compiled..."
        echo "Disabled CPU Binding (Stability Issue)"
        $MPIEXEC -n 12 -ppn 12 $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/llama4/train_configs/llama4_3b_total_moe_hsdp_ep6_xpu_compile.toml 2>&1 | tee debug_log.txt
        ;;
    hsdp_correct_binding)
        echo "Running Llama 4 3B Total (Correct Binding) MoE (HSDP EP=6) Compiled..."
        
        # CPU Binding (4 cores per rank)
        # Socket 0: 4-27 (6 ranks)
        # Socket 1: 56-79 (6 ranks)
        export CPU_BIND="verbose,list:4-7:8-11:12-15:16-19:20-23:24-27:56-59:60-63:64-67:68-71:72-75:76-79"
        
        # Thread Affinity (First core of each rank's set)
        export HOROVOD_THREAD_AFFINITY="4,8,12,16,20,24,56,60,64,68,72,76"
        
        # CCL Worker Affinity (Dedicated cores at end of each socket)
        # Socket 0: 42-47
        # Socket 1: 94-99
        export CCL_WORKER_AFFINITY="42,43,44,45,46,47,94,95,96,97,98,99"
        
        echo "Applying Correct CPU Binding & Affinity..."
        $MPIEXEC -n 12 -ppn 12 --cpu-bind=${CPU_BIND} --envall $PYTHON_EXE -u $BASE_DIR/mpi_train_wrapper.py \
            --job.config_file $BASE_DIR/torchtitan/models/llama4/train_configs/llama4_3b_total_moe_hsdp_ep6_xpu_compile.toml 2>&1 | tee debug_log_binding.txt
        ;;
    env_check)
        echo "Running Environment Check (With Binding)..."
        $MPIEXEC -n 12 -ppn 12 --envall $BASE_DIR/../scripts/bind_cpu.sh $PYTHON_EXE check_topo.py
        ;;
    *)
        echo "Unknown config: $CONFIG"
        echo "Available options: dense, moe_noep, moe_ep12, moe_hsdp, dense_compile, hsdp_compile, ep12_compile, hsdp_compile_bs8, dense_compile_bs12, dense_compile_bs8, hsdp_compile_3b_total"
        exit 1
        ;;
esac
