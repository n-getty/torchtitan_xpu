#!/bin/bash
# MoE 1B EP=12 Benchmark (Compiled, No AC)
# NOTE: AC is disabled for EP due to memory leak from all_to_all autograd
set -e

cd /lus/flare/projects/AuroraGPT/ngetty/torchtitan_xpu
module load frameworks 2>/dev/null || true

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_WORKER_COUNT=4
# Include torchtitan_xpu in PYTHONPATH
export PYTHONPATH=/lus/flare/projects/AuroraGPT/ngetty/torchtitan_xpu:/lus/flare/projects/AuroraGPT/ngetty/torchtitan_xpu/torchtitan:$PYTHONPATH
export MASTER_PORT=29510
export MASTER_ADDR=localhost

echo "=== MoE 1B EP=12 Benchmark (Compiled, No AC) ==="
echo "NOTE: AC disabled to avoid memory leak with EP"
echo "Starting at: $(date)"
echo ""

mpiexec -n 12 -ppn 12 --envall python -u torchtitan/mpi_train_wrapper.py \
    --job.config_file torchtitan/torchtitan/models/llama4/train_configs/llama4_1b_moe_ep12_xpu_compile.toml

echo ""
echo "Completed at: $(date)"
