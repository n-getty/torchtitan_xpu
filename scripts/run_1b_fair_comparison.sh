#!/bin/bash
# Fair 1B Benchmark Comparison: Dense vs MoE EP
# Run on Aurora compute node with 12 XPU tiles

set -e

# Environment setup
module load frameworks 2>/dev/null || true
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_WORKER_COUNT=4
export PYTHONPATH=/lus/flare/projects/AuroraGPT/ngetty/torchtitan_xpu:/lus/flare/projects/AuroraGPT/ngetty/torchtitan_xpu/torchtitan:$PYTHONPATH
export MASTER_PORT=29510
export MASTER_ADDR=localhost

cd /lus/flare/projects/AuroraGPT/ngetty/torchtitan_xpu

echo "========================================"
echo "1B Model Benchmark Comparison"
echo "========================================"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "========================================"
echo ""

run_benchmark() {
    local name=$1
    local config=$2
    
    echo ""
    echo "========================================"
    echo "Running: $name"
    echo "Config: $config"
    echo "========================================"
    echo ""
    
    mpiexec -n 12 -ppn 12 --envall python -u torchtitan/mpi_train_wrapper.py \
        --job.config_file "$config" \
        2>&1 | tee "outputs_${name}.log"
    
    echo ""
    echo "Completed: $name"
    echo "========================================"
}

# Benchmark configurations to run
CONFIG_DIR="torchtitan/torchtitan/models/llama4/train_configs"

echo "Available configurations:"
ls -la $CONFIG_DIR/llama4_1b*.toml 2>/dev/null | head -10
echo ""

# Run benchmarks in order
echo "Starting benchmarks..."
echo ""

# 1. Dense 1B (compiled, selective AC)
run_benchmark "1b_dense_compile" "$CONFIG_DIR/llama4_1b_dense_xpu_compile.toml"

# 2. MoE 1B EP=12 (compiled, selective AC) - uses native EP by default now
run_benchmark "1b_moe_ep12_compile" "$CONFIG_DIR/llama4_1b_moe_ep12_xpu_compile.toml"

echo ""
echo "========================================"
echo "All benchmarks completed!"
echo "========================================"
echo ""
echo "Summary of results:"
echo ""
for log in outputs_1b_*.log; do
    if [[ -f "$log" ]]; then
        echo "=== $log ==="
        grep -E "(step:|MFU|tps:|memory:|TFLOPS)" "$log" | tail -5
        echo ""
    fi
done
