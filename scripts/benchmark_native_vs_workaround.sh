#!/bin/bash
# Benchmark: Native XCCL Ops vs XPU Workarounds
#
# Compares MFU and throughput between:
# 1. Baseline: XPUExpertParallel (all_gather-based) + Triton MoE disabled
# 2. Native EP: ExpertParallel (all_to_all-based) + Triton MoE disabled
# 3. Triton MoE: XPUExpertParallel + Triton MoE enabled
# 4. Full Native: ExpertParallel + Triton MoE enabled
#
# Run on Aurora compute node:
#   bash scripts/benchmark_native_vs_workaround.sh

set -e

# Configuration
CONFIG="torchtitan/torchtitan/models/llama4/train_configs/llama4_3b_moe_ep12_xpu.toml"
STEPS=100
RANKS=12
PPN=12
OUTPUT_DIR="./benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Benchmark: Native XCCL Ops vs XPU Workarounds"
echo "============================================================"
echo "Config: $CONFIG"
echo "Steps: $STEPS"
echo "Ranks: $RANKS (ppn=$PPN)"
echo "Output: $OUTPUT_DIR"
echo "Timestamp: $TIMESTAMP"
echo "============================================================"

# Ensure environment
module load frameworks 2>/dev/null || true
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export PYTHONPATH=$(pwd):$(pwd)/torchtitan:$PYTHONPATH

# Common mpiexec options
MPI_OPTS="-n $RANKS -ppn $PPN --envall"

# Function to extract metrics from log
extract_metrics() {
    local logfile=$1
    echo "--- Metrics from $logfile ---"
    # Extract last few lines with MFU/throughput
    grep -E "step.*loss|MFU|tps|wps|tokens" "$logfile" | tail -20
    echo ""
}

# Function to run benchmark
run_benchmark() {
    local name=$1
    local env_vars=$2
    local extra_args=$3
    local logfile="$OUTPUT_DIR/${name}_${TIMESTAMP}.log"
    
    echo ""
    echo "============================================================"
    echo "Benchmark: $name"
    echo "============================================================"
    echo "Log: $logfile"
    echo "Environment: $env_vars"
    echo "Extra args: $extra_args"
    echo ""
    
    # Set environment variables
    if [ -n "$env_vars" ]; then
        export $env_vars
    fi
    
    # Run training
    mpiexec $MPI_OPTS \
        python -u torchtitan/mpi_train_wrapper.py \
        --job.config_file "$CONFIG" \
        --training.steps "$STEPS" \
        $extra_args \
        2>&1 | tee "$logfile"
    
    # Unset environment variables
    unset TORCHTITAN_XPU_FORCE_NATIVE_EP 2>/dev/null || true
    
    echo ""
    echo "Completed: $name"
    extract_metrics "$logfile"
}

# ============================================================
# Benchmark 1: Baseline (XPUExpertParallel + Triton MoE disabled)
# ============================================================
run_benchmark \
    "baseline_xpu_ep" \
    "" \
    "--model.use_triton_moe=false"

# ============================================================
# Benchmark 2: Native EP (ExpertParallel all_to_all + Triton MoE disabled)
# ============================================================
run_benchmark \
    "native_ep_all_to_all" \
    "TORCHTITAN_XPU_FORCE_NATIVE_EP=1" \
    "--model.use_triton_moe=false"

# ============================================================
# Benchmark 3: Triton MoE (XPUExpertParallel + Triton MoE enabled)
# ============================================================
run_benchmark \
    "triton_moe_enabled" \
    "" \
    "--model.use_triton_moe=true"

# ============================================================
# Benchmark 4: Full Native (ExpertParallel + Triton MoE enabled)
# ============================================================
run_benchmark \
    "full_native" \
    "TORCHTITAN_XPU_FORCE_NATIVE_EP=1" \
    "--model.use_triton_moe=true"

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "BENCHMARK COMPLETE"
echo "============================================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Log files:"
ls -la "$OUTPUT_DIR"/*_${TIMESTAMP}.log
echo ""
echo "To analyze results:"
echo "  python scripts/analyze_benchmark.py $OUTPUT_DIR/*_${TIMESTAMP}.log"
echo "============================================================"
