#!/bin/bash
# AC + EP Compatibility Test Suite
# Tests different combinations of Activation Checkpointing and Expert Parallelism
# to identify which configurations are stable on XPU
#
# Test Matrix:
# 1. EP=12 + No AC (baseline - known to work)
# 2. EP=12 + Selective AC (layer-based, default "2")
# 3. EP=12 + Selective AC (op-based)
# 4. EP=12 + Full AC
# 5. HSDP EP=6 + Selective AC (layer-based)
# 6. HSDP EP=6 + Selective AC (op-based)
#
# Usage: bash scripts/test_ac_ep_matrix.sh <test_number>
# Example: bash scripts/test_ac_ep_matrix.sh 1

set -e

cd /lus/flare/projects/AuroraGPT/ngetty/torchtitan_xpu
module load frameworks 2>/dev/null || true

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_WORKER_COUNT=4
export PYTHONPATH=/lus/flare/projects/AuroraGPT/ngetty/torchtitan_xpu:/lus/flare/projects/AuroraGPT/ngetty/torchtitan_xpu/torchtitan:$PYTHONPATH
export MASTER_PORT=29510
export MASTER_ADDR=localhost

TEST_NUM=${1:-all}
CONFIG_DIR="torchtitan/torchtitan/models/llama4/train_configs"

run_test() {
    local name=$1
    local config=$2
    local ranks=$3
    
    echo ""
    echo "========================================"
    echo "TEST: $name"
    echo "Config: $config"
    echo "Ranks: $ranks"
    echo "Started: $(date)"
    echo "========================================"
    echo ""
    
    mpiexec -n $ranks -ppn $ranks --envall python -u torchtitan/mpi_train_wrapper.py \
        --job.config_file "$config" \
        2>&1 | tee "/tmp/ac_ep_test_${name}.log"
    
    local status=$?
    echo ""
    echo "TEST $name completed with status: $status"
    echo "Finished: $(date)"
    echo "========================================"
    echo ""
    
    return $status
}

print_summary() {
    echo ""
    echo "========================================"
    echo "TEST SUMMARY"
    echo "========================================"
    echo ""
    for log in /tmp/ac_ep_test_*.log; do
        if [ -f "$log" ]; then
            name=$(basename "$log" .log | sed 's/ac_ep_test_//')
            if grep -q "Training completed" "$log"; then
                steps=$(grep -c "step:" "$log" | head -1)
                final_mem=$(grep "step:" "$log" | tail -1 | grep -oP 'memory: \K[0-9.]+GiB')
                echo "✅ $name: SUCCESS (memory: $final_mem)"
            elif grep -q "OutOfMemoryError\|OOM\|OUT_OF_RESOURCES" "$log"; then
                failed_step=$(grep "step:" "$log" | tail -1 | grep -oP 'step:\s*\K[0-9]+')
                echo "❌ $name: OOM at step $failed_step"
            else
                echo "❓ $name: UNKNOWN (check log)"
            fi
        fi
    done
    echo ""
}

case $TEST_NUM in
    1)
        echo "Test 1: EP=12 + No AC (baseline)"
        run_test "ep12_noac" "$CONFIG_DIR/llama4_1b_moe_ep12_xpu_compile.toml" 12
        ;;
    2)
        echo "Test 2: EP=12 + Selective AC (layer-based, default)"
        run_test "ep12_sac_layer" "$CONFIG_DIR/llama4_1b_moe_ep12_xpu_compile_sac_layer.toml" 12
        ;;
    3)
        echo "Test 3: EP=12 + Selective AC (op-based)"
        run_test "ep12_sac_op" "$CONFIG_DIR/llama4_1b_moe_ep12_xpu_compile_sac_op.toml" 12
        ;;
    4)
        echo "Test 4: EP=12 + Full AC"
        run_test "ep12_fullac" "$CONFIG_DIR/llama4_1b_moe_ep12_xpu_compile_fullac.toml" 12
        ;;
    5)
        echo "Test 5: HSDP EP=6 + Selective AC (layer-based)"
        run_test "hsdp_ep6_sac_layer" "$CONFIG_DIR/llama4_1b_moe_hsdp_ep6_xpu_compile_sac_layer.toml" 12
        ;;
    6)
        echo "Test 6: HSDP EP=6 + Selective AC (op-based)"
        run_test "hsdp_ep6_sac_op" "$CONFIG_DIR/llama4_1b_moe_hsdp_ep6_xpu_compile_sac_op.toml" 12
        ;;
    all)
        echo "Running all tests sequentially..."
        echo "This will take approximately 15-20 minutes"
        echo ""
        
        # Clean up old logs
        rm -f /tmp/ac_ep_test_*.log
        
        run_test "ep12_noac" "$CONFIG_DIR/llama4_1b_moe_ep12_xpu_compile.toml" 12 || true
        run_test "ep12_sac_layer" "$CONFIG_DIR/llama4_1b_moe_ep12_xpu_compile_sac_layer.toml" 12 || true
        run_test "ep12_sac_op" "$CONFIG_DIR/llama4_1b_moe_ep12_xpu_compile_sac_op.toml" 12 || true
        run_test "ep12_fullac" "$CONFIG_DIR/llama4_1b_moe_ep12_xpu_compile_fullac.toml" 12 || true
        run_test "hsdp_ep6_sac_layer" "$CONFIG_DIR/llama4_1b_moe_hsdp_ep6_xpu_compile_sac_layer.toml" 12 || true
        run_test "hsdp_ep6_sac_op" "$CONFIG_DIR/llama4_1b_moe_hsdp_ep6_xpu_compile_sac_op.toml" 12 || true
        
        print_summary
        ;;
    summary)
        print_summary
        ;;
    *)
        echo "Usage: $0 <test_number|all|summary>"
        echo ""
        echo "Tests:"
        echo "  1  - EP=12 + No AC (baseline)"
        echo "  2  - EP=12 + Selective AC (layer-based)"
        echo "  3  - EP=12 + Selective AC (op-based)"
        echo "  4  - EP=12 + Full AC"
        echo "  5  - HSDP EP=6 + Selective AC (layer-based)"
        echo "  6  - HSDP EP=6 + Selective AC (op-based)"
        echo "  all     - Run all tests"
        echo "  summary - Print summary of previous run"
        exit 1
        ;;
esac
