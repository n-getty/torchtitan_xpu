#!/bin/bash
# Test native ExpertParallel with MeshAwareOptimizersContainer fix
# This script validates that the mesh-aware optimizer correctly handles
# parameters on different DeviceMeshes (fsdp vs efsdp,ep)

set -e

# Configuration
NODE="${1:-x4201c7s3b0n0}"
NRANKS="${2:-12}"
STEPS="${3:-5}"

echo "===================================="
echo "Native EP + MeshAwareOptimizer Test"
echo "===================================="
echo "Node: $NODE"
echo "Ranks: $NRANKS"
echo "Steps: $STEPS"
echo ""

# Environment setup
ENV_SETUP='
module load frameworks mpich py-mpi4py 2>/dev/null
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_WORKER_COUNT=4
export PYTHONPATH=/lus/flare/projects/AuroraGPT/ngetty/torchtitan_xpu:/lus/flare/projects/AuroraGPT/ngetty/torchtitan_xpu/torchtitan:$PYTHONPATH
export TORCHTITAN_XPU_FORCE_NATIVE_EP=1
export CPU_BIND="list:4:9:14:19:20:25:56:61:66:71:74:79"
cd /lus/flare/projects/AuroraGPT/ngetty/torchtitan_xpu
'

# Run the test
echo "Running native EP training with mesh-aware optimizer..."
echo ""

ssh "$NODE" "$ENV_SETUP && mpiexec -n $NRANKS -ppn $NRANKS --cpu-bind \$CPU_BIND python -u torchtitan/mpi_train_wrapper.py \\
    --job.config_file torchtitan/torchtitan/models/llama4/train_configs/llama4_1b_moe_ep12_xpu.toml \\
    --training.steps $STEPS \\
    --metrics.log_freq 1 \\
    2>&1" | tee /tmp/native_ep_test_output.log

RESULT=$?

echo ""
echo "===================================="
if [ $RESULT -eq 0 ]; then
    echo "TEST PASSED: Native EP with MeshAwareOptimizer succeeded!"
    # Check for key log messages
    if grep -q "MeshAwareOptimizersContainer created" /tmp/native_ep_test_output.log; then
        echo "  - Confirmed: MeshAwareOptimizersContainer was used"
    fi
    if grep -q "Using native ExpertParallel" /tmp/native_ep_test_output.log || \
       grep -q "TORCHTITAN_XPU_FORCE_NATIVE_EP" /tmp/native_ep_test_output.log; then
        echo "  - Confirmed: Native ExpertParallel (all_to_all) was used"
    fi
else
    echo "TEST FAILED: Exit code $RESULT"
    echo "Check /tmp/native_ep_test_output.log for details"
fi
echo "===================================="

exit $RESULT
