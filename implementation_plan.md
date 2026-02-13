# Implementation Plan: Expert Parallelism Benchmarking on Intel XPU

## Executive Summary

This document tracks our progress on enabling and benchmarking **Expert Parallelism (EP)** for MoE models on Intel Aurora XPU. The primary goal is a **fair comparison** between:

1. **Dense** - Baseline dense model
2. **MoE (No EP)** - MoE with all experts replicated on each rank
3. **MoE (EP)** - MoE with experts sharded across ranks (native all_to_all)
4. **MoE (EP Workaround)** - MoE using XPUExpertParallel (all_gather-based)
5. **MoE (EP+HSDP)** - MoE with Expert Parallel + Hybrid Sharded Data Parallel

---

## Current Status: NATIVE EP WORKS!

### Major Discovery (2026-02-07)

**Native EP (all_to_all-based) now works correctly on XCCL 25.190.0!**

Tested on node `x4311c3s4b0n0` with 1B MoE EP=12, AC=none:

| Implementation | Steps Completed | Memory Pattern | Status |
|----------------|-----------------|----------------|--------|
| Native EP (all_to_all) | 10/10 | 32→36 GiB (stable) | ✅ WORKS |
| XPU EP (all_gather) | 3/10 | 31→35 GiB then HANG | ❌ HANGS |

### XPU EP Issues Identified

1. **Memory leak**: ~4 GiB jump between step 2 and 3 (despite `.detach()` fix)
2. **Hang**: `all_gather` in `_all_gather_combine` hangs at step 4

### Previous Issue (now resolved)

**3B models had OOM issues** - both EP implementations failed with resource errors:

**This is surprising because:**
- 3B params at BF16 = ~6 GiB model weights
- 12 XPUs × 64 GiB = 768 GiB total memory
- Dense 3B runs fine at 41.20 GiB (64%) per rank
- Previous 1B MoE EP=12 runs worked (9.56% MFU in TESTS.md)

### Working Baseline

Dense 3B model (no compile, bs=4) runs successfully:
```
step: 10  loss: 4.8728  memory: 41.20GiB(64.39%)  tps: 363  tflops: 7.76  mfu: 2.60%
```

---

## Root Cause Investigation (TODO)

### Hypothesis 1: EP Communication Buffers

The EP implementations allocate large communication buffers:
- **XPUExpertParallel**: `all_gather` creates tensors of size `EP_degree × max_tokens × hidden_dim`
- **Native EP**: Token reordering creates intermediate tensors for dispatch/combine

With EP=12, seq_len=2048, bs=4, dim=3072:
- Tokens per rank: 4 × 2048 = 8,192
- Per-token size at BF16: 3072 × 2 = 6,144 bytes
- Gathered buffer: 12 × 8,192 × 6,144 = ~600 MB per layer
- With 14 MoE layers: ~8.4 GB just for gather buffers

### Hypothesis 2: Optimizer State Explosion

AdamW requires 2× model size for momentum/variance:
- MoE 3B total params: 9.4B params
- Optimizer states: 9.4B × 2 × 4 bytes = ~75 GB (for FP32 states)
- Even with FSDP sharding: 75 GB / 12 = 6.25 GB per rank

### Hypothesis 3: Activation Memory

With selective AC and seq_len=2048:
- Activations per layer: bs × seq × dim × 2 = 4 × 2048 × 3072 × 2 = ~50 MB
- 28 layers: ~1.4 GB base activations
- MoE layers have additional router/expert activations

### Action Items

1. [ ] Add memory profiling before/after each EP operation
2. [ ] Compare memory usage: Dense vs MoE No-EP vs MoE EP
3. [ ] Test with smaller batch sizes (bs=2, bs=1)
4. [ ] Test 1B MoE to verify it still works
5. [ ] Profile with `torch.xpu.memory_stats()`

---

## Benchmark Matrix

### Target Configurations (3B ISO-Active Comparison)

We want to compare models with **equivalent active parameters** (~3B):

| Config | Model Flavor | Total Params | Active Params | EP | Compile | AC | Status |
|--------|--------------|--------------|---------------|----|---------|----|--------|
| Dense | 3B_dense_12_rank | 3.6B | 3.6B | 1 | Yes | Selective | WORKS |
| MoE No-EP | 3B_12_rank | 9.4B | 3.6B | 1 | Yes | Selective | TODO |
| MoE EP=12 (native) | 3B_12_rank | 9.4B | 3.6B | 12 | Yes | Selective | OOM |
| MoE EP=12 (workaround) | 3B_12_rank | 9.4B | 3.6B | 12 | Yes | Selective | OOM |
| MoE HSDP EP=6 | 3B_12_rank | 9.4B | 3.6B | 6 | Yes | Selective | TODO |

### Model Architecture Details

**3B_12_rank (MoE)**:
```python
dim=3072, n_layers=28, n_heads=24, n_kv_heads=8
vocab_size=128256, num_experts=12, top_k=1
interleave_moe_layer_step=2  # Alternating MoE/FFN layers
# 14 MoE layers, 14 FFN layers
```

**3B_dense_12_rank (Dense)**:
```python
dim=3072, n_layers=28, n_heads=24, n_kv_heads=8
vocab_size=128256
interleave_moe_layer_step=100  # All FFN layers
```

### Required Configuration Consistency

All benchmarks MUST use:
- `compile = true` - Enables torch.compile for fair comparison
- `activation_checkpoint.mode = "selective"` - Consistent AC overhead
- `local_batch_size = 4` - Start with same batch, increase if memory allows
- `seq_len = 2048` - Standard sequence length
- `mixed_precision_param = "bfloat16"` - BF16 training

---

## Configuration Files

### Existing Configs (need audit)

| Config | Compile | AC | Batch | Notes |
|--------|---------|-----|-------|-------|
| `llama4_3b_dense_xpu_compile_bs4.toml` | Yes | Selective | 4 | GOOD |
| `llama4_3b_moe_ep12_xpu_compile_bs4.toml` | Yes | Selective | 4 | GOOD |
| `llama4_3b_moe_hsdp_ep6_xpu_compile_bs8.toml` | Yes | Selective | 8 | GOOD |
| `llama4_3b_moe_noep_xpu.toml` | No | Selective | 4 | NEEDS COMPILE |
| `llama4_3b_dense_xpu_nocompile.toml` | No | Selective | 4 | For debugging |
| `llama4_3b_moe_ep12_xpu_nocompile.toml` | No | Selective | 4 | For debugging |

### Missing Configs Needed

1. `llama4_3b_moe_noep_xpu_compile.toml` - MoE without EP, with compile

---

## Technical Fixes Completed

### 1. Native EP all_to_all Shape Fix (DONE)

**File**: `torchtitan/torchtitan/distributed/expert_parallel.py:167-188`

The combine operation must **reverse** the dispatch split sizes:
```python
def _token_combine(self, mod, routed_output, device_mesh):
    routed_output = _unpermute(routed_output, self.input_shape, self.permuted_indices)
    # FIXED: Swap input_splits and output_splits for reverse operation
    routed_output = all_to_all_single_autograd(
        routed_output,
        self.input_splits,   # receive: what we originally sent
        self.output_splits,  # send: what we currently have
        device_mesh.get_group(),
    )
    return routed_output
```

### 2. MeshAwareOptimizersContainer (DONE)

**File**: `torchtitan/torchtitan/components/optimizer.py`

Created optimizer that groups parameters by DeviceMesh to avoid DTensor mesh mismatch errors during fused Adam.

### 3. FSDP PREMUL_SUM Workaround (DONE)

**File**: `torchtitan/torchtitan/models/llama4/infra/parallelize.py`

XCCL doesn't support PREMUL_SUM (NCCL-specific). Monkey-patched to use SUM + post-division.

### 4. Activation Checkpoint early_stop Fix (DONE)

**File**: `torchtitan/distributed/activation_checkpoint.py`

Removed unsupported `early_stop` argument from `ptd_checkpoint_wrapper` calls.

---

## EP Implementation Comparison

| Aspect | Native ExpertParallel | XPUExpertParallel |
|--------|----------------------|-------------------|
| Communication | all_to_all | all_gather |
| Collectives/layer | 2 | 4 |
| Memory overhead | Token reordering buffers | Padded gather buffers |
| Status | ✅ WORKS (XCCL 25.190.0) | ❌ HANGS after ~3 steps |
| 1B EP=12 Results | 10 steps, 32-36 GiB | Hangs at step 4 |
| Recommended | **USE THIS** | Deprecated |

---

## Test Commands

### Environment Setup
```bash
ssh <compute_node>
cd /lus/flare/projects/AuroraGPT/ngetty/torchtitan_xpu
module load frameworks
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_WORKER_COUNT=4
export PYTHONPATH=/lus/flare/projects/AuroraGPT/ngetty:/lus/flare/projects/AuroraGPT/ngetty/torchtitan_xpu/torchtitan:$PYTHONPATH
export MASTER_PORT=29510
```

### Run Benchmarks
```bash
# Dense 3B (baseline - WORKS)
mpiexec -n 12 -ppn 12 --envall python -u torchtitan/mpi_train_wrapper.py \
    --job.config_file torchtitan/torchtitan/models/llama4/train_configs/llama4_3b_dense_xpu_compile_bs4.toml

# MoE 3B No-EP (TODO - create compile config)
mpiexec -n 12 -ppn 12 --envall python -u torchtitan/mpi_train_wrapper.py \
    --job.config_file torchtitan/torchtitan/models/llama4/train_configs/llama4_3b_moe_noep_xpu_compile.toml

# MoE 3B EP=12 (OOM - need to debug)
mpiexec -n 12 -ppn 12 --envall python -u torchtitan/mpi_train_wrapper.py \
    --job.config_file torchtitan/torchtitan/models/llama4/train_configs/llama4_3b_moe_ep12_xpu_compile_bs4.toml

# MoE 3B HSDP EP=6 (TODO)
mpiexec -n 12 -ppn 12 --envall python -u torchtitan/mpi_train_wrapper.py \
    --job.config_file torchtitan/torchtitan/models/llama4/train_configs/llama4_3b_moe_hsdp_ep6_xpu_compile_bs8.toml

# 1B MoE EP=12 (verify still works)
mpiexec -n 12 -ppn 12 --envall python -u torchtitan/mpi_train_wrapper.py \
    --job.config_file torchtitan/torchtitan/models/llama4/train_configs/llama4_1b_moe_ep12_xpu.toml
```

### Select EP Implementation
```bash
# XPUExpertParallel (all_gather-based) - default
unset TORCHTITAN_XPU_FORCE_NATIVE_EP

# Native ExpertParallel (all_to_all-based)
export TORCHTITAN_XPU_FORCE_NATIVE_EP=1
```

---

## Next Steps (Priority Order)

### Phase 1: Validate Native EP (HIGH PRIORITY) ✅ DONE

1. ✅ **Verify 1B works with Native EP**: Completed 10 steps successfully
2. ✅ **Compare Native vs XPU EP**: Native works, XPU EP hangs

### Phase 2: Test Native EP with AC Enabled

3. ✅ **Test Native EP + AC=selective**: MEMORY LEAK! ~8 GiB/step, OOM after step 5
4. [ ] **Test Native EP + compile**: Enable torch.compile for performance
5. [ ] **Investigate AC + EP memory leak**: Both implementations leak with AC enabled

**KEY FINDING**: Activation checkpointing + Expert Parallelism causes memory leaks!
- Native EP + AC=none: Works (10 steps, 32-36 GiB stable)
- Native EP + AC=selective: Leaks ~8 GiB/step, OOM after step 5
- XPU EP + AC=none: Hangs at step 4 (separate issue)

### Phase 3: Scale to 3B Model

6. [ ] **Test 3B Dense (baseline)**: Already confirmed working at 41 GiB
7. [ ] **Test 3B MoE No-EP**: Verify MoE without EP works
8. [ ] **Test 3B MoE EP=12 Native**: Use native EP for 3B model
9. [ ] **Test 3B MoE HSDP EP=6**: Hybrid sharded data parallel

### Phase 4: Cleanup & Document

10. [ ] **Deprecate XPUExpertParallel**: Mark as legacy/broken
11. [ ] **Update `__init__.py`**: Make native EP the default
12. [ ] **Update README/TESTS.md**: Document native EP as the solution
13. [ ] **Prepare for upstream contribution**

---

## Key Questions to Answer

1. **Why does 3B MoE OOM when 1B worked?**
   - Is it model size, optimizer states, or EP buffer scaling?

2. **What's the memory breakdown for EP vs No-EP?**
   - Model weights, optimizer states, activations, EP buffers

3. **Is the OOM in forward or backward pass?**
   - XPUExpertParallel OOMs in `_all_gather_combine` (seems like backward)
   - Native EP OOMs in `TokenReorderer.forward`

4. **Can we reduce EP memory overhead?**
   - Chunked all_gather, in-place operations, buffer reuse

---

## Reference: Previous Working Results (1B Model)

From TESTS.md, the 1B MoE EP=12 achieved:

| Metric | Value |
|--------|-------|
| MFU | 9.56% |
| Tokens/sec | 4,246 |
| TFLOPS | 28.50 |
| Memory | ~39 GiB (~61%) |

This proves the EP implementation works - the issue is scaling to 3B.

---

## Appendix: File Locations

| Component | File |
|-----------|------|
| XPUExpertParallel | `expert_parallel_xpu.py` |
| Native ExpertParallel | `torchtitan/torchtitan/distributed/expert_parallel.py` |
| MoE Layer | `torchtitan/torchtitan/models/moe/moe.py` |
| Model Parallelization | `torchtitan/torchtitan/models/llama4/infra/parallelize.py` |
| Mesh-Aware Optimizer | `torchtitan/torchtitan/components/optimizer.py` |
| Model Flavors | `torchtitan/torchtitan/models/llama4/__init__.py` |
| 3B Configs | `torchtitan/torchtitan/models/llama4/train_configs/llama4_3b_*.toml` |

---

## Changelog

| Date | Change |
|------|--------|
| 2026-02-07 | **MAJOR FINDING**: Native EP (all_to_all) works on XCCL 25.190.0! XPU EP (all_gather) has hang issues. |
| 2026-02-07 | Updated plan with OOM investigation focus, benchmark matrix, and clear next steps |
| 2026-02-06 | Added benchmark results - native EP works with identical performance to workaround |
| 2026-02-06 | Initial document - all_to_all fix completed, mesh issue identified |
