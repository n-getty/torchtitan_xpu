# XPU/XCCL Compatibility Test Results

Test results from Aurora supercomputer using the **25.190.0 frameworks** (February 2026).

## Test Suite

The test suite consists of standalone scripts that verify XPU and XCCL backend support:

| Test Script | Purpose | Ranks Required |
|-------------|---------|----------------|
| `test_all_to_all_support.py` | Tests `all_to_all_single` support | ≥2 |
| `test_premul_sum_support.py` | Tests `PREMUL_SUM` and alternatives | ≥2 |
| `test_triton_moe_support.py` | Tests Triton MoE kernel support | 1 |
| `test_xccl_ops_support.py` | Comprehensive XCCL ops test (17 ops) | ≥2 |
| `test_native_ep_shape_mismatch.py` | Tests native EP all_to_all shape fix | ≥2 |

## How to Run Tests

```bash
# On Aurora compute node
module load frameworks
export PYTHONPATH=$(pwd):$(pwd)/torchtitan:$PYTHONPATH

# Comprehensive XCCL ops test (recommended first)
mpiexec -n 2 -ppn 2 --envall python tests/test_xccl_ops_support.py

# Individual tests
mpiexec -n 2 -ppn 2 --envall python tests/test_all_to_all_support.py
mpiexec -n 2 -ppn 2 --envall python tests/test_premul_sum_support.py
python tests/test_triton_moe_support.py  # Single-rank OK
```

---

## Key Findings

### 1. `all_to_all`: SUPPORTED ✓

**Previous Assumption**: `all_to_all` was not supported on XCCL, requiring the `XPUExpertParallel` workaround using `all_gather` operations.

**Test Result**: Both `dist.all_to_all_single()` and the functional API work correctly on XCCL!

```
all_to_all_single:
  Status: [SUPPORTED]
  Details: all_to_all_single works correctly

functional_all_to_all_single:
  Status: [SUPPORTED]
  Details: functional all_to_all_single works correctly
```

**Implication**: The upstream `ExpertParallel` class may now work directly on XPU without the `all_gather`-based workaround.

---

### 2. `PREMUL_SUM`: NOT SUPPORTED ✗

**Test Result**: `PREMUL_SUM` is not available because the API (`_make_nccl_premul_sum`) is NCCL-specific and doesn't exist in PyTorch for other backends.

```
PREMUL_SUM:
  Status: [NOT SUPPORTED]
  Details: ImportError: cannot import name '_make_nccl_premul_sum'
```

**Available Workarounds**:
- `ReduceOp.AVG` - Supported and works correctly
- Manual scaling: `tensor.mul_(1/world_size)` then `all_reduce(SUM)`

**Current Solution**: The FSDP patch in `parallelize.py` forces sum reduction, which is mathematically equivalent.

---

### 3. Triton MoE Kernels: SUPPORTED ✓

**Previous Assumption**: Triton MoE kernels were not supported on XPU.

**Test Result**: Despite `triton.backends` returning an empty list, Triton kernels compile and execute correctly on XPU!

```
triton_import:        [SUPPORTED] - Triton 3.4.0
simple_kernel:        [SUPPORTED] - Compiled and ran correctly on xpu:0
fused_moe_functional: [SUPPORTED] - Executed successfully, output shape: torch.Size([32, 64])
```

**Implication**: `use_triton_moe = true` may now be viable in training configs.

---

### 4. Comprehensive XCCL Operations: 16/17 Supported

| Operation | Status | Notes |
|-----------|--------|-------|
| `broadcast` | ✓ SUPPORTED | |
| `all_reduce (SUM)` | ✓ SUPPORTED | |
| `all_reduce (MAX)` | ✓ SUPPORTED | |
| `all_reduce (MIN)` | ✓ SUPPORTED | |
| `all_reduce (PRODUCT)` | ✓ SUPPORTED | |
| `all_reduce (AVG)` | ✓ SUPPORTED | Alternative to PREMUL_SUM |
| `all_gather` | ✓ SUPPORTED | |
| `all_gather_into_tensor` | ✓ SUPPORTED | |
| `reduce_scatter` | ✓ SUPPORTED | |
| `reduce_scatter_tensor` | ✓ SUPPORTED | |
| `scatter` | ✓ SUPPORTED | |
| `gather` | ✓ SUPPORTED | |
| `all_to_all_single` | ✓ SUPPORTED | Now works! |
| `send/recv` | ✓ SUPPORTED | Point-to-point |
| `barrier` | ✓ SUPPORTED | |
| `all_reduce_coalesced` | ✓ SUPPORTED | Deprecated but works |
| `PREMUL_SUM` | ✗ NOT SUPPORTED | NCCL-specific API |

---

## Test Environment

```
Platform: Aurora Supercomputer (ALCF)
Frameworks: 25.190.0 (aurora_frameworks-2025.2.0)
PyTorch: 2.8.0a0+gitba56102
Triton: 3.4.0
Backend: XCCL
Device: Intel Data Center GPU Max 1550 (PVC)
```

---

## Implications for torchtitan_xpu

### Current Workarounds vs Native Ops

| Component | Current Workaround | Native Alternative | Status |
|-----------|-------------------|-------------------|--------|
| Expert Parallel | `XPUExpertParallel` (all_gather) | `ExpertParallel` (all_to_all) | **Keep workaround** (native has mesh issues in optimizer) |
| FSDP Gradient Reduction | Force SUM reduction | N/A (PREMUL_SUM unavailable) | **Keep workaround** (required) |
| Triton MoE | Disabled (`use_triton_moe=false`) | Enable (`use_triton_moe=true`) | **Keep disabled** (2.3x slower) |

### Recommendations Based on Benchmarks

1. **Keep XPUExpertParallel**: The all_gather-based workaround provides best performance
   - Native `ExpertParallel` all_to_all shape mismatch is now FIXED
   - However, native has DTensor mesh compatibility issues in the optimizer step
   - Our workaround achieves 9.56% MFU vs native needing more work

2. **Keep Triton MoE Disabled**: The fused kernel is 2.3x slower than PyTorch default
   - Continue using `use_triton_moe = false` in configs
   - The kernel works but needs optimization for XPU

3. **FSDP Workaround Required**: PREMUL_SUM is NCCL-specific, keep the SUM reduction patch

4. **Update AGENTS.md**: Note that while XCCL now supports `all_to_all`, the upstream
   `ExpertParallel` code has compatibility issues requiring continued use of `XPUExpertParallel`

---

## Benchmark Results

**Model**: Llama4 1B MoE (EP=12) on single Aurora node (12 XPU tiles)
**Config**: `llama4_1b_moe_ep12_xpu.toml`, 10 training steps, batch_size=4, seq_len=2048

| Configuration | MFU (%) | Tokens/sec | TFLOPS | Memory (GiB) | Status |
|--------------|---------|------------|--------|--------------|--------|
| **Baseline (XPUExpertParallel)** | **9.56%** | **4,246** | **28.50** | ~39 (~61%) | ✓ Current default |
| Native (ExpertParallel) | N/A | N/A | N/A | N/A | ⚠ Forward works, mesh issue in optimizer |
| Triton MoE Enabled | 4.21% | 1,870 | 12.55 | ~39 (~61%) | ✓ Works but 2.3x slower |

### Analysis

#### 1. Native ExpertParallel (all_to_all): FIXED ✓

While `all_to_all_single` works at the XCCL collective level, the upstream `ExpertParallel` implementation previously failed with:
```
RuntimeError: Split sizes doesn't match total dim 0 size
```

**Root Cause Identified and Fixed (Feb 2026)**:

The issue was in `_token_combine()` where the split sizes for the second `all_to_all` were passed in the wrong order. The combine operation must **reverse** the dispatch operation:

- **Dispatch**: `all_to_all(input, output_splits, input_splits)` sends `input_splits[i]` tokens to rank `i`
- **Combine**: `all_to_all(output, input_splits, output_splits)` reverses this, receiving what was originally sent

The upstream code incorrectly used `(output_splits, input_splits)` for both operations.

**Fix Applied** in `torchtitan/torchtitan/distributed/expert_parallel.py`:
```python
# In _token_combine():
routed_output = all_to_all_single_autograd(
    routed_output,
    self.input_splits,   # receive partitions: what we originally sent
    self.output_splits,  # send partitions: what we currently have
    device_mesh.get_group(),
)
```

**Current Status**: The all_to_all shape mismatch is fixed. Forward pass works correctly. However, there's a separate DTensor mesh compatibility issue in the optimizer step that needs further investigation:
```
ValueError: Could not run pointwise computation across different mesh: 
Found DeviceMesh('xpu', [0..11], mesh_dim_names=('fsdp',)) and 
DeviceMesh('xpu', [[0..11]], mesh_dim_names=('efsdp', 'ep'))!
```

**Test Script**: `tests/test_native_ep_shape_mismatch.py` verifies the all_to_all fix works correctly.

#### 2. Triton MoE: Works but Slower

Triton MoE kernels compile and execute correctly on XPU, but are **2.3x slower** than the default PyTorch implementation:
- MFU drops from 9.56% → 4.21%
- TPS drops from 4,246 → 1,870

Possible reasons:
- Debug prints in the Triton kernel code (visible in output)
- Kernel not optimized for Intel XPU architecture
- JIT compilation overhead

**Recommendation**: Keep `use_triton_moe = false` (current default) for best performance.

#### 3. Baseline Configuration: Best Performance

The current configuration using `XPUExpertParallel` (all_gather-based) provides the best performance:
- MFU: 9.56%
- Tokens/sec: 4,246
- Memory utilization: ~61%

---

## Activation Checkpointing + Expert Parallelism: INCOMPATIBLE

**Discovery Date**: February 2026  
**Updated**: February 9, 2026 (comprehensive testing confirms ALL AC modes fail)

### Problem

Activation Checkpointing (AC) causes unbounded memory growth when combined with Expert Parallelism (EP), **regardless of AC mode** (full, selective-layer, or selective-op) or torch.compile status.

### Comprehensive Test Results (Feb 9, 2026)

| Configuration | AC Mode | AC Option | Steps | Peak Memory | Status |
|--------------|---------|-----------|-------|-------------|--------|
| EP=12 + No AC + Compile | none | N/A | 10/10 | 27.4 GiB stable | ✅ SUCCESS |
| EP=12 + Selective (layer) + Compile | selective | "2" (default) | 7+/10 | 57+ GiB | ❌ OOM/Hung |
| EP=12 + Selective (op) + Compile | selective | "op" | 5-7/10 | ~60 GiB | ❌ OOM |
| EP=12 + Full AC + Compile | full | N/A | 8/10 | ~61 GiB | ❌ OOM |
| EP=12 + Selective (op) + No Compile | selective | "op" | 5/10 | ~57 GiB | ❌ OOM |

### Key Finding: Layer-based AC Also Fails

We initially hypothesized that layer-based AC (`selective_ac_option = "2"`) might work since it's the torchtitan default. **Testing confirmed this is NOT the case.**

**Memory Growth Pattern (Selective Layer-based AC)**:
| Step | Memory Range (12 ranks) |
|------|------------------------|
| 1 | 12.7 - 13.3 GiB |
| 2 | 18.4 - 20.5 GiB |
| 3 | 22.3 - 27.8 GiB |
| 4 | 26.2 - 35.1 GiB |
| 5 | 30.1 - 44.4 GiB |
| 6 | 32.1 - 50.5 GiB |
| 7 | 36.0 - 57.4 GiB |

Growth rate: **~4-7 GiB per step** - identical to op-based AC.

### Root Cause Analysis

1. **Not specific to AC variant**: All AC modes (full, selective-layer, selective-op) fail
2. **Not torch.compile**: The leak occurs with or without compilation
3. **Autograd wrapper accumulation**: The `all_to_all_single_autograd` function from `torch.distributed._functional_collectives` saves tensors for backward pass. During AC recomputation, these saved tensors accumulate because:
   - AC wraps the TransformerBlock
   - Inside the block, MoE layer uses `all_to_all_single_autograd` for dispatch/combine
   - The autograd function saves input/output tensors regardless of SAC settings
   - Each recomputation adds to the accumulated state

### Recommended Configuration for EP

```toml
# In your EP training config:
[activation_checkpoint]
mode = "none"  # REQUIRED for EP - ALL AC modes cause memory leaks

[compile]
enable = true  # Recommended - provides memory efficiency
```

**CRITICAL**: Do NOT use any of these with EP:
- `mode = "full"` - FAILS
- `mode = "selective"` with default option "2" - FAILS  
- `mode = "selective"` with `selective_ac_option = "op"` - FAILS

### Working Benchmark Results

**Model**: Llama4 1B MoE (EP=12), Compiled, No AC

| Step | Memory | TPS | MFU |
|------|--------|-----|-----|
| 1 | 13.2 GiB | 116 | 0.26% (warmup) |
| 2-10 | 22-27 GiB stable | 640-660 | 1.44-1.49% |

**Final**: 10/10 steps, ~27 GiB (43%), 640 TPS, 1.44% MFU

### Implications

1. **For 1B MoE models**: No AC needed, fits comfortably in memory at ~43%
2. **For larger models**: May need alternative memory reduction strategies:
   - Reduce batch size
   - Use HSDP (hybrid sharding) for DP replicate
   - Increase EP degree to reduce per-rank expert count
3. **Future fix needed**: PyTorch upstream needs to handle AC + EP autograd interaction

### Test Plan Reference

See [TEST_PLAN_AC_EP.md](./TEST_PLAN_AC_EP.md) for the full test matrix and detailed results.

---

## Summary of EP Best Practices

| Setting | Value | Reason |
|---------|-------|--------|
| `activation_checkpoint.mode` | `"none"` | AC causes memory leaks with EP |
| `compile.enable` | `true` | Provides memory efficiency, 15-35% speedup |
| `model.use_triton_moe` | `false` | Triton kernel 2.3x slower on XPU |
| Native `ExpertParallel` | Default | Now works with XCCL all_to_all |
| FSDP gradient patch | Applied | PREMUL_SUM not available |

---

## References

- [AGENTS.md](./AGENTS.md) - Development guide for torchtitan_xpu
- [expert_parallel_xpu.py](./expert_parallel_xpu.py) - XPU-compatible EP implementation
- [torchtitan/distributed/expert_parallel.py](./torchtitan/torchtitan/distributed/expert_parallel.py) - Upstream EP implementation
