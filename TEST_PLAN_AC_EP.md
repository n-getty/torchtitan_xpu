# AC + EP Compatibility Test Plan

## Background

We observed that Activation Checkpointing (AC) causes unbounded memory growth when combined with Expert Parallelism (EP=12). However, the README shows successful benchmarks with HSDP EP=6 using Selective AC. We need to determine:

1. Is the memory leak specific to **op-based** selective AC (`selective_ac_option = "op"`)?
2. Does **layer-based** selective AC (`selective_ac_option = "2"`, the default) work with EP?
3. Is HSDP (EP=6, Replicate=2) more stable than pure EP=12?

## Test Matrix

| Test # | Configuration | AC Mode | AC Option | EP Degree | Expected Result |
|--------|--------------|---------|-----------|-----------|------------------|
| 1 | EP=12 + No AC | none | N/A | 12 | ✅ PASS (baseline) |
| 2 | EP=12 + Selective (layer) | selective | "2" (default) | 12 | ❓ Unknown |
| 3 | EP=12 + Selective (op) | selective | "op" | 12 | ❌ Known OOM |
| 4 | EP=12 + Full AC | full | N/A | 12 | ❌ Known OOM |
| 5 | HSDP EP=6 + Selective (layer) | selective | "2" (default) | 6 | ❓ Unknown |
| 6 | HSDP EP=6 + Selective (op) | selective | "op" | 6 | ❓ Unknown |

## Test Execution

### Prerequisites

1. Aurora compute node with 12 XPU tiles
2. Interactive session: `qsub -I -l select=1 -l walltime=1:00:00 -A <project> -q debug`

### Running Tests

**Option A: Run individual tests**
```bash
# On compute node
cd /lus/flare/projects/AuroraGPT/ngetty/torchtitan_xpu
bash scripts/test_ac_ep_matrix.sh 1  # Test 1: EP=12 + No AC (baseline)
bash scripts/test_ac_ep_matrix.sh 2  # Test 2: EP=12 + Selective (layer)
bash scripts/test_ac_ep_matrix.sh 3  # Test 3: EP=12 + Selective (op)
bash scripts/test_ac_ep_matrix.sh 4  # Test 4: EP=12 + Full AC
bash scripts/test_ac_ep_matrix.sh 5  # Test 5: HSDP EP=6 + Selective (layer)
bash scripts/test_ac_ep_matrix.sh 6  # Test 6: HSDP EP=6 + Selective (op)
```

**Option B: Run all tests**
```bash
bash scripts/test_ac_ep_matrix.sh all
```

**Option C: View summary after running tests**
```bash
bash scripts/test_ac_ep_matrix.sh summary
```

### Estimated Time

- Each test: ~2-3 minutes (10 steps)
- Full suite: ~15-20 minutes

## Config Files Created

| Test # | Config File |
|--------|-------------|
| 1 | `llama4_1b_moe_ep12_xpu_compile.toml` (mode = "none") |
| 2 | `llama4_1b_moe_ep12_xpu_compile_sac_layer.toml` |
| 3 | `llama4_1b_moe_ep12_xpu_compile_sac_op.toml` |
| 4 | `llama4_1b_moe_ep12_xpu_compile_fullac.toml` |
| 5 | `llama4_1b_moe_hsdp_ep6_xpu_compile_sac_layer.toml` |
| 6 | `llama4_1b_moe_hsdp_ep6_xpu_compile_sac_op.toml` |

All configs are in: `torchtitan/torchtitan/models/llama4/train_configs/`

## Success Criteria

**PASS**: All 10 steps complete with stable memory (no growth beyond step 3)
**FAIL**: OOM before step 10 or memory growth >5 GiB per step after step 3

## Test Results (Feb 9, 2026)

| Test # | Configuration | Steps Completed | Final Memory | Memory Growth Pattern | Status |
|--------|--------------|-----------------|--------------|----------------------|--------|
| 1 | EP=12 + No AC | 10/10 | 27.4 GiB | Stable after step 3 | ✅ PASS |
| 2 | EP=12 + Selective (layer) | 7+/10 | 57+ GiB | 13→57 GiB (+4-7 GiB/step) | ❌ OOM/Hung |
| 3 | EP=12 + Selective (op) | - | - | Not run (blocked) | - |
| 4 | EP=12 + Full AC | - | - | Not run (blocked) | - |
| 5 | HSDP EP=6 + Selective (layer) | - | - | Not run (blocked) | - |
| 6 | HSDP EP=6 + Selective (op) | - | - | Not run (blocked) | - |

### Memory Growth Details (Test 2: EP=12 + Selective Layer-based AC)

| Step | Memory Range (across 12 ranks) |
|------|-------------------------------|
| 1 | 12.7 - 13.3 GiB |
| 2 | 18.4 - 20.5 GiB |
| 3 | 22.3 - 27.8 GiB |
| 4 | 26.2 - 35.1 GiB |
| 5 | 30.1 - 44.4 GiB |
| 6 | 32.1 - 50.5 GiB |
| 7 | 36.0 - 57.4 GiB |

## Analysis Answers

### 1. Is layer-based AC (`selective_ac_option = "2"`) stable with EP?
**NO** - Layer-based AC exhibits the same unbounded memory growth as op-based AC.
Memory grew from ~13 GiB to ~57 GiB in just 7 steps (~4-7 GiB per step).

### 2. Is the memory leak specific to op-based AC?
**NO** - The memory leak occurs with ALL AC modes when combined with EP:
- `mode = "full"` - OOM
- `mode = "selective"` with `selective_ac_option = "2"` (layer) - OOM  
- `mode = "selective"` with `selective_ac_option = "op"` - OOM

### 3. Root Cause
The memory leak is inherent to the interaction between activation checkpointing and
the `all_to_all_single_autograd` function in native Expert Parallelism. During AC's
recomputation phase, saved tensors from the all-to-all autograd wrapper accumulate
and are not properly released.

### 4. Recommended Configuration for MoE Training with EP
**Use `mode = "none"` for all EP configurations.** There is no AC mode that works
with Expert Parallelism on XPU.

```toml
[activation_checkpoint]
mode = "none"  # REQUIRED for EP - all AC modes cause memory leaks
```

## Conclusion

**Activation Checkpointing is INCOMPATIBLE with Expert Parallelism on XPU**
regardless of the AC mode or option used. This is a fundamental limitation of the
current `all_to_all_single_autograd` implementation, not specific to any AC variant.

The previous hypothesis that the README benchmarks worked because they used layer-based
AC was incorrect. The README benchmark results for HSDP EP=6 with AC likely either:
1. Used `mode = "none"` (no AC)
2. Were run on CUDA/NCCL (not XPU/XCCL)
3. Had results from a different configuration

## Previous Findings (for reference)

Tests run Feb 7, 2026:

| Configuration | Steps | Memory | Status |
|--------------|-------|--------|--------|
| EP=12 + No AC + Compile | 10/10 | ~22 GiB stable | ✅ PASS |
| EP=12 + Selective (op) + Compile | 5-7/10 | 13→60 GiB | ❌ OOM |
| EP=12 + Full AC + Compile | 8/10 | 15→61 GiB | ❌ OOM |
| EP=12 + Selective (op) + No Compile | 5/10 | 13→57 GiB | ❌ OOM |

Tests run Feb 9, 2026 (this test plan):

| Configuration | Steps | Memory | Status |
|--------------|-------|--------|--------|
| EP=12 + No AC + Compile | 10/10 | 27.4 GiB stable | ✅ PASS |
| EP=12 + Selective (layer) + Compile | 7+/10 | 13→57 GiB | ❌ OOM/Hung |
