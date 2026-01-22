# Torchtitan XPU Extensions

XPU-compatible implementations for torchtitan that enable Expert Parallelism (EP) training on Intel Aurora systems.

## Problem

Torchtitan's Expert Parallelism uses `torch.distributed._functional_collectives.all_to_all_single()` which is **not implemented** in Intel's XCCL backend, causing hangs during MoE layer execution.

## Solution

This module provides `XPUExpertParallel` which replaces `all_to_all` with padded `all_gather` operations:

1. **Token Dispatch**: `all_gather(num_tokens_per_expert)` + `all_gather(padded_tokens)`
2. **Token Combine**: `all_gather(padded_outputs)` + reconstruct from metadata

This approach:
- Uses O(2) collective calls per layer (vs O(N) with naive broadcast loop)
- Handles dynamic token loads across experts
- Works with FSDP sharding

---

## Benchmark Results

### DeepSeek-V3 1B Baseline (`baseline_model_1b_12rank`)

| Configuration | Environment | Throughput (TPS) | MFU | Memory/Rank |
|--------------|-------------|-----------------|-----|-------------|
| **MoE (no EP)** | Frameworks (PT 2.8) | 2,577 | 1.45% | 7.1 GiB |
| **MoE (no EP)** | Dev (PT 2.9) | 2,620 | 1.47% | 7.1 GiB |
| **EP=12** (FSDP) | **Frameworks (PT 2.8)** | **6,292** | **3.54%** | **14.7 GiB** |
| **EP=12** (FSDP) | Dev (PT 2.9) | 5,618 | 3.16% | 14.5 GiB |
| **EP=6** (HSDP, Repl=2) | Dev (PT 2.9) | 7,330 | 4.13% | 5.0 GiB |

**Model Configuration:**
- Hidden Dim: 1024, Layers: 10, Heads: 8, Vocab: 50k
- Experts: 60, MoE every layer
- Batch: 4/rank, Seq: 512, Steps: 10

**Key Findings:**
- **System Frameworks (PT 2.8) is Faster**: Switching to the system-provided `frameworks` module improved EP=12 performance by **12%** (6292 vs 5618 TPS) and increased MFU to 3.54%.
- **HSDP (EP=6)** remains the fastest configuration (7,330 TPS), though it was tested on the previous Dev environment. It is expected to scale similarly on Frameworks.
- **Memory Efficiency**: EP=12 uses more memory (~14.7 GiB) than No-EP due to expert buffers, but fits comfortably within XPU limits.

### MFU Optimization: Batch Size Impact

The originally low MFU (~1-4%) was caused by **small batch size** resulting in low arithmetic intensity. With larger batch sizes (16*2048 tokens/rank), MFU increased dramatically:

| Configuration | Batch | Throughput (TPS) | MFU | Memory/Rank |
|--------------|-------|-----------------|-----|-------------|
| **MoE (No EP)** | 4*512 | 2,577 | 1.45% | 7.1 GiB |
| **MoE (No EP)** | **16*2048** | **16,242** | **10.17%** | **60.8 GiB** |
| **EP=12** | 4*512 | 6,292 | 3.54% | 14.7 GiB |
| **EP=12** | **16*2048** | **13,889** | **8.70%** | **61.0 GiB** |

**Conclusions:**
- MFU improved **7x** (1.45% → 10.17%) by maximizing batch size.
- EP overhead reduced from ~40% to ~15% at higher MFU.
- Use batch sizes that consume ~90%+ of XPU memory for optimal MFU.
- **MoE is Essential:** A Dense model with equivalent total parameters (~50B) runs at only **26 TPS**, giving MoE a **36x speedup** (935 TPS) due to sparse activation.

---

### Llama 3.2 1B & 3B Optimization Highlights

We performed a deep dive into optimizing Llama 3.2 1B and 3B model variants, leveraging `torch.compile` and high-utilization batch sizes.

#### 1. Llama 3.2 1B Performance
*Batch=4 (Baseline) -> 16 (Peak), Seq=2048*

| Configuration | Throughput (TPS) | MFU | Memory/Rank | Speedup |
| :--- | :---: | :---: | :---: | :---: |
| Dense | 8,953 | 20.15% | 23.6 GB | 1.0x |
| Dense + Compile | 12,003 | 27.01% | 13.2 GB | +34% |
| **Dense + Compile (BS=16)** | **14,101** | **31.73%** | **45.9 GB** | **+57%** |
| MoE (HSDP EP=6) | 5,353 | 12.05% | 26.1 GB | 0.6x |
| MoE (HSDP) + Compile | 6,302 | 14.18% | 15.9 GB | +17% |

#### 2. Llama 3.2 3B Performance
*Batch=4 (Baseline) -> 12 (Peak), Seq=2048*

| Model | Configuration | Throughput (TPS) | MFU | Memory/Rank | Speedup |
| :--- | :--- | :---: | :---: | :---: | :---: |
| 3B | Dense | 3,344 | 23.99% | 41.9 GB | 1.0x |
| 3B | Dense + Compile | 4,302 | 30.86% | 25.1 GB | +28% |
| **3B** | **Dense + Compile (BS=12)** | **4,834** | **34.67%** | **59.1 GB** | **+44%** |
| 3B | MoE (HSDP EP=6) | 1,969 | 14.13% | 47.7 GB | 0.6x |
| 3B | MoE (HSDP) + Compile | 2,256 | 16.18% | 35.1 GB | +15% |

#### 3. Iso-Parameter Comparison (Dense vs MoE)
By comparing a **3B Dense** model vs a **3B Total MoE** (smaller dimension, more layers/experts), we show that MoE is ~30% faster for the same storage cost.

| Feature | 3B Dense (Baseline) | 3B MoE (Iso-Param) | Delta |
| :--- | :---: | :---: | :---: |
| **Total Params** | ~3.2B | ~3.2B | Match |
| **Active Params** | ~3.2B | ~1.1B | **-65%** |
| **Throughput (TPS)** | **4,834** | **6,244** | **+29% Faster** |
| **Memory** | 59.1 GB | 51.6 GB | -13% |

### Large Model Benchmark Results (Llama4 ~50B Params)

We benchmarked the **Llama4 Large** model (32 Layers, 4608 Dim, 12 Experts) and a **Dense Equivalent** (same total parameter count) on 12 XPUs to evaluate the impact of Expert Parallelism (EP).

| Configuration | Model Type | Local Batch | Throughput (TPS) | MFU | Memory/Rank | Speedup (vs Dense) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Dense Equivalent** | Dense (~50B) | 2 | 26 | 1.94% | 53.9 GiB | 1x |
| **MoE No-EP** | MoE (12 Exp) | 2 | 260 | 3.95% | 48.4 GiB | 10x |
| **MoE EP=12** | MoE (12 Exp) | 2 | 684 | 10.39% | 29.2 GiB | 26x |
| **MoE EP=12 (Max)** | MoE (12 Exp) | **8** | **935** | **14.21%** | **47.6 GiB** | **36x** |

**Key Findings:**
1.  **MoE Superiority:** Even without optimization, MoE (No-EP) is **10x faster** than the equivalent Dense model due to sparse activation.
2.  **EP Efficiency:** Expert Parallelism (EP) provides another **2.6x speedup** over No-EP (684 vs 260 TPS) by efficiently sharding experts and reducing memory usage.
3.  **Scalability:** EP enables larger batch sizes (Batch 8 vs 2), unleashing the full potential of the XPU to reach **935 TPS (36x total speedup)** and **14.21% MFU**.

---

### Large Model (`largemodel_12_rank`) - Realistic Memory Usage

| Configuration | Throughput (TPS) | Memory/Rank | MFU |
|--------------|-----------------|-------------|-----|
| **Dense** (no MoE) | 914 | 21 GiB (33%) | 14% |
| **MoE (no EP)** | 263 | 48 GiB (76%) | 4% |
| **MoE + EP=12** | 710 | 29 GiB (46%) | 11% |

**Model Configuration:**
- Hidden Dim: 4608, Layers: 32, Heads: 36, KV Heads: 12, Vocab: 32k
- Experts: 12, MoE every 2 layers
- Batch: 2/rank, Seq: 512, Steps: 10

**Key Findings:**
- **EP=12 is 2.7x faster than MoE (No EP)** — each rank computes 1/12 of experts
- **EP saves 40% memory** (29 GiB vs 48 GiB) by sharding expert weights
- MoE without EP uses 76% of 64 GiB XPU memory, barely fits

---

### Medium Model (`mediummodel_12_rank`)

| Configuration | Throughput (TPS) | Memory/Rank | MFU |
|--------------|-----------------|-------------|-----|
| **Dense** (no MoE) | 4,842 | 9.25 GiB | 14.5% |
| **MoE (no EP)** | 2,212 | 14.75 GiB | 6.6% |
| **MoE + EP=12** | 3,329 | 15.2 GiB | 10.0% |

**Model Configuration:**
- Hidden Dim: 2304, Layers: 24, Heads: 24, Vocab: 32k
- Experts: 12, MoE every 2 layers
- Batch: 4/rank, Seq: 512, Steps: 20

**Analysis:**
- MoE is ~54% slower than Dense due to router + expert compute overhead
- **EP=12 is ~50% faster than MoE (no EP)** — each rank only computes 1/12 of experts
- EP communication overhead is offset by reduced per-rank compute

### Debug Model (`debugmodel_12_rank`)

| Configuration | Throughput (TPS) | Memory/Rank |
|--------------|-----------------|-------------|
| Dense | 96-100k | 1.0 GiB |
| EP=12 | 55-62k | 5.7 GiB |

---

## XPU Adaptation & Technical Details

Getting `torchtitan` to run on Intel Aurora (XPU) required specific environment configurations and code adaptations to bridge the gap between upstream PyTorch/IPEX and the XPU hardware capabilities.

### 1. Environment Configuration

The following environment setup is critical for correctness and performance. Using the **frameworks** module is recommended:

- **Modules**: `frameworks`, `mpich`, `py-mpi4py`.
  ```bash
  module load frameworks
  module load mpich         # Required for libmpi.so.12
  module load py-mpi4py     # Required for MPI communication
  ```
- **Environment Variables**:
  - `ZE_FLAT_DEVICE_HIERARCHY=FLAT`: Essential for correct device enumeration on Aurora's multi-tile GPU architecture.
  - `ONECCL_BINDINGS_FOR_PYTORCH_ENV_VERBOSE=1`: Enables detailed logging for the communication backend (ccl/xccl).
- **PYTHONPATH**:
  - Must include project root.
  - **Important**: When using `mpiexec` with `frameworks`, ensuring `intel_extension_for_pytorch` is visible may require explicitly appending the frameworks site-packages to `PYTHONPATH`. The provided launch scripts handle this.

### 2. Expert Parallelism (EP) Backend (`expert_parallel_xpu.py`)

The default `torchtitan` implementation uses `torch.distributed.all_to_all_single`, which relies on the underlying NCCL `all_to_all` primitive. The Intel XCCL backend for XPU does not currently support `all_to_all` in a way compatible with this API (it hangs or fails).

**The Workaround:**
We implemented `XPUExpertParallel`, a drop-in replacement that simulates `all_to_all` using `all_gather` operations:
- **Dispatch**: Instead of sending tokens directly to target rank:
  1. `all_gather` the per-expert token counts from all ranks.
  2. `all_gather` the padded tokens from all ranks.
  3. Local rank slices out the tokens relevant to its experts from the gathered buffer.
- **Combine**: The reverse process is used to send expert outputs back to their source ranks.

This changes the communication complexity from *O(N)* small messages to *O(2)* large collective calls per layer, which is more stable on the current XCCL stack.

### 3. FSDP Backward Pass Fix (`parallelize.py`)

PyTorch's FSDP2 implementation defaults to using `ReduceOp.PREMUL_SUM` for gradient reduction. This operator pre-multiplies gradients by `1/world_size` before reduction to avoid overflow. However, the XCCL backend does not verify support for `PREMUL_SUM`.

**The Patch:**
We monkey-patched `torch.distributed.fsdp._fully_shard._fsdp_collectives._get_gradient_divide_factors` to enforce `force_sum_reduction_for_comms=True`. This compels FSDP to use standard `ReduceOp.SUM` followed by a post-reduction division, which is fully supported by XCCL.

### 4. DeepSeek Compatibility

The `DeepSeek-V3` implementation in `torchtitan` uses newer PyTorch features like `torch._higher_order_ops.inductor_compiled_code`. This API was missing in the installed PyTorch 2.9 dev version.

**The Fix:**
We modified `deepseek_v3/infra/parallelize.py` to make the import of `inductor_compiled_code` optional, allowing the model to load and run without it.

### 5. PyTorch 2.8 Tensor Parallel Mesh Compatibility (`parallel_dims.py`)

PyTorch 2.8 enforces stricter validation on DeviceMesh structures, requiring the Tensor Parallel (TP) dimension to be the innermost dimension of the parent mesh. `torchtitan` by default creates a placeholder `etp` (Expert Tensor Parallel) dimension even when it is unused (degree 1). This caused a validation error because `ep` (Expert Parallel) was seen as obstructing `etp` in the dimension hierarchy.

**The Fix:**
We modified `torchtitan/distributed/parallel_dims.py` to conditionally exclude the `etp` dimension from the mesh creation process when `etp_degree` is 1. This ensures `ep` becomes the innermost dimension, satisfying the PyTorch validator.

---

## Installation

```bash
# Add to PYTHONPATH (before torchtitan's own path)
export PYTHONPATH="${PYTHONPATH}:/path/to/Aurora-Inferencing"
```

## Usage

### Auto-detection (Recommended)

The `parallelize_llama` function in torchtitan automatically uses `XPUExpertParallel` when XPU is detected:

```python
# In torchtitan/models/llama4/infra/parallelize.py
try:
    from torchtitan_xpu import get_expert_parallel_class
    experts_plan = get_expert_parallel_class()()
except ImportError:
    experts_plan = ExpertParallel()
```

### Explicit Usage

```python
from torchtitan_xpu import XPUExpertParallel

model = parallelize_module(model, mesh, {"experts": XPUExpertParallel()})
```

---

## Files Modified

### Core Changes

| File | Change |
|------|--------|
| `expert_parallel_xpu.py` | Optimized `all_gather` implementation for dispatch/combine |
| `torchtitan/models/llama4/infra/parallelize.py` | Import hook for `XPUExpertParallel` |
| `torchtitan/models/llama4/__init__.py` | Added `debugmodel_12_rank` flavor with dim=384 |

### FSDP Compatibility

| File | Change |
|------|--------|
| `torchtitan/models/llama4/infra/parallelize.py` | Monkey-patch `_get_gradient_divide_factors` to force `ReduceOp.SUM` (XCCL doesn't support `PREMUL_SUM`) |

### Configuration Files

| File | Purpose |
|------|---------|
| `llama4_ep12_triton_xpu.toml` | EP=12 with Triton MoE kernel |
| `llama4_ep12_fallback_xpu.toml` | EP=12 with PyTorch MoE (no Triton) |
| `llama4_dense12_xpu.toml` | Dense baseline (MoE disabled) |
| `llama4_1b_dense_xpu_compile_bs16.toml` | High-utilization 1B Dense config |
| `llama4_3b_moe_hsdp_ep6_xpu_compile_bs8.toml` | High-utilization 3B HSDP config |
| `llama4_3b_total_moe_hsdp_ep6_xpu_compile.toml` | Iso-Parameter 3B MoE config |
| `scripts/run_1b_comparison.sh` | Consolidated 1B benchmark script |
| `scripts/run_3b_comparison.sh` | Consolidated 3B benchmark script |

---

### Running Benchmarks

Use the consolidated `run_deepseek_1b.sh` script to run benchmarks. This script automatically handles module loading (`frameworks`, `mpich`, `py-mpi4py`) and PYTHONPATH setup.

#### DeepSeek 1B MoE (No EP)
```bash
bash scripts/run_deepseek_1b.sh moe_noep
```

#### DeepSeek 1B EP=12
```bash
bash scripts/run_deepseek_1b.sh ep12
```

#### Llama4 Large EP=12 (Max Batch)
```bash
bash scripts/run_deepseek_1b.sh llama4_large_ep12_maxbatch
```

### Requirements
- Aurora compute node with 12 XPU tiles (e.g. `qsub -I -l select=1 ...`)
- System `frameworks` module (PyTorch 2.8+)
- `mpich` and `py-mpi4py` modules

---

## Testing

```bash
# Single rank test (on compute node)
python scripts/test_xpu_ep_ops.py

# Multi-rank test
mpiexec -n 4 python scripts/test_xpu_ep_ops.py
```

---

## Performance Optimization & Best Practices

Based on our benchmarking on Aurora XPU, we recommend the following for maximum performance and stability:

### 1. Enable `torch.compile`
Compilation is highly effective on XPU:
- **Throughput**: 15% - 35% speedup.
- **Memory**: 30% - 40% reduction in footprint.
- **Stability**: Prevents OOM in memory-intensive MoE configurations (e.g., EP=12).
*Implementation:* Set `enable = true` in the `[compile]` section of your `.toml` config.

### 2. Maximize Batch Size
The memory headroom provided by `torch.compile` should be used to increase batch size.
- Aim for **90%+ XPU memory utilization** to maximize MFU.
- 1B models can often scale to BS=16 or BS=20; 3B models to BS=12.

### 3. Node Topology & CPU Binding
Aurora's dual-socket, multi-tile architecture requires careful rank placement.
- **Recommendation**: Avoid manual `numactl` wrappers (can cause instability).
- **Correct Method**: Use `mpiexec` affinity lists.
    ```bash
    # Example for 12 ranks per node (6 per socket)
    CPU_BIND="verbose,list:4-7:8-11:12-15:16-19:20-23:24-27:56-59:60-63:64-67:68-71:72-75:76-79"
    mpiexec ... --cpu-bind=${CPU_BIND} --envall ...
    ```
- **Stability**: Correct binding prevents "Out of Resources" errors in high-load MoE runs.

---

## Known Limitations

1. **Triton MoE Kernel**: Does not work on XPU (Triton driver issue). Use `use_triton_moe = false` in config.
2. **`all_to_all`**: Not supported by XCCL - this module is the workaround.
3. **Performance**: ~40% slower than Dense due to collective overhead.

## Future Work

- Implement fused `all_to_all` kernel for XCCL
- Optimize `all_gather` with async prefetching
- Support Expert Tensor Parallelism (ETP)

## High MFU Validation (8k Tokens/Rank)

To validate the environment's capability to reach high MFU comparable to Llama 3 8B baselines (~23-26%), we benchmarked Llama 4 MoE configurations at high arithmetic intensity using a cached local dataset (to bypass streaming latency).

### Results Comparison

| Configuration | Batch | Seq | Tokens/Rank | AC Mode | Status | MFU | Memory |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Llama 3 8B (Baseline)** | 4 | 2048 | 8,192 | **Selective** | Success | **26.22%** | 58 GB |
| **Llama 4 MoE (No EP)** | 4 | 2048 | 8,192 | **Full** | Success | **15.18%** | 50.1 GB |
| **Llama 4 MoE (HSDP EP=6)** | 4 | 2048 | 8,192 | **Full** | Success | **15.44%** | **44.7 GB** |
| **Llama 4 MoE (EP=12)** | 4 | 2048 | 8,192 | Full | **Failed** | N/A | OOM |

### Analysis

1.  **Llama 3 8B Performance (26%):** The high MFU is enabled by **Selective Activation Checkpointing (AC)**, which minimizes recomputation overhead. The jump from 23% (initial) to 26% is due to running longer (120 steps vs 10) and using a **cached local dataset** which eliminated dataloading latency.
2.  **Llama 4 MoE Bottleneck (15%):** The MoE model (~50B params) is too large to fit in XPU memory with Selective AC at this batch size. We were forced to use **Full AC**, which requires re-running the forward pass during backprop. This massive compute overhead caps the MFU at ~15%.
3.  **EP vs No-EP Convergence:** Unlike low-batch runs where EP was 2.6x faster, at high batch sizes/Full AC, the performance converges. The system becomes compute-bound (by AC recomputation), making the communication efficiency of EP less impactful than the raw compute cost.
4.  **HSDP Efficiency:** Hybrid Sharded Data Parallel (HSDP) with EP=6 (Replicate=2) successfully reduced memory usage to **44.7 GB** (vs 50 GB for No-EP), preventing the OOM seen with EP=12.

### Critical Fixes Needed

To run these benchmarks, several code fixes were applied:

1.  **Activation Checkpointing (`activation_checkpoint.py`):**
    *   **Problem:** `TypeError: wrapper() got an unexpected keyword argument 'early_stop'`
    *   **Fix:** Patched `torchtitan/distributed/activation_checkpoint.py` to remove the `early_stop` argument from `ptd_checkpoint_wrapper` calls, as it's not supported in the installed PyTorch version.

2.  **HSDP Mesh Slicing (`parallelize.py`):**
    *   **Problem:** `RuntimeError: Cannot create a submesh from a submesh` when using HSDP meshes.
    *   **Fix:** Patched `torchtitan/models/llama4/infra/parallelize.py` to avoid slicing the mesh object directly for size checks. Instead, looking up the dimension index and using `mesh.size(idx)`.
