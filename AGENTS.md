# Torchtitan XPU - Agent Development Guide

This repo optimizes [torchtitan](https://github.com/pytorch/torchtitan) for Intel Aurora XPU, focusing on Expert Parallelism (EP) for MoE models.

## Project Structure

```
torchtitan_xpu/
├── scripts/                  # Benchmark launch scripts
├── tests/                    # XPU/XCCL compatibility tests
├── TESTS.md                  # Test results and benchmark documentation
├── torchtitan/               # Upstream torchtitan (with XPU patches)
│   ├── torchtitan/           # Main library code
│   │   ├── distributed/      # Parallelism & EP (expert_parallel_xpu.py)
│   │   ├── models/           # Model definitions (llama3, llama4, deepseek_v3)
│   │   └── train.py          # Training entry point
│   └── tests/                # Test suite
└── context/                  # Reference documentation
```

## Build & Environment

```bash
# On Aurora compute node
module load frameworks mpich py-mpi4py

# Install dev dependencies (in torchtitan/)
pip install -e ".[dev]"

# Required environment variables
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export PYTHONPATH=$(pwd):$(pwd)/torchtitan:$PYTHONPATH
```

## Test Commands

```bash
# Run all unit tests
pytest -s torchtitan/tests/unit_tests/

# Run single test file
pytest -s torchtitan/tests/unit_tests/test_parallel_dims.py

# Run specific test function
pytest -s torchtitan/tests/unit_tests/test_job_config.py::TestJobConfig::test_command_line_args

# Run integration tests (requires GPUs)
python -m tests.integration_tests.run_tests <output_dir> --test_suite features --ngpu 12

# Run single integration test
python -m tests.integration_tests.run_tests <output_dir> --test_name gradient_accumulation --ngpu 2

# XPU EP operations test
mpiexec -n 12 -ppn 12 --envall python scripts/test_xpu_ep_ops.py

# XPU/XCCL compatibility tests (see TESTS.md for details)
mpiexec -n 2 -ppn 2 --envall python tests/test_xccl_ops_support.py
mpiexec -n 2 -ppn 2 --envall python tests/test_all_to_all_support.py
python tests/test_triton_moe_support.py
```

## Lint & Format

```bash
# Run all pre-commit checks (in torchtitan/)
pre-commit run --all-files

# Individual tools
ufmt format .                    # Format with black + usort
flake8 --config=.flake8 .        # Lint
codespell --toml pyproject.toml  # Spell check
```

## Training Commands

```bash
# Single-node 12-tile benchmark
bash scripts/run_deepseek_1b.sh ep12

# Generic launch pattern
mpiexec -n 12 -ppn 12 --envall python -u torchtitan/mpi_train_wrapper.py \
    --job.config_file torchtitan/torchtitan/models/llama4/train_configs/llama4_1b_moe_ep12_xpu.toml
```

## Code Style Guidelines

### Formatting

- **Formatter**: black (line length 88) + usort for imports
- **Linter**: flake8 with flake8-bugbear, pep8-naming, torchfix
- **Type Checker**: pyrefly (configured in pyproject.toml)

### Imports

Order (enforced by usort): stdlib, third-party, local. Example:

```python
# Standard library
from abc import ABC, abstractmethod
from typing import Optional

# Third-party
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed.tensor import DeviceMesh, DTensor, Shard

# Local
from torchtitan.distributed.expert_parallel import ExpertParallel
```

### Naming Conventions

- Classes: `PascalCase` (e.g., `XPUExpertParallel`, `BaseExpertParallel`)
- Functions/methods: `snake_case` (e.g., `_partition_fn`, `_token_dispatch`)
- Private methods: prefix with `_` (e.g., `_all_gather_dispatch`)
- Constants: `UPPER_SNAKE_CASE`
- Config keys in TOML: `snake_case`

### Type Annotations

- Required for public APIs, optional for internal/private
- Use `torch.Tensor` or `Tensor` (imported from torch)
- Use `Optional[T]` for nullable, `tuple[A, B]` for tuples

```python
def _token_dispatch(
    self, mod: nn.Module, inputs: tuple, device_mesh: DeviceMesh
) -> tuple[Tensor, Tensor]:
```

### Error Handling

- Use assertions for internal invariants with descriptive messages
- Raise explicit exceptions for user-facing errors

```python
assert current_processed == processed_counts[ep_rank], \
    f"Rank {ep_rank} processed mismatch: expected {processed_counts[ep_rank]}, got {current_processed}"
```

### License Headers

All Python files must include the BSD license header (enforced by pre-commit):

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
```

## XPU-Specific Patterns

### Device Detection

```python
def is_xpu_available() -> bool:
    try:
        import intel_extension_for_pytorch  # noqa: F401
        return torch.xpu.is_available()
    except ImportError:
        return False
```

### Distributed Backend

**CRITICAL**: Use `xccl` backend, NOT `nccl`:

```python
torch.distributed.init_process_group(backend="xccl")
```

### Expert Parallelism

**Status (Feb 2026)**: Native `ExpertParallel` with `all_to_all` now works on XCCL.
The shape mismatch issue has been fixed in the upstream code.

```python
# Default: uses native ExpertParallel (all_to_all-based)
from torchtitan_xpu import get_expert_parallel_class
experts_plan = get_expert_parallel_class()()  # Returns native ExpertParallel

# Legacy: force all_gather-based workaround (if needed):
export TORCHTITAN_XPU_FORCE_ALLGATHER_EP=1
```

### FSDP Gradient Reduction

XCCL doesn't support `PREMUL_SUM`. Force `SUM` reduction:

```python
# Monkey-patch in parallelize.py
torch.distributed.fsdp._fully_shard._fsdp_collectives._get_gradient_divide_factors = patched_fn
```

## Key Files to Modify

| Task                     | File                                                       |
| ------------------------ | ---------------------------------------------------------- |
| Add EP implementation    | `torchtitan/torchtitan/distributed/expert_parallel_xpu.py` |
| Model parallelization    | `torchtitan/models/<model>/infra/parallelize.py`           |
| New model config         | `torchtitan/models/<model>/train_configs/*.toml`           |
| Activation checkpointing | `torchtitan/distributed/activation_checkpoint.py`          |
| Parallel dimensions      | `torchtitan/distributed/parallel_dims.py`                  |

## Known Limitations

1. **Activation Checkpointing + EP**: AC causes unbounded memory growth with EP due to autograd tensor accumulation in `all_to_all_single_autograd`. **Must use `mode = "none"` for EP configs.** Tested and confirmed: ALL AC modes fail (full, selective-layer, selective-op).
2. **Triton MoE Kernel**: Works on XPU but 2.3x slower than PyTorch default - use `use_triton_moe = false`
3. **all_to_all**: XCCL now supports at collective level, native `ExpertParallel` now works (shape fix applied)
4. **PREMUL_SUM**: Not supported (NCCL-specific API) - requires FSDP patch
5. **torch.compile**: Works and recommended for EP - provides memory efficiency

## Performance Optimization Checklist

- [ ] Enable `torch.compile` (`[compile] enable = true` in TOML)
- [ ] **Disable AC for EP** (`mode = "none"` - REQUIRED to avoid memory leaks)
- [ ] Maximize batch size (aim for 90%+ XPU memory utilization)
- [ ] Use HSDP (EP=6, Replicate=2) for memory efficiency
- [ ] Set `CCL_WORKER_COUNT=4` for better throughput
