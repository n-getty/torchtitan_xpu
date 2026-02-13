# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3
"""
Test script to verify if PREMUL_SUM reduce operation is supported on XPU/XCCL.

PREMUL_SUM is a fused operation that multiplies by a scalar before summing,
used by FSDP for gradient reduction. XCCL does not support this operation.

Run with mpiexec (requires at least 2 ranks):
    mpiexec -n 2 -ppn 2 --envall python tests/test_premul_sum_support.py

Expected result on Aurora XPU with XCCL: NOT SUPPORTED
(This is why we patch FSDP to use SUM reduction instead)
"""

import os
import sys

import torch
import torch.distributed as dist


def is_xpu_available() -> bool:
    """Check if XPU device is available."""
    try:
        import intel_extension_for_pytorch  # noqa: F401

        return torch.xpu.is_available()
    except ImportError:
        return False


def get_rank_and_world_size():
    """
    Get rank and world size from various MPI environment variables.

    Supports: PALS (Aurora), PMI, PMIX, SLURM, and standard RANK/WORLD_SIZE.
    """
    # Try PALS (Aurora/HPE Cray) - check PALS_RANKID and PALS_LOCAL_SIZE
    if "PALS_RANKID" in os.environ:
        rank = int(os.environ["PALS_RANKID"])
        # For single-node, PALS_LOCAL_SIZE == world_size
        world_size = int(os.environ.get("PALS_LOCAL_SIZE", "1"))
        return rank, world_size

    # Try PMIX (used by PALS under the hood)
    if "PMIX_RANK" in os.environ:
        rank = int(os.environ["PMIX_RANK"])
        world_size = int(os.environ.get("PALS_LOCAL_SIZE", "1"))
        return rank, world_size

    # Try PMI
    if "PMI_RANK" in os.environ:
        rank = int(os.environ["PMI_RANK"])
        world_size = int(os.environ.get("PMI_SIZE", "1"))
        return rank, world_size

    # Try SLURM
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ.get("SLURM_NTASKS", "1"))
        return rank, world_size

    # Try standard env vars (set by torchrun, etc.)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size


def setup_distributed():
    """Initialize distributed environment for testing."""
    os.environ["CCL_PROCESS_LAUNCHER"] = "none"
    os.environ["CCL_ATL_TRANSPORT"] = "ofi"

    rank, world_size = get_rank_and_world_size()

    # Require at least 2 ranks for meaningful reduce test
    if world_size < 2:
        print("=" * 60)
        print("ERROR: PREMUL_SUM test requires at least 2 ranks")
        print(f"Detected: rank={rank}, world_size={world_size}")
        print(
            "Run with: mpiexec -n 2 -ppn 2 --envall python tests/test_premul_sum_support.py"
        )
        print("=" * 60)
        sys.exit(1)

    # Set MASTER_ADDR and MASTER_PORT if not already set
    if "MASTER_ADDR" not in os.environ:
        import socket

        hostname = socket.gethostname()
        os.environ["MASTER_ADDR"] = hostname
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

    xpu_available = is_xpu_available()

    if xpu_available:
        backend = "xccl"
        device = torch.device(f"xpu:{rank % torch.xpu.device_count()}")
        torch.xpu.set_device(device)
    else:
        backend = "gloo"
        device = torch.device("cpu")

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    return device, backend, rank, world_size


def test_basic_sum(device, rank, world_size):
    """
    Test basic SUM reduce operation (sanity check).

    This should always work on any backend.
    """
    print(f"\n[Rank {rank}] Testing basic SUM reduce (sanity check)...")

    tensor = torch.full((10, 10), float(rank + 1), device=device)
    original = tensor.clone()

    try:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        if device.type == "xpu":
            torch.xpu.synchronize()

        # Expected: sum of (1 + 2 + ... + world_size) = world_size * (world_size + 1) / 2
        expected_val = sum(range(1, world_size + 1))
        expected = torch.full_like(tensor, expected_val)

        if torch.allclose(tensor, expected):
            return True, f"SUM works correctly (result: {tensor[0, 0].item()})"
        else:
            return (
                False,
                f"SUM produced wrong result: {tensor[0, 0].item()} vs expected {expected_val}",
            )

    except Exception as e:
        return False, f"SUM failed: {type(e).__name__}: {e}"


def test_premul_sum(device, rank, world_size):
    """
    Test PREMUL_SUM reduce operation.

    PREMUL_SUM multiplies each tensor by a scalar (1/world_size typically)
    before summing. This is used by FSDP to average gradients.
    """
    print(f"\n[Rank {rank}] Testing PREMUL_SUM reduce...")

    tensor = torch.full((10, 10), float(rank + 1), device=device)

    try:
        # Check if PREMUL_SUM exists
        if not hasattr(dist.ReduceOp, "PREMUL_SUM"):
            return False, "PREMUL_SUM not available in torch.distributed.ReduceOp"

        # Create the premul_sum operation with scale factor
        # This mimics what FSDP does for gradient averaging
        scale_factor = 1.0 / world_size

        # The _make_nccl_premul_sum function creates the operation
        try:
            from torch.distributed.distributed_c10d import _make_nccl_premul_sum

            premul_op = _make_nccl_premul_sum(scale_factor)
        except ImportError:
            # Older PyTorch versions
            return False, "_make_nccl_premul_sum not available"

        dist.all_reduce(tensor, op=premul_op)

        if device.type == "xpu":
            torch.xpu.synchronize()

        # Expected: (1 + 2 + ... + world_size) / world_size
        expected_val = sum(range(1, world_size + 1)) / world_size
        expected = torch.full_like(tensor, expected_val)

        if torch.allclose(tensor, expected, rtol=1e-4, atol=1e-4):
            return (
                True,
                f"PREMUL_SUM works correctly (result: {tensor[0, 0].item():.4f})",
            )
        else:
            return (
                False,
                f"PREMUL_SUM produced wrong result: {tensor[0, 0].item():.4f} vs expected {expected_val:.4f}",
            )

    except NotImplementedError as e:
        return False, f"NotImplementedError: {e}"
    except RuntimeError as e:
        error_msg = str(e)
        if "not support" in error_msg.lower() or "premul" in error_msg.lower():
            return False, f"RuntimeError (not supported): {error_msg}"
        else:
            return False, f"RuntimeError: {error_msg}"
    except AttributeError as e:
        return False, f"AttributeError: {e}"
    except Exception as e:
        return False, f"Unexpected error ({type(e).__name__}): {e}"


def test_avg_reduce(device, rank, world_size):
    """
    Test AVG reduce operation as an alternative to PREMUL_SUM.

    Some backends support AVG directly.
    """
    print(f"\n[Rank {rank}] Testing AVG reduce (alternative to PREMUL_SUM)...")

    tensor = torch.full((10, 10), float(rank + 1), device=device)

    try:
        if not hasattr(dist.ReduceOp, "AVG"):
            return False, "AVG not available in torch.distributed.ReduceOp"

        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)

        if device.type == "xpu":
            torch.xpu.synchronize()

        # Expected: (1 + 2 + ... + world_size) / world_size
        expected_val = sum(range(1, world_size + 1)) / world_size
        expected = torch.full_like(tensor, expected_val)

        if torch.allclose(tensor, expected, rtol=1e-4, atol=1e-4):
            return True, f"AVG works correctly (result: {tensor[0, 0].item():.4f})"
        else:
            return (
                False,
                f"AVG produced wrong result: {tensor[0, 0].item():.4f} vs expected {expected_val:.4f}",
            )

    except NotImplementedError as e:
        return False, f"NotImplementedError: {e}"
    except RuntimeError as e:
        return False, f"RuntimeError: {e}"
    except Exception as e:
        return False, f"Unexpected error ({type(e).__name__}): {e}"


def test_manual_premul_sum_workaround(device, rank, world_size):
    """
    Test the manual workaround: scale locally, then SUM.

    This is what FSDP does when force_sum_reduction_for_comms=True.
    """
    print(f"\n[Rank {rank}] Testing manual PREMUL_SUM workaround (scale + SUM)...")

    tensor = torch.full((10, 10), float(rank + 1), device=device)
    scale_factor = 1.0 / world_size

    try:
        # Scale locally first
        tensor.mul_(scale_factor)

        # Then sum
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        if device.type == "xpu":
            torch.xpu.synchronize()

        # Expected: (1 + 2 + ... + world_size) / world_size
        expected_val = sum(range(1, world_size + 1)) / world_size
        expected = torch.full_like(tensor, expected_val)

        if torch.allclose(tensor, expected, rtol=1e-4, atol=1e-4):
            return (
                True,
                f"Workaround works correctly (result: {tensor[0, 0].item():.4f})",
            )
        else:
            return (
                False,
                f"Workaround produced wrong result: {tensor[0, 0].item():.4f} vs expected {expected_val:.4f}",
            )

    except Exception as e:
        return False, f"Workaround failed: {type(e).__name__}: {e}"


def main():
    print("=" * 70)
    print("Testing PREMUL_SUM Support on XPU/XCCL")
    print("=" * 70)

    device, backend, rank, world_size = setup_distributed()

    if rank == 0:
        print(f"\nConfiguration:")
        print(f"  Backend: {backend}")
        print(f"  Device: {device}")
        print(f"  World size: {world_size}")
        print(f"  XPU available: {is_xpu_available()}")

    dist.barrier()

    results = {}

    # Test 1: Basic SUM (sanity check)
    supported, message = test_basic_sum(device, rank, world_size)
    results["basic_SUM"] = (supported, message)

    dist.barrier()

    # Test 2: PREMUL_SUM
    supported, message = test_premul_sum(device, rank, world_size)
    results["PREMUL_SUM"] = (supported, message)

    dist.barrier()

    # Test 3: AVG (alternative)
    supported, message = test_avg_reduce(device, rank, world_size)
    results["AVG_reduce"] = (supported, message)

    dist.barrier()

    # Test 4: Manual workaround
    supported, message = test_manual_premul_sum_workaround(device, rank, world_size)
    results["manual_workaround"] = (supported, message)

    dist.barrier()

    # Report results (only rank 0)
    if rank == 0:
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

        premul_supported = results["PREMUL_SUM"][0]

        for test_name, (supported, message) in results.items():
            status = "[SUPPORTED]" if supported else "[NOT SUPPORTED]"
            print(f"\n{test_name}:")
            print(f"  Status: {status}")
            print(f"  Details: {message}")

        print("\n" + "=" * 70)
        if premul_supported:
            print("OVERALL: PREMUL_SUM IS SUPPORTED on this backend")
        else:
            print("OVERALL: PREMUL_SUM is NOT SUPPORTED on this backend")
            if results["manual_workaround"][0]:
                print("         Workaround (scale + SUM) is available and working")
            print("         FSDP should use force_sum_reduction_for_comms=True")
        print("=" * 70)

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
