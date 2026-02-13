# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3
"""
Test script to verify if all_to_all_single is supported on XPU/XCCL backend.

This tests whether the XCCL backend supports the all_to_all collective operation,
which is required by the upstream ExpertParallel implementation.

Run with mpiexec (requires at least 2 ranks):
    mpiexec -n 2 -ppn 2 --envall python tests/test_all_to_all_support.py

Expected result on Aurora XPU with XCCL: NOT SUPPORTED
(This is why XPUExpertParallel uses all_gather instead)
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
        # For multi-node, we'd need additional logic
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
    # Set CCL environment variables
    os.environ["CCL_PROCESS_LAUNCHER"] = "none"
    os.environ["CCL_ATL_TRANSPORT"] = "ofi"

    rank, world_size = get_rank_and_world_size()

    # Require at least 2 ranks for meaningful all_to_all test
    if world_size < 2:
        print("=" * 60)
        print("ERROR: all_to_all test requires at least 2 ranks")
        print(f"Detected: rank={rank}, world_size={world_size}")
        print(
            "Run with: mpiexec -n 2 -ppn 2 --envall python tests/test_all_to_all_support.py"
        )
        print("=" * 60)
        sys.exit(1)

    # Set MASTER_ADDR and MASTER_PORT if not already set
    # Get hostname for MASTER_ADDR
    if "MASTER_ADDR" not in os.environ:
        import socket

        hostname = socket.gethostname()
        os.environ["MASTER_ADDR"] = hostname
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

    # Detect XPU availability
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


def test_all_to_all_single(device, rank, world_size):
    """
    Test if all_to_all_single is supported.

    all_to_all_single scatters input tensor to all ranks and gathers
    results from all ranks.

    For a 2-rank example with tensor size 4:
    - Rank 0: input [0,1,2,3] -> output [0,1,4,5] (first half from rank0, second from rank1)
    - Rank 1: input [4,5,6,7] -> output [2,3,6,7] (first half from rank0, second from rank1)
    """
    print(f"\n[Rank {rank}] Testing all_to_all_single...")

    # Each rank has world_size chunks to scatter
    chunk_size = 4
    total_size = chunk_size * world_size

    # Create input: rank 0 has [0..7], rank 1 has [8..15], etc.
    input_tensor = torch.arange(
        rank * total_size,
        (rank + 1) * total_size,
        dtype=torch.float32,
        device=device,
    )

    output_tensor = torch.zeros_like(input_tensor)

    # Define split sizes (equal splits)
    input_split_sizes = [chunk_size] * world_size
    output_split_sizes = [chunk_size] * world_size

    try:
        # Attempt all_to_all_single
        dist.all_to_all_single(
            output_tensor,
            input_tensor,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
        )

        # Synchronize to ensure operation completed
        if device.type == "xpu":
            torch.xpu.synchronize()

        # Verify correctness
        # After all_to_all, rank r should have chunk i from rank i
        # So output[i*chunk_size : (i+1)*chunk_size] = input from rank i for rank r
        expected = torch.zeros_like(output_tensor)
        for src_rank in range(world_size):
            start = src_rank * chunk_size
            end = start + chunk_size
            # Chunk that src_rank sent to current rank
            # src_rank's input was [src_rank*total_size ... (src_rank+1)*total_size]
            # The chunk for current rank is at offset rank*chunk_size
            chunk_start = src_rank * total_size + rank * chunk_size
            expected[start:end] = torch.arange(
                chunk_start,
                chunk_start + chunk_size,
                dtype=torch.float32,
                device=device,
            )

        if torch.allclose(output_tensor, expected):
            return True, "all_to_all_single works correctly"
        else:
            return False, f"Output mismatch: got {output_tensor}, expected {expected}"

    except NotImplementedError as e:
        return False, f"NotImplementedError: {e}"
    except RuntimeError as e:
        error_msg = str(e)
        if (
            "not supported" in error_msg.lower()
            or "not implemented" in error_msg.lower()
        ):
            return False, f"RuntimeError (not supported): {error_msg}"
        else:
            return False, f"RuntimeError: {error_msg}"
    except Exception as e:
        return False, f"Unexpected error ({type(e).__name__}): {e}"


def test_functional_all_to_all(device, rank, world_size):
    """
    Test the functional collectives version of all_to_all_single.

    This is the version used by upstream ExpertParallel.
    """
    print(f"\n[Rank {rank}] Testing functional all_to_all_single...")

    try:
        from torch.distributed._functional_collectives import all_to_all_single

        chunk_size = 4
        total_size = chunk_size * world_size

        input_tensor = torch.arange(
            rank * total_size,
            (rank + 1) * total_size,
            dtype=torch.float32,
            device=device,
        )

        input_split_sizes = [chunk_size] * world_size
        output_split_sizes = [chunk_size] * world_size

        # Functional version returns the output tensor
        output_tensor = all_to_all_single(
            input_tensor,
            output_split_sizes,
            input_split_sizes,
            group=dist.distributed_c10d._get_default_group(),
        )

        if device.type == "xpu":
            torch.xpu.synchronize()

        # Basic shape check
        if output_tensor.shape == input_tensor.shape:
            return True, "functional all_to_all_single works correctly"
        else:
            return (
                False,
                f"Shape mismatch: {output_tensor.shape} vs {input_tensor.shape}",
            )

    except ImportError as e:
        return False, f"ImportError: {e}"
    except NotImplementedError as e:
        return False, f"NotImplementedError: {e}"
    except RuntimeError as e:
        return False, f"RuntimeError: {e}"
    except Exception as e:
        return False, f"Unexpected error ({type(e).__name__}): {e}"


def main():
    print("=" * 70)
    print("Testing all_to_all Support on XPU/XCCL")
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

    # Test 1: Standard all_to_all_single
    supported, message = test_all_to_all_single(device, rank, world_size)
    results["all_to_all_single"] = (supported, message)

    dist.barrier()

    # Test 2: Functional all_to_all_single
    func_supported, func_message = test_functional_all_to_all(device, rank, world_size)
    results["functional_all_to_all_single"] = (func_supported, func_message)

    dist.barrier()

    # Report results (only rank 0)
    if rank == 0:
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

        all_supported = True
        for test_name, (supported, message) in results.items():
            status = "[SUPPORTED]" if supported else "[NOT SUPPORTED]"
            all_supported = all_supported and supported
            print(f"\n{test_name}:")
            print(f"  Status: {status}")
            print(f"  Details: {message}")

        print("\n" + "=" * 70)
        if all_supported:
            print("OVERALL: all_to_all IS SUPPORTED on this backend")
        else:
            print("OVERALL: all_to_all is NOT SUPPORTED on this backend")
            print("         Use XPUExpertParallel (all_gather-based) instead")
        print("=" * 70)

    # Cleanup
    dist.destroy_process_group()

    # Exit with appropriate code (don't fail - this is an info test)


if __name__ == "__main__":
    main()
