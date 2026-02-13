# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3
"""
Comprehensive test script for XCCL distributed operations support.

This tests all major distributed collective operations to verify
what is supported by the XCCL backend on Intel XPU.

Run with mpiexec (requires at least 2 ranks):
    mpiexec -n 2 -ppn 2 --envall python tests/test_xccl_ops_support.py

Expected result on Aurora XPU with XCCL (25.190.0 frameworks):
    Most operations are SUPPORTED including all_to_all!
"""

import os
import socket
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
    """Get rank and world size from MPI environment variables."""
    if "PALS_RANKID" in os.environ:
        rank = int(os.environ["PALS_RANKID"])
        world_size = int(os.environ.get("PALS_LOCAL_SIZE", "1"))
        return rank, world_size
    if "PMIX_RANK" in os.environ:
        rank = int(os.environ["PMIX_RANK"])
        world_size = int(os.environ.get("PALS_LOCAL_SIZE", "1"))
        return rank, world_size
    if "PMI_RANK" in os.environ:
        rank = int(os.environ["PMI_RANK"])
        world_size = int(os.environ.get("PMI_SIZE", "1"))
        return rank, world_size
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size


def setup_distributed():
    """Initialize distributed environment."""
    os.environ["CCL_PROCESS_LAUNCHER"] = "none"
    os.environ["CCL_ATL_TRANSPORT"] = "ofi"

    rank, world_size = get_rank_and_world_size()

    if world_size < 2:
        print("=" * 60)
        print("ERROR: This test requires at least 2 ranks")
        print(f"Detected: rank={rank}, world_size={world_size}")
        print(
            "Run with: mpiexec -n 2 -ppn 2 --envall python tests/test_xccl_ops_support.py"
        )
        print("=" * 60)
        sys.exit(1)

    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = socket.gethostname()
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29503"

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


def sync_device(device):
    """Synchronize device."""
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def test_operation(name, test_fn, device, rank, world_size):
    """Run a test and return (success, message)."""
    try:
        result = test_fn(device, rank, world_size)
        sync_device(device)
        return True, result
    except NotImplementedError as e:
        return False, f"NotImplementedError: {e}"
    except RuntimeError as e:
        return False, f"RuntimeError: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# =============================================================================
# Test Functions for Each Operation
# =============================================================================


def test_broadcast(device, rank, world_size):
    tensor = torch.full((4,), float(rank * 10), device=device)
    dist.broadcast(tensor, src=0)
    expected = 0.0  # Broadcast from rank 0
    assert tensor[0].item() == expected, f"Expected {expected}, got {tensor[0].item()}"
    return f"result={tensor[0].item()}"


def test_all_reduce_sum(device, rank, world_size):
    tensor = torch.full((4,), float(rank + 1), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected = sum(range(1, world_size + 1))
    assert tensor[0].item() == expected, f"Expected {expected}, got {tensor[0].item()}"
    return f"result={tensor[0].item()}"


def test_all_reduce_max(device, rank, world_size):
    tensor = torch.full((4,), float(rank + 1), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    expected = world_size
    assert tensor[0].item() == expected, f"Expected {expected}, got {tensor[0].item()}"
    return f"result={tensor[0].item()}"


def test_all_reduce_min(device, rank, world_size):
    tensor = torch.full((4,), float(rank + 1), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    expected = 1.0
    assert tensor[0].item() == expected, f"Expected {expected}, got {tensor[0].item()}"
    return f"result={tensor[0].item()}"


def test_all_reduce_product(device, rank, world_size):
    tensor = torch.full((4,), float(rank + 1), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.PRODUCT)
    expected = 1.0
    for i in range(1, world_size + 1):
        expected *= i
    assert tensor[0].item() == expected, f"Expected {expected}, got {tensor[0].item()}"
    return f"result={tensor[0].item()}"


def test_all_reduce_avg(device, rank, world_size):
    tensor = torch.full((4,), float(rank + 1), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    expected = sum(range(1, world_size + 1)) / world_size
    assert abs(tensor[0].item() - expected) < 1e-4, (
        f"Expected {expected}, got {tensor[0].item()}"
    )
    return f"result={tensor[0].item():.2f}"


def test_all_gather(device, rank, world_size):
    local = torch.full((4,), float(rank), device=device)
    gathered = [torch.zeros_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    for i, t in enumerate(gathered):
        assert t[0].item() == float(i), f"Rank {i} mismatch"
    return "all ranks gathered correctly"


def test_all_gather_into_tensor(device, rank, world_size):
    local = torch.full((4,), float(rank), device=device)
    output = torch.zeros(4 * world_size, device=device)
    dist.all_gather_into_tensor(output, local)
    return f"output_shape={output.shape}"


def test_reduce_scatter(device, rank, world_size):
    input_list = [
        torch.full((4,), float(i + rank * 10), device=device) for i in range(world_size)
    ]
    output = torch.zeros(4, device=device)
    dist.reduce_scatter(output, input_list, op=dist.ReduceOp.SUM)
    return f"output={output[0].item()}"


def test_reduce_scatter_tensor(device, rank, world_size):
    input_tensor = (
        torch.arange(4 * world_size, dtype=torch.float32, device=device) + rank * 100
    )
    output = torch.zeros(4, device=device)
    dist.reduce_scatter_tensor(output, input_tensor, op=dist.ReduceOp.SUM)
    return f"output={output[0].item()}"


def test_all_to_all(device, rank, world_size):
    chunk_size = 4
    total_size = chunk_size * world_size
    input_tensor = torch.arange(
        rank * total_size,
        (rank + 1) * total_size,
        dtype=torch.float32,
        device=device,
    )
    output_tensor = torch.zeros_like(input_tensor)
    input_split = [chunk_size] * world_size
    output_split = [chunk_size] * world_size
    dist.all_to_all_single(
        output_tensor,
        input_tensor,
        output_split_sizes=output_split,
        input_split_sizes=input_split,
    )
    return f"output_shape={output_tensor.shape}"


def test_scatter(device, rank, world_size):
    if rank == 0:
        scatter_list = [
            torch.full((4,), float(i), device=device) for i in range(world_size)
        ]
    else:
        scatter_list = None
    output = torch.zeros(4, device=device)
    dist.scatter(output, scatter_list, src=0)
    assert output[0].item() == float(rank), f"Expected {rank}, got {output[0].item()}"
    return f"received={output[0].item()}"


def test_gather(device, rank, world_size):
    local = torch.full((4,), float(rank), device=device)
    if rank == 0:
        gather_list = [torch.zeros(4, device=device) for _ in range(world_size)]
    else:
        gather_list = None
    dist.gather(local, gather_list, dst=0)
    if rank == 0:
        for i, t in enumerate(gather_list):
            assert t[0].item() == float(i), f"Rank {i} mismatch"
    return "gathered correctly" if rank == 0 else "sent"


def test_barrier(device, rank, world_size):
    dist.barrier()
    return "synchronized"


def test_send_recv(device, rank, world_size):
    if rank == 0:
        tensor = torch.full((4,), 42.0, device=device)
        dist.send(tensor, dst=1)
        return "sent 42.0 to rank 1"
    else:
        tensor = torch.zeros(4, device=device)
        dist.recv(tensor, src=0)
        assert tensor[0].item() == 42.0, f"Expected 42.0, got {tensor[0].item()}"
        return f"received {tensor[0].item()}"


def test_all_reduce_coalesced(device, rank, world_size):
    tensors = [torch.full((4,), float(rank + i), device=device) for i in range(3)]
    dist.all_reduce_coalesced(tensors, op=dist.ReduceOp.SUM)
    return f"coalesced {len(tensors)} tensors"


def test_premul_sum(device, rank, world_size):
    """Test PREMUL_SUM (NCCL-specific, expected to fail on XCCL)."""
    from torch.distributed.distributed_c10d import _make_nccl_premul_sum

    tensor = torch.full((4,), float(rank + 1), device=device)
    scale = 1.0 / world_size
    premul_op = _make_nccl_premul_sum(scale)
    dist.all_reduce(tensor, op=premul_op)
    return f"result={tensor[0].item():.4f}"


def main():
    print("=" * 70)
    print("XCCL Distributed Operations Support Test")
    print("=" * 70)

    device, backend, rank, world_size = setup_distributed()

    if rank == 0:
        print(f"\nConfiguration:")
        print(f"  Backend: {backend}")
        print(f"  Device: {device}")
        print(f"  World size: {world_size}")
        print(f"  XPU available: {is_xpu_available()}")
        print(f"  PyTorch version: {torch.__version__}")

    dist.barrier()

    # Define all tests
    tests = [
        # Basic collectives
        ("broadcast", test_broadcast),
        ("all_reduce (SUM)", test_all_reduce_sum),
        ("all_reduce (MAX)", test_all_reduce_max),
        ("all_reduce (MIN)", test_all_reduce_min),
        ("all_reduce (PRODUCT)", test_all_reduce_product),
        ("all_reduce (AVG)", test_all_reduce_avg),
        # Gather/Scatter
        ("all_gather", test_all_gather),
        ("all_gather_into_tensor", test_all_gather_into_tensor),
        ("reduce_scatter", test_reduce_scatter),
        ("reduce_scatter_tensor", test_reduce_scatter_tensor),
        ("scatter", test_scatter),
        ("gather", test_gather),
        # All-to-all
        ("all_to_all_single", test_all_to_all),
        # Point-to-point
        ("send/recv", test_send_recv),
        # Synchronization
        ("barrier", test_barrier),
        # Coalesced
        ("all_reduce_coalesced", test_all_reduce_coalesced),
        # NCCL-specific (expected to fail)
        ("PREMUL_SUM", test_premul_sum),
    ]

    results = {}
    for name, test_fn in tests:
        dist.barrier()
        success, message = test_operation(name, test_fn, device, rank, world_size)
        results[name] = (success, message)
        dist.barrier()

    # Report results (rank 0 only)
    if rank == 0:
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

        supported_count = 0
        total_count = len(results)

        for name, (success, message) in results.items():
            status = "[SUPPORTED]" if success else "[NOT SUPPORTED]"
            if success:
                supported_count += 1
            print(f"\n{name}:")
            print(f"  Status: {status}")
            print(f"  Details: {message}")

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"\n  Supported: {supported_count}/{total_count} operations")
        print(f"\n  Operations supported on XCCL backend:")
        for name, (success, _) in results.items():
            if success:
                print(f"    - {name}")
        print(f"\n  Operations NOT supported:")
        for name, (success, _) in results.items():
            if not success:
                print(f"    - {name}")
        print("=" * 70)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
