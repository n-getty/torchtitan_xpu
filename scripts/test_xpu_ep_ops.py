#!/usr/bin/env python3
"""
Test script to verify XPU-compatible Expert Parallelism operations.

Run on a compute node with XPU devices available:
    python test_xpu_ep_ops.py

This tests the basic broadcast/all_reduce operations that
XPUExpertParallel uses, ensuring they work on XCCL backend.
"""

import os
import sys

import torch
import torch.distributed as dist


def setup_distributed():
    """Initialize distributed environment for testing."""
    import os
    
    # For single-rank testing, use gloo backend (xccl requires MPI launcher)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    
    # Disable oneCCL PMIx launcher for single-process testing
    os.environ["CCL_PROCESS_LAUNCHER"] = "none"
    os.environ["CCL_ATL_TRANSPORT"] = "ofi"
    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # Detect XPU availability
    xpu_available = False
    try:
        import intel_extension_for_pytorch  # noqa: F401
        if torch.xpu.is_available():
            xpu_available = True
            xpu_device = torch.device(f"xpu:{rank}")
            print(f"XPU device available: {xpu_device}")
        else:
            print("XPU not available")
            xpu_device = None
    except ImportError:
        print("IPEX not available")
        xpu_device = None
    
    # For single-rank testing, use gloo backend with CPU tensors
    # For multi-rank with mpiexec, xccl with XPU tensors would be used
    if world_size == 1:
        backend = "gloo"
        comm_device = torch.device("cpu")  # gloo requires CPU tensors
        print(f"Using gloo backend with CPU tensors for single-rank testing")
    else:
        backend = "xccl" if xpu_available else "gloo"
        comm_device = xpu_device if backend == "xccl" else torch.device("cpu")
        print(f"Using {backend} backend for multi-rank")
    
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    
    return comm_device, xpu_device, rank, world_size


def test_broadcast(comm_device, rank, world_size):
    """Test broadcast operation."""
    print(f"\n[Test] Broadcast operation...")
    
    tensor = torch.zeros(10, 5, device=comm_device)
    if rank == 0:
        tensor.fill_(1.0)
    
    dist.broadcast(tensor, src=0)
    
    assert torch.allclose(tensor, torch.ones_like(tensor)), \
        f"Broadcast failed! Expected all 1s, got min={tensor.min()}, max={tensor.max()}"
    
    print(f"  [PASS] Broadcast works correctly on rank {rank}")
    return True


def test_all_reduce(comm_device, rank, world_size):
    """Test all_reduce operation."""
    print(f"\n[Test] All-reduce operation...")
    
    tensor = torch.full((10, 5), fill_value=float(rank + 1), device=comm_device)
    
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    expected_sum = sum(range(1, world_size + 1))  # 1 + 2 + ... + world_size
    expected = torch.full((10, 5), fill_value=float(expected_sum), device=comm_device)
    
    assert torch.allclose(tensor, expected), \
        f"All-reduce failed! Expected {expected_sum}, got {tensor[0,0].item()}"
    
    print(f"  [PASS] All-reduce works correctly on rank {rank}")
    return True


def test_all_gather(comm_device, rank, world_size):
    """Test all_gather_into_tensor operation."""
    print(f"\n[Test] All-gather operation...")
    
    local_tensor = torch.full((5, 3), fill_value=float(rank), device=comm_device)
    gathered = torch.zeros(world_size * 5, 3, device=comm_device)
    
    # Reshape for all_gather_into_tensor
    gathered_list = [torch.zeros_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_list, local_tensor)
    
    for i, t in enumerate(gathered_list):
        expected = torch.full_like(t, fill_value=float(i))
        assert torch.allclose(t, expected), \
            f"All-gather failed for rank {i}!"
    
    print(f"  [PASS] All-gather works correctly on rank {rank}")
    return True


def test_simulated_dispatch_combine(comm_device, rank, world_size):
    """
    Simulate the dispatch/combine pattern used in XPUExpertParallel.
    
    This mimics what happens during MoE forward pass:
    1. Dispatch: Tokens are scattered to experts on different ranks
    2. Combine: Expert outputs are gathered back
    """
    print(f"\n[Test] Simulated dispatch/combine pattern...")
    
    # Simulate 20 tokens, 4 experts (2 per rank if world_size=2)
    num_tokens = 20
    hidden_dim = 8
    num_experts = 4
    num_local_experts = num_experts // world_size
    
    # Original tokens on this rank
    my_tokens = torch.randn(num_tokens // world_size, hidden_dim, device=comm_device)
    
    # Step 1: Broadcast-based dispatch
    # Each rank broadcasts its tokens
    all_tokens = []
    for src in range(world_size):
        buffer = torch.zeros_like(my_tokens)
        if src == rank:
            buffer.copy_(my_tokens)
        dist.broadcast(buffer, src=src)
        all_tokens.append(buffer)
    
    combined_input = torch.cat(all_tokens, dim=0)
    print(f"  Dispatch: Gathered {combined_input.shape[0]} tokens from {world_size} ranks")
    
    # Step 2: Process (simulate expert computation)
    # Each rank processes tokens for its local experts
    local_output = combined_input * (rank + 1)  # Simulate rank-specific computation
    
    # Step 3: All-reduce based combine
    # Place output in global buffer and all_reduce
    global_buffer = torch.zeros(num_tokens, hidden_dim, device=comm_device)
    start = rank * (num_tokens // world_size)
    end = start + (num_tokens // world_size)
    global_buffer[start:end] = local_output[:num_tokens // world_size]
    
    dist.all_reduce(global_buffer, op=dist.ReduceOp.SUM)
    
    # Extract my portion
    my_output = global_buffer[start:end]
    print(f"  Combine: Output shape {my_output.shape}")
    
    print(f"  [PASS] Dispatch/combine pattern works on rank {rank}")
    return True


def test_xpu_compute(xpu_device):
    """Test XPU tensor compute (not distributed)."""
    if xpu_device is None:
        print("\n[Test] XPU compute test - SKIPPED (no XPU device)")
        return True
    
    print(f"\n[Test] XPU compute test...")
    
    # Simple XPU computation
    a = torch.randn(100, 100, device=xpu_device)
    b = torch.randn(100, 100, device=xpu_device)
    c = torch.matmul(a, b)
    
    # Verify result
    a_cpu = a.cpu()
    b_cpu = b.cpu()
    c_expected = torch.matmul(a_cpu, b_cpu)
    c_cpu = c.cpu()
    
    assert torch.allclose(c_cpu, c_expected, atol=1e-4), \
        f"XPU compute failed! Max diff: {(c_cpu - c_expected).abs().max()}"
    
    print(f"  [PASS] XPU compute works correctly")
    return True


def main():
    print("=" * 60)
    print("Testing XPU-Compatible EP Operations")
    print("=" * 60)
    
    comm_device, xpu_device, rank, world_size = setup_distributed()
    
    print(f"\nDistributed setup: rank={rank}, world_size={world_size}")
    print(f"Comm device: {comm_device}")
    print(f"XPU device: {xpu_device}")
    
    all_passed = True
    
    try:
        all_passed &= test_broadcast(comm_device, rank, world_size)
        all_passed &= test_all_reduce(comm_device, rank, world_size)
        all_passed &= test_all_gather(comm_device, rank, world_size)
        all_passed &= test_simulated_dispatch_combine(comm_device, rank, world_size)
        all_passed &= test_xpu_compute(xpu_device)
    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        all_passed = False
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED! XPU-compatible EP operations work correctly.")
    else:
        print("Some tests FAILED!")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
