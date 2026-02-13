#!/usr/bin/env python
"""
Test script to diagnose memory leak in EP implementations.
Runs a minimal EP training loop with explicit memory tracking.
"""

import os
import sys
import socket
import torch
import torch.distributed as dist
from mpi4py import MPI

# Setup
os.environ.setdefault("ZE_FLAT_DEVICE_HIERARCHY", "FLAT")


def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    local_rank = rank % 12  # Assume 12 devices per node

    # Set up master address
    master_addr = socket.gethostname() + ".hsn.cm.aurora.alcf.anl.gov"
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = "29520"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    device = torch.device(f"xpu:{local_rank}")
    torch.xpu.set_device(device)

    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")

    # Simple memory test
    def get_memory_mb():
        torch.xpu.synchronize()
        return torch.xpu.memory_allocated(device) / 1024 / 1024

    def get_reserved_mb():
        torch.xpu.synchronize()
        return torch.xpu.memory_reserved(device) / 1024 / 1024

    if rank == 0:
        print(
            f"Initial memory: allocated={get_memory_mb():.1f}MB, reserved={get_reserved_mb():.1f}MB"
        )

    # Create test tensors
    batch_size = 4
    seq_len = 2048
    dim = 2048
    num_tokens = batch_size * seq_len

    for step in range(5):
        # Simulate token dispatch/combine
        tokens = torch.randn(num_tokens, dim, device=device, dtype=torch.bfloat16)

        # All-gather test (similar to EP)
        gathered = [torch.empty_like(tokens) for _ in range(world_size)]
        dist.all_gather(gathered, tokens)

        # Simulate expert processing
        result = torch.cat(gathered, dim=0)
        output = result.sum()  # Just to use it

        # Backward
        tokens_with_grad = tokens.clone().requires_grad_(True)
        loss = tokens_with_grad.sum()
        loss.backward()

        # Cleanup
        del tokens, gathered, result, output, tokens_with_grad, loss
        torch.xpu.empty_cache()

        if rank == 0:
            mem_alloc = get_memory_mb()
            mem_reserved = get_reserved_mb()
            print(
                f"Step {step + 1}: allocated={mem_alloc:.1f}MB, reserved={mem_reserved:.1f}MB"
            )

    dist.barrier()
    if rank == 0:
        print("Test completed successfully!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
