#!/usr/bin/env python3
"""MPI-based wrapper for torchtitan training on Aurora XPU.

This wrapper sets up the distributed environment using mpi4py following
the ALCF documentation pattern for Aurora distributed training with xccl backend.
"""
from mpi4py import MPI
import os
import sys
import socket

# Get MPI rank info
SIZE = MPI.COMM_WORLD.Get_size()
RANK = MPI.COMM_WORLD.Get_rank()
LOCAL_RANK = os.environ.get('PALS_LOCAL_RANKID', str(RANK % 12))

# Set environment variables for PyTorch distributed
os.environ['RANK'] = str(RANK)
os.environ['WORLD_SIZE'] = str(SIZE)
os.environ['LOCAL_RANK'] = str(LOCAL_RANK)

# Broadcast master address from rank 0
MASTER_ADDR = socket.gethostname() if RANK == 0 else None
MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
os.environ['MASTER_ADDR'] = f"{MASTER_ADDR}.hsn.cm.aurora.alcf.anl.gov"
os.environ['MASTER_PORT'] = str(29600)

print(f"MPI Wrapper: Rank {RANK}/{SIZE}, Local Rank {LOCAL_RANK}, Master: {os.environ['MASTER_ADDR']}")

# Import torch and IPEX after setting env vars
import intel_extension_for_pytorch as ipex
import torch

# Initialize distributed with xccl backend
print(f"Rank {RANK}: Initializing torch.distributed with backend='xccl'")
torch.distributed.init_process_group(
    backend='xccl',
    init_method='env://',
    rank=int(RANK),
    world_size=int(SIZE)
)
print(f"Rank {RANK}: torch.distributed initialized successfully")

# Pin GPU to local rank
torch.xpu.set_device(int(LOCAL_RANK))
print(f"Rank {RANK}: Set XPU device to {LOCAL_RANK}")

# Now import and run torchtitan train
# Pass through command line arguments
sys.argv[0] = 'torchtitan/train.py'  # Fix argv[0] for config parsing

# Import train module and run
from torchtitan.train import main, Trainer

if __name__ == "__main__":
    try:
        main(Trainer)
    finally:
        # Cleanup
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            print(f"Rank {RANK}: Destroyed process group")
