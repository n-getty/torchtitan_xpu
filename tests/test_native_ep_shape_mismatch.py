#!/usr/bin/env python3
"""
Minimal reproduction script for ExpertParallel shape mismatch error.

This script isolates the _permute/_unpermute + all_to_all interaction
to identify the exact tensor shape mismatch.

Run with:
    mpiexec -n 2 -ppn 2 --envall python tests/test_native_ep_shape_mismatch.py
"""

import os
import sys

# Force native EP
os.environ["TORCHTITAN_XPU_FORCE_NATIVE_EP"] = "1"

import torch
import torch.distributed as dist


def is_xpu_available():
    try:
        import intel_extension_for_pytorch  # noqa: F401

        return torch.xpu.is_available()
    except ImportError:
        return False


def get_rank_and_world_size():
    if "PALS_RANKID" in os.environ:
        rank = int(os.environ["PALS_RANKID"])
        world_size = int(os.environ.get("PALS_LOCAL_SIZE", "1"))
        return rank, world_size
    if "PMI_RANK" in os.environ:
        rank = int(os.environ["PMI_RANK"])
        world_size = int(os.environ.get("PMI_SIZE", "1"))
        return rank, world_size
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def setup_distributed():
    os.environ["CCL_PROCESS_LAUNCHER"] = "none"
    os.environ["CCL_ATL_TRANSPORT"] = "ofi"

    rank, world_size = get_rank_and_world_size()

    if world_size < 2:
        print("ERROR: Requires at least 2 ranks")
        print(
            "Run: mpiexec -n 2 -ppn 2 --envall python tests/test_native_ep_shape_mismatch.py"
        )
        sys.exit(1)

    if "MASTER_ADDR" not in os.environ:
        import socket

        os.environ["MASTER_ADDR"] = socket.gethostname()
    os.environ.setdefault("MASTER_PORT", "29500")

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

    return device, rank, world_size


def test_permute_unpermute_shapes():
    """Test _permute/_unpermute shape changes in isolation."""
    from torchtitan.models.moe.utils import _permute, _unpermute

    device, rank, world_size = setup_distributed()

    print(f"\n{'=' * 60}")
    print(f"[Rank {rank}] Testing _permute/_unpermute shape changes")
    print(f"{'=' * 60}")

    # Simulate token counts after first all_to_all
    # Each rank has tokens for num_experts total experts
    num_experts = 4  # Total experts
    num_local_experts = num_experts // world_size  # 2 per rank
    ep_degree = world_size

    # Simulate varying token distribution
    # num_tokens_per_expert_group shape: (num_experts * ep_degree,)
    # Format: [rank0_exp0, rank0_exp1, ..., rank1_exp0, rank1_exp1, ...]
    if rank == 0:
        num_tokens_per_expert_group = torch.tensor(
            [5, 3, 7, 2, 4, 6, 3, 5], device=device
        )
    else:
        num_tokens_per_expert_group = torch.tensor(
            [4, 6, 3, 5, 5, 3, 7, 2], device=device
        )

    total_tokens = num_tokens_per_expert_group.sum().item()
    hidden_dim = 64

    # Create input tensor
    x = torch.randn(total_tokens, hidden_dim, device=device)

    print(f"[Rank {rank}] BEFORE _permute:")
    print(f"  x.shape = {x.shape}")
    print(f"  total_tokens = {total_tokens}")
    print(f"  num_tokens_per_expert_group = {num_tokens_per_expert_group.tolist()}")

    # Call _permute
    input_shape, x_permuted, permuted_indices, new_ntpe = _permute(
        x, num_tokens_per_expert_group, ep_degree, num_local_experts
    )

    print(f"\n[Rank {rank}] AFTER _permute:")
    print(f"  input_shape = {input_shape}")
    print(f"  x_permuted.shape = {x_permuted.shape}")
    print(f"  permuted_indices.shape = {permuted_indices.shape}")
    print(f"  new_ntpe = {new_ntpe.tolist()}")
    print(f"  sum(new_ntpe) = {new_ntpe.sum().item()}")

    # Simulate expert computation (same shape)
    expert_output = x_permuted * 2.0

    # Call _unpermute
    out = _unpermute(expert_output, input_shape, permuted_indices)

    print(f"\n[Rank {rank}] AFTER _unpermute:")
    print(f"  out.shape = {out.shape}")
    print(f"  expected for all_to_all: sum(output_splits) = {total_tokens}")
    print(f"  MATCH: {out.shape[0] == total_tokens}")

    dist.barrier()
    dist.destroy_process_group()


def test_full_ep_flow():
    """Test the full ExpertParallel flow to reproduce the error."""
    from torch.distributed._functional_collectives import (
        all_to_all_single,
        all_to_all_single_autograd,
    )

    from torchtitan.models.moe.utils import _permute, _unpermute

    device, rank, world_size = setup_distributed()

    print(f"\n{'=' * 60}")
    print(f"[Rank {rank}] Testing full EP flow with all_to_all")
    print(f"{'=' * 60}")

    # In the real EP flow:
    # - num_experts = total number of experts across all ranks
    # - Each rank owns num_local_experts = num_experts // ep_degree
    # - num_tokens_per_expert has shape (num_experts,) where:
    #   num_tokens_per_expert[i] = how many tokens THIS rank is sending to expert i
    #
    # The view(ep_degree, -1).sum(dim=1) computes:
    #   input_splits[r] = total tokens I'm sending to all experts on rank r

    num_experts = 4  # Total experts (2 per rank)
    num_local_experts = num_experts // world_size  # 2
    ep_degree = world_size
    hidden_dim = 64

    # Generate random token distribution
    # num_tokens_per_expert[i] = tokens this rank sends to expert i
    torch.manual_seed(42 + rank)
    num_tokens_per_expert = torch.randint(5, 15, (num_experts,), device=device)

    total_local_tokens = num_tokens_per_expert.sum().item()
    routed_input = torch.randn(total_local_tokens, hidden_dim, device=device)

    print(f"\n[Rank {rank}] INITIAL STATE:")
    print(f"  num_experts = {num_experts}, num_local_experts = {num_local_experts}")
    print(f"  num_tokens_per_expert = {num_tokens_per_expert.tolist()}")
    print(f"  routed_input.shape = {routed_input.shape}")

    # === DISPATCH PHASE ===
    print(f"\n[Rank {rank}] === DISPATCH PHASE ===")

    # Step 1: Exchange token counts via all_to_all
    # After this, we know how many tokens each rank is sending to each expert
    with torch.no_grad():
        num_tokens_per_expert_group = all_to_all_single(
            num_tokens_per_expert,
            None,
            None,
            group=dist.distributed_c10d._get_default_group(),
        )
        num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
            num_tokens_per_expert_group
        )

        # input_splits[r] = total tokens I'm sending to rank r
        # = sum of tokens for experts owned by rank r
        # Experts 0,1 are on rank 0; experts 2,3 are on rank 1 (for ep=2, local=2)
        input_splits_tensor = num_tokens_per_expert.view(ep_degree, -1).sum(dim=1)
        print(
            f"[Rank {rank}] input_splits_tensor (on device) = {input_splits_tensor.tolist()}"
        )
        input_splits = input_splits_tensor.to(
            torch.device("cpu"), non_blocking=True
        ).tolist()

        # output_splits[r] = total tokens I'm receiving from rank r
        # num_tokens_per_expert_group has format:
        #   [rank0_exp0, rank0_exp1, ..., rank1_exp0, rank1_exp1, ...]
        output_splits_tensor = num_tokens_per_expert_group.view(ep_degree, -1).sum(
            dim=1
        )
        print(
            f"[Rank {rank}] output_splits_tensor (on device) = {output_splits_tensor.tolist()}"
        )
        output_splits = output_splits_tensor.to(
            torch.device("cpu"), non_blocking=False
        ).tolist()

    print(f"[Rank {rank}] After token count exchange:")
    print(f"  num_tokens_per_expert_group = {num_tokens_per_expert_group.tolist()}")
    print(
        f"  num_tokens_per_expert.view({ep_degree}, -1) = {num_tokens_per_expert.view(ep_degree, -1).tolist()}"
    )
    print(f"  input_splits = {input_splits}, sum = {sum(input_splits)}")
    print(f"  output_splits = {output_splits}, sum = {sum(output_splits)}")

    # Step 2: First all_to_all for tokens
    print(f"[Rank {rank}] Calling all_to_all_single_autograd...")
    print(f"  Input shape: {routed_input.shape}")
    print(f"  Expected output tokens: {sum(output_splits)}")

    try:
        routed_input = all_to_all_single_autograd(
            routed_input,
            output_splits,
            input_splits,
            dist.distributed_c10d._get_default_group(),
        )
        print(f"[Rank {rank}] First all_to_all SUCCESS!")
        print(f"  routed_input.shape = {routed_input.shape}")
    except Exception as e:
        print(f"[Rank {rank}] First all_to_all FAILED: {e}")
        dist.destroy_process_group()
        return

    # Step 3: _permute
    print(f"\n[Rank {rank}] Calling _permute...")
    input_shape, routed_input, permuted_indices, num_tokens_per_expert_group = _permute(
        routed_input, num_tokens_per_expert_group, ep_degree, num_local_experts
    )

    print(f"[Rank {rank}] After _permute:")
    print(f"  input_shape = {input_shape}")
    print(f"  routed_input.shape = {routed_input.shape}")

    # === EXPERT COMPUTATION (simulated) ===
    routed_output = routed_input * 2.0

    # === COMBINE PHASE ===
    print(f"\n[Rank {rank}] === COMBINE PHASE ===")

    # Step 4: _unpermute
    print(f"[Rank {rank}] Calling _unpermute...")
    print(f"  routed_output.shape before = {routed_output.shape}")

    routed_output = _unpermute(routed_output, input_shape, permuted_indices)

    print(f"[Rank {rank}] After _unpermute:")
    print(f"  routed_output.shape = {routed_output.shape}")
    print(f"  Expected for all_to_all: sum(output_splits) = {sum(output_splits)}")
    print(f"  SHAPE MATCH: {routed_output.shape[0] == sum(output_splits)}")

    # Step 5: Second all_to_all - REVERSE the dispatch operation
    # In dispatch: sent input_splits[i] to rank i, received output_splits[i] from rank i
    # In combine: send output_splits[i] to rank i, receive input_splits[i] from rank i
    print(f"\n[Rank {rank}] Calling second all_to_all_single_autograd...")
    print(f"  Input shape: {routed_output.shape}")
    print(
        f"  input_splits (recv sizes - what we originally sent): {input_splits}, sum = {sum(input_splits)}"
    )
    print(
        f"  output_splits (send sizes - what we have now): {output_splits}, sum = {sum(output_splits)}"
    )

    try:
        routed_output = all_to_all_single_autograd(
            routed_output,
            input_splits,  # what we're receiving (what we originally sent)
            output_splits,  # what we're sending (what we currently have)
            dist.distributed_c10d._get_default_group(),
        )
        print(f"[Rank {rank}] Second all_to_all SUCCESS!")
        print(f"  routed_output.shape = {routed_output.shape}")
        print(f"  Expected: sum(input_splits) = {sum(input_splits)}")
        print(f"  MATCH: {routed_output.shape[0] == sum(input_splits)}")
    except RuntimeError as e:
        print(f"\n[Rank {rank}] Second all_to_all FAILED!")
        print(f"  ERROR: {e}")
        print(f"\n  DIAGNOSIS:")
        print(f"    Tensor dim 0 = {routed_output.shape[0]}")
        print(f"    sum(output_splits) = {sum(output_splits)}")
        print(f"    Difference = {routed_output.shape[0] - sum(output_splits)}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        choices=["permute", "full"],
        default="full",
        help="Which test to run",
    )
    args = parser.parse_args()

    if args.test == "permute":
        test_permute_unpermute_shapes()
    else:
        test_full_ep_flow()
