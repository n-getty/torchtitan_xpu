import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    compute_type: tl.constexpr,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)

@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # Strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.
    Simplified version for BF16/FP32 only.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    
    # Handle cases where expert is not assigned (-1)
    if off_experts == -1:
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Check dimensions for pointer arithmetic
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
            other=0.0,
        )
        
        accumulator += tl.dot(a, b.to(a.dtype)) # simplified accumulation

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def invoke_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict,
):
    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )

    fused_moe_kernel[grid](
        A,
        B,
        C,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1], # N
        A.shape[1], # K
        sorted_token_ids.shape[0], # EM
        A.shape[0] * top_k, # num_valid_tokens
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        BLOCK_SIZE_M=config['BLOCK_SIZE_M'],
        BLOCK_SIZE_N=config['BLOCK_SIZE_N'],
        BLOCK_SIZE_K=config['BLOCK_SIZE_K'],
        GROUP_SIZE_M=config['GROUP_SIZE_M'],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=tl.float32, # Accumulate in FP32
    )

def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor, 
    w3: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    return_inter: bool = False,
) -> torch.Tensor:
    """
    Fused MoE kernel launch wrapper.
    Args:
        hidden_states: [num_tokens, hidden_dim]
        w1: [num_experts, hidden_dim, inter_dim] (Gate)
        w2: [num_experts, inter_dim, hidden_dim] (Down)
        w3: [num_experts, hidden_dim, inter_dim] (Up)
        topk_weights: [num_tokens, top_k]
        topk_ids: [num_tokens, top_k]
    """
    num_tokens, hidden_dim = hidden_states.shape
    num_experts = w1.shape[0]
    inter_dim = w1.shape[2]
    top_k = topk_weights.shape[1]

    # Handle DTensor inputs (for Tensor Parallelism)
    if hasattr(w1, "to_local"):
        w1 = w1.to_local()
    if hasattr(w2, "to_local"):
        w2 = w2.to_local()
    if hasattr(w3, "to_local"):
        w3 = w3.to_local()
    if hasattr(hidden_states, "to_local"):
        hidden_states = hidden_states.to_local()
    
    # 1. flatten topk_ids
    topk_ids = topk_ids.flatten()
    
    # 2. Sort tokens by expert
    sorted_indices = torch.argsort(topk_ids, stable=True)
    sorted_expert_ids = topk_ids[sorted_indices]
    
    # 3. Calculate token counts per expert
    expert_counts = torch.bincount(sorted_expert_ids, minlength=num_experts)
    
    # 4. Calculate padding needed per expert to align to BLOCK_SIZE_M
    block_size_m = 64
    
    expert_counts_aligned = (expert_counts + block_size_m - 1) // block_size_m * block_size_m
    padding_needed = expert_counts_aligned - expert_counts
    
    total_tokens_padded = expert_counts_aligned.sum().item()
    
    # 5. Construct sorted_token_ids with padding
    # sorted_token_ids should contain indices into the flattened input (0 .. num_tokens*top_k - 1)
    
    # We construct lists and cat them.
    padded_chunks = []
    expert_ids_chunks = []
    
    # Original sorted indices (flattened indices)
    sorted_indices_original = sorted_indices
    
    current_unpadded_ptr = 0
    num_tokens_flat = num_tokens * top_k
    
    for i in range(num_experts):
        count = expert_counts[i].item()
        if count == 0:
            continue
            
        # Get flattened indices for this expert
        tokens = sorted_indices_original[current_unpadded_ptr : current_unpadded_ptr + count]
        padded_chunks.append(tokens)
        
        # Expert ID for the blocks
        num_blocks = (count + block_size_m - 1) // block_size_m
        expert_ids_chunks.append(torch.full((num_blocks,), i, dtype=torch.int32, device=hidden_states.device))
        
        # Padding
        pad = padding_needed[i].item()
        if pad > 0:
            # Pad with out-of-bounds index
            padded_chunks.append(torch.full((pad,), num_tokens_flat, dtype=torch.int32, device=hidden_states.device))
            
        current_unpadded_ptr += count
        
    if not padded_chunks:
        sorted_token_ids = torch.zeros(0, dtype=torch.int32, device=hidden_states.device)
        expert_ids = torch.zeros(0, dtype=torch.int32, device=hidden_states.device)
        num_tokens_post_padded = 0
    else:
        sorted_token_ids = torch.cat(padded_chunks)
        expert_ids = torch.cat(expert_ids_chunks)
        num_tokens_post_padded = sorted_token_ids.numel()
        
    num_tokens_post_padded_ptr = torch.tensor([num_tokens_post_padded], device=hidden_states.device, dtype=torch.int32)
    
    
    # DEBUG
    print(f"DEBUG: num_tokens={num_tokens}, top_k={top_k}, num_experts={num_experts}")
    print(f"DEBUG: sorted_indices shape: {sorted_indices.shape}")
    print(f"DEBUG: expert_counts: {expert_counts}")
    print(f"DEBUG: num_tokens_post_padded: {num_tokens_post_padded}")
    
    config = {
        'BLOCK_SIZE_M': block_size_m,
        'BLOCK_SIZE_N': 64,
        'BLOCK_SIZE_K': 32,
        'GROUP_SIZE_M': 8,
    }
    
    # --- Stage 1: Up Projection (Gate * Up) ---
    w13 = torch.cat([w1, w3], dim=2) 
    w13_t = w13.transpose(1, 2).contiguous()
    
    # DEBUG
    print(f"DEBUG: w13_t shape: {w13_t.shape}, strides: {w13_t.stride()}")
    
    inter_states = torch.empty((num_tokens_post_padded, 2 * inter_dim), device=hidden_states.device, dtype=hidden_states.dtype)
    
    invoke_fused_moe_kernel(
        hidden_states,
        w13_t,
        inter_states,
        topk_weights,
        sorted_token_ids,
        expert_ids, # BLOCK-level expert ids
        num_tokens_post_padded_ptr,
        mul_routed_weight=False,
        top_k=top_k, # Use actual top_k to divide indices
        config=config
    )
    
    # --- Activation + Element-wise Mul (SwiGLU) ---
    gate, up = inter_states.chunk(2, dim=-1)
    inter_states = F.silu(gate) * up
    
    # --- Stage 2: Down Projection ---
    w2_t = w2.transpose(1, 2).contiguous()
    down_out = torch.empty((num_tokens_post_padded, hidden_dim), device=hidden_states.device, dtype=hidden_states.dtype)
    
    invoke_fused_moe_kernel(
        inter_states, # Input is [num_post_padded, D]
        w2_t,
        down_out,
        topk_weights,
        sorted_token_ids, # Reuse same sorting
        expert_ids,
        num_tokens_post_padded_ptr,
        mul_routed_weight=True,
        top_k=1, # Treat as top_k=1 so we index inter_states linearly
        config=config
    )
    
    # --- Reduction ---
    final_out = torch.zeros((num_tokens, hidden_dim), device=hidden_states.device, dtype=hidden_states.dtype)
    
    # Filter valid results
    valid_mask = sorted_token_ids < num_tokens_flat
    valid_indices_flat = sorted_token_ids[valid_mask]
    valid_results = down_out[valid_mask]
    
    # Map flattened indices back to token indices
    valid_token_indices = valid_indices_flat // top_k
    
    final_out.index_add_(0, valid_token_indices.to(torch.int64), valid_results)
    
    if return_inter:
        return final_out, inter_states, sorted_token_ids, topk_weights
        
    return final_out 

