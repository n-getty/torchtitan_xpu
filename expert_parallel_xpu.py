# SPDX-License-Identifier: Apache-2.0
# XPU-Compatible Expert Parallelism for Torchtitan
#
# This module provides XPU-compatible implementations of ExpertParallel
# that use all_gather instead of all_to_all operations, since
# Intel's XCCL backend doesn't support all_to_all.
#
# Optimization v2:
# Replaced naive N^2 broadcast loop with padding + all_gather.
# This significantly reduces collective communication overhead.

from abc import ABC, abstractmethod
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.tensor.parallel import ParallelStyle


def is_xpu_available() -> bool:
    """Check if XPU device is available."""
    try:
        import intel_extension_for_pytorch  # noqa: F401
        return torch.xpu.is_available()
    except ImportError:
        return False


class XPUExpertParallel(ParallelStyle):
    """
    XPU-compatible Expert Parallelism using all_gather operations.
    
    This implementation avoids using `all_to_all_single` which is not supported
    on Intel XPU's XCCL backend. Instead, it uses:
    1. all_gather(num_tokens_per_expert) to exchange metadata.
    2. all_gather(padded_tokens) for dispatch and combine.
    
    This handles dynamic token loads and avoids the N^2 overhead of broadcast loops.
    """
    
    def __init__(self):
        super().__init__()
        self._dispatch_state: Optional[dict] = None
    
    def _partition_fn(self, name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
        """Shard expert weights across EP ranks on the expert dimension."""
        # Simple sharding on dim 0 (experts)
        for param_name, param in mod.named_parameters(recurse=False):
            dist_param = nn.Parameter(
                distribute_tensor(param, device_mesh, [Shard(0)])
            )
            mod.register_parameter(param_name, dist_param)
    
    def _all_gather_dispatch(
        self,
        tensor: Tensor,
        num_tokens_per_expert: Tensor,
        device_mesh: DeviceMesh,
    ) -> Tensor:
        """
        Dispatch tokens to experts using all_gather.
        """
        ep_group = device_mesh.get_group()
        ep_size = device_mesh.shape[0]
        ep_rank = dist.get_rank(ep_group)
        num_experts = num_tokens_per_expert.shape[0]
        num_local_experts = num_experts // ep_size
        
        # 1. Exchange Metadata: Gather num_tokens_per_expert from all ranks
        # local: (num_experts,) -> global: (ep_size, num_experts)
        local_ntpe = num_tokens_per_expert
        global_ntpe_list = [torch.zeros_like(local_ntpe) for _ in range(ep_size)]
        dist.all_gather(global_ntpe_list, local_ntpe, group=ep_group)
        global_ntpe = torch.stack(global_ntpe_list) # (ep_size, num_experts)
        
        # 2. Determine padding needed for all_gather of tokens
        total_tokens_per_rank = global_ntpe.sum(dim=1) # (ep_size,)
        max_tokens = total_tokens_per_rank.max().item()
        
        # 3. Pad and All-Gather Input Tokens
        # tensor is (local_total_tokens, dim)
        current_tokens = tensor.shape[0]
        pad_size = max_tokens - current_tokens
        
        if pad_size > 0:
            padded_input = F.pad(tensor, (0, 0, 0, int(pad_size)))
        else:
            padded_input = tensor
            
        # (ep_size, max_tokens, dim)
        gathered_tokens_list = [torch.zeros_like(padded_input) for _ in range(ep_size)]
        dist.all_gather(gathered_tokens_list, padded_input, group=ep_group)
        
        # 4. Extract tokens destined for MY local experts
        # My expert indices: [start_expert, end_expert)
        start_expert = ep_rank * num_local_experts
        end_expert = (ep_rank + 1) * num_local_experts
        
        dispatched_tokens = []
        
        for src_rank in range(ep_size):
            src_tokens_count = total_tokens_per_rank[src_rank].item()
            # Unpad by slicing
            src_tensor = gathered_tokens_list[src_rank][:src_tokens_count]
            
            # Calculate offsets to find where MY experts' tokens are in src_tensor
            # src_tensor is sorted by expert_id (0..N)
            src_hist = global_ntpe[src_rank] # (num_experts,)
            
            # Cumulative sum to find start indices
            src_cumsum = torch.cumsum(src_hist, 0)
            # Prepend 0
            src_offsets = torch.cat([torch.tensor([0], device=tensor.device), src_cumsum])
            
            # Indices for my range of experts
            idx_start = src_offsets[start_expert].item()
            idx_end = src_offsets[end_expert].item()
            
            # Extract and append
            tokens_for_me = src_tensor[idx_start:idx_end]
            dispatched_tokens.append(tokens_for_me)
            
        # Concatenate all inputs for my experts
        output_buffer = torch.cat(dispatched_tokens, dim=0)
        
        # Save state for combine
        self._dispatch_state = {
            'global_ntpe': global_ntpe,
            'ep_size': ep_size,
            'ep_rank': ep_rank,
            'num_local_experts': num_local_experts,
            'original_input_shape': tensor.shape
        }
        
        return output_buffer

    def _all_gather_combine(
        self,
        routed_output: Tensor, # (total_tokens_for_my_experts, dim)
        device_mesh: DeviceMesh,
    ) -> Tensor:
        """
        Combine expert outputs using all_gather.
        """
        assert self._dispatch_state is not None
        state = self._dispatch_state
        global_ntpe = state['global_ntpe'] # (ep_size, num_experts)
        ep_size = state['ep_size']
        ep_rank = state['ep_rank']
        num_local_experts = state['num_local_experts']
        ep_group = device_mesh.get_group()
        
        # 1. Determine padding for all_gather of outputs
        # Calculate how many tokens each rank PRODUCED (processed)
        # Rank R processed tokens for experts [R*n_local, (R+1)*n_local]
        # These tokens came from ALL source ranks.
        # So processed count for Rank R = sum(global_ntpe[:, experts_of_R])
        
        processed_counts = []
        for r in range(ep_size):
            r_experts_start = r * num_local_experts
            r_experts_end = (r + 1) * num_local_experts
            count = global_ntpe[:, r_experts_start:r_experts_end].sum().item()
            processed_counts.append(count)
        
        max_processed = max(processed_counts)
        current_processed = routed_output.shape[0]
        
        # Sanity check
        assert current_processed == processed_counts[ep_rank], \
            f"Rank {ep_rank} processed mismatch: expected {processed_counts[ep_rank]}, got {current_processed}"

        # 2. Pad and All-Gather Outputs
        pad_size = max_processed - current_processed
        if pad_size > 0:
            padded_output = F.pad(routed_output, (0, 0, 0, int(pad_size)))
        else:
            padded_output = routed_output
            
        gathered_outputs_list = [torch.zeros_like(padded_output) for _ in range(ep_size)]
        dist.all_gather(gathered_outputs_list, padded_output, group=ep_group)
        
        # 3. Reconstruct My Answer
        # We need to pull pieces from every rank's output buffer that belong to ME.
        # My original tokens were sorted by expert_id.
        # So I need to recover them in that order.
        
        final_output = torch.empty(
            state['original_input_shape'],
            dtype=routed_output.dtype,
            device=routed_output.device
        )
        
        # We iterate through experts 0..N. 
        # For each expert, we identify which Rank holds it, and where my tokens are in that Rank's buffer.
        
        # My original local distribution
        my_ntpe = global_ntpe[ep_rank] # (num_experts,)
        
        current_fill_idx = 0
        num_experts = my_ntpe.shape[0]
        
        for expert_id in range(num_experts):
            count = my_ntpe[expert_id].item()
            if count == 0:
                continue
            
            # Which rank processed this expert?
            target_rank = expert_id // num_local_experts
            
            # Get that rank's buffer (unpadded is implicitly handled by careful indexing)
            target_buffer = gathered_outputs_list[target_rank]
            
            # Calculate Offset in target_buffer
            # Target Buffer structure (from dispatch): 
            # [Src0_Tokens], [Src1_Tokens]...
            # Within [SrcX_Tokens], structure is [Ext_Start, ... Ext_End]
            
            # We need to skip:
            # 1. All tokens from sources 0..(Me-1) sent to this Target Rank's experts.
            target_experts_range = slice(target_rank * num_local_experts, (target_rank + 1) * num_local_experts)
            tokens_from_prev_ranks = global_ntpe[:ep_rank, target_experts_range].sum().item()
            
            # 2. All tokens from Me sent to experts *before* exp_id (but within this Target Rank).
            expert_offset_in_rank = expert_id - (target_rank * num_local_experts)
            tokens_from_me_prev_experts = global_ntpe[ep_rank, target_rank * num_local_experts : expert_id].sum().item()
            
            start_offset = tokens_from_prev_ranks + tokens_from_me_prev_experts
            
            # Extract
            data = target_buffer[start_offset : start_offset + count]
            final_output[current_fill_idx : current_fill_idx + count] = data
            current_fill_idx += count
            
        self._dispatch_state = None
        return final_output

    
    def _token_dispatch(
        self, mod: nn.Module, inputs: tuple, device_mesh: DeviceMesh
    ) -> tuple[Tensor, Tensor]:
        """Dispatch tokens."""
        routed_input, num_tokens_per_expert = inputs
        dispatched = self._all_gather_dispatch(routed_input, num_tokens_per_expert, device_mesh)
        
        # Calculate local num_tokens_per_expert for the experts computation
        # dispatched buffer contains tokens for [start_expert, end_expert]
        # We need to compute the local histogram for these tokens
        # Actually, we can derive it from global_ntpe we already gathered!
        state = self._dispatch_state
        global_ntpe = state['global_ntpe']
        ep_rank = state['ep_rank']
        num_local = state['num_local_experts']
        
        # Sum across all source ranks for my experts
        # shape (ep_size, num_experts) -> want (num_local_experts,)
        # Sum dim 0 (sources), slice dim 1 (my experts)
        my_experts_counts = global_ntpe[:, ep_rank*num_local : (ep_rank+1)*num_local].sum(dim=0)
        
        return dispatched, my_experts_counts
    
    def _token_combine(
        self, mod: nn.Module, routed_output: Tensor, device_mesh: DeviceMesh
    ) -> Tensor:
        """Combine tokens."""
        return self._all_gather_combine(routed_output, device_mesh)
    
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )


def get_expert_parallel_class():
    if is_xpu_available():
        return XPUExpertParallel
    else:
        from torchtitan.distributed.expert_parallel import ExpertParallel
        return ExpertParallel
