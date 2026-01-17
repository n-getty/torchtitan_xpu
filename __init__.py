# SPDX-License-Identifier: Apache-2.0
# Torchtitan XPU Extensions
#
# This package provides XPU-compatible implementations for torchtitan
# that work on Intel Aurora systems.

from .expert_parallel_xpu import (
    XPUExpertParallel,
    get_expert_parallel_class,
    is_xpu_available,
)

__all__ = [
    "XPUExpertParallel",
    "get_expert_parallel_class", 
    "is_xpu_available",
]
