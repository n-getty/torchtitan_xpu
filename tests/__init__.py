# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
XPU/XCCL Compatibility Tests for torchtitan_xpu.

This package contains standalone test scripts to verify XPU and XCCL backend
support for various distributed operations and kernels.

Test Scripts:
    - test_all_to_all_support.py: Tests if all_to_all_single is supported
    - test_premul_sum_support.py: Tests if PREMUL_SUM reduce op is supported
    - test_triton_moe_support.py: Tests if Triton MoE kernels work on XPU
    - test_xccl_ops_support.py: Comprehensive test of all XCCL distributed ops

Run distributed tests with mpiexec (requires 2+ ranks):
    mpiexec -n 2 -ppn 2 --envall python tests/test_all_to_all_support.py
    mpiexec -n 2 -ppn 2 --envall python tests/test_premul_sum_support.py
    mpiexec -n 2 -ppn 2 --envall python tests/test_xccl_ops_support.py

Run Triton test (single-rank OK):
    python tests/test_triton_moe_support.py

Test Results on Aurora (25.190.0 frameworks, February 2026):
    - all_to_all: SUPPORTED (contrary to earlier assumptions!)
    - PREMUL_SUM: NOT SUPPORTED (NCCL-specific, use AVG or manual workaround)
    - Triton MoE: SUPPORTED (kernels compile and run on XPU)
    - 16/17 distributed ops: SUPPORTED on XCCL
"""
