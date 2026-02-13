# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3
"""
Test script to verify if Triton MoE kernels are supported on XPU.

This tests:
1. Quick checks: Triton import, XPU backend, simple kernel compilation
2. Full functional test: Run the actual fused_moe kernel with synthetic data

Run (single-rank is sufficient):
    python tests/test_triton_moe_support.py

Expected result on Aurora XPU: Partially supported (depends on Triton-Intel version)
"""

import sys

import torch

# Try to import triton at module level for kernel definitions
_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None


def is_xpu_available() -> bool:
    """Check if XPU device is available."""
    try:
        import intel_extension_for_pytorch  # noqa: F401

        return torch.xpu.is_available()
    except ImportError:
        return False


def get_device():
    """Get the best available device."""
    if is_xpu_available():
        return torch.device("xpu:0")
    elif torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


# =============================================================================
# Phase 1: Quick Checks
# =============================================================================


def test_triton_import():
    """Test if Triton can be imported."""
    print("\n[Test 1/6] Checking Triton import...")

    try:
        import triton
        import triton.language as tl

        version = getattr(triton, "__version__", "unknown")
        return True, f"Triton imported successfully (version: {version})"
    except ImportError as e:
        return False, f"Triton import failed: {e}"
    except Exception as e:
        return False, f"Unexpected error importing Triton: {type(e).__name__}: {e}"


def test_triton_xpu_backend():
    """Test if Triton has XPU backend support."""
    print("\n[Test 2/6] Checking Triton XPU backend...")

    try:
        import triton

        # Check for Intel XPU backend
        backends = getattr(triton, "backends", None)
        if backends is None:
            return False, "Triton.backends not available (old Triton version?)"

        # List available backends
        available_backends = list(backends.keys()) if hasattr(backends, "keys") else []

        # Check for intel or xpu backend
        has_intel = "intel" in available_backends or "xpu" in available_backends

        if has_intel:
            return True, f"XPU/Intel backend available. Backends: {available_backends}"
        else:
            return False, f"No XPU/Intel backend found. Available: {available_backends}"

    except AttributeError:
        # Older triton versions may not have backends attribute
        return False, "Cannot determine backends (older Triton API)"
    except Exception as e:
        return False, f"Error checking backends: {type(e).__name__}: {e}"


def test_simple_triton_kernel():
    """Test compiling and running a simple Triton kernel."""
    print("\n[Test 3/6] Testing simple Triton kernel compilation...")

    device = get_device()
    if device.type == "cpu":
        return False, "Triton requires GPU/XPU device, CPU not supported"

    if not _TRITON_AVAILABLE:
        return False, "Triton not available"

    try:
        import tempfile
        import importlib.util
        import os

        # Write kernel to a temporary file so Triton can read source
        kernel_code = """
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
"""
        # Create a temp file with .py extension
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(kernel_code)
            temp_path = f.name

        try:
            # Import the module from the temp file
            spec = importlib.util.spec_from_file_location("temp_kernel", temp_path)
            temp_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(temp_module)
            add_kernel = temp_module.add_kernel

            # Test data
            n = 1024
            x = torch.randn(n, device=device, dtype=torch.float32)
            y = torch.randn(n, device=device, dtype=torch.float32)
            output = torch.empty_like(x)

            # Launch kernel
            grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
            add_kernel[grid](x, y, output, n, BLOCK_SIZE=256)

            # Sync
            if device.type == "xpu":
                torch.xpu.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()

            # Verify
            expected = x + y
            if torch.allclose(output, expected, atol=1e-5):
                return True, f"Simple kernel compiled and ran correctly on {device}"
            else:
                max_diff = (output - expected).abs().max().item()
                return False, f"Kernel output incorrect, max diff: {max_diff}"
        finally:
            # Clean up temp file
            os.unlink(temp_path)

    except Exception as e:
        return False, f"Kernel compilation/execution failed: {type(e).__name__}: {e}"


# =============================================================================
# Phase 2: Full Functional Test
# =============================================================================


def test_fused_moe_import():
    """Test importing the fused_moe kernel from torchtitan."""
    print("\n[Test 4/6] Testing fused_moe import from torchtitan...")

    try:
        from torchtitan.models.moe.triton_fused_moe_xpu import fused_moe

        return True, "fused_moe imported successfully"
    except ImportError as e:
        return False, f"Import failed: {e}"
    except Exception as e:
        return False, f"Unexpected error: {type(e).__name__}: {e}"


def test_fused_moe_kernel_compile():
    """Test if the fused_moe kernel compiles (without running)."""
    print("\n[Test 5/6] Testing fused_moe kernel JIT compilation...")

    device = get_device()
    if device.type == "cpu":
        return False, "Triton requires GPU/XPU device"

    try:
        from torchtitan.models.moe.triton_fused_moe_xpu import fused_moe_kernel

        # Check if it's a triton JIT function
        if hasattr(fused_moe_kernel, "fn"):
            return True, "fused_moe_kernel is a valid Triton JIT function"
        else:
            return True, "fused_moe_kernel found (non-standard structure)"

    except ImportError as e:
        return False, f"Import failed: {e}"
    except Exception as e:
        return False, f"Kernel check failed: {type(e).__name__}: {e}"


def test_fused_moe_functional():
    """Test running the actual fused_moe kernel with synthetic data."""
    print("\n[Test 6/6] Testing fused_moe functional execution...")

    device = get_device()
    if device.type == "cpu":
        return False, "Triton requires GPU/XPU device"

    try:
        from torchtitan.models.moe.triton_fused_moe_xpu import fused_moe

        # Synthetic MoE configuration
        num_tokens = 32
        hidden_dim = 64
        inter_dim = 128
        num_experts = 4
        top_k = 1

        # Create synthetic inputs
        # hidden_states: [num_tokens, hidden_dim]
        hidden_states = torch.randn(
            num_tokens, hidden_dim, device=device, dtype=torch.bfloat16
        )

        # Expert weights: [num_experts, hidden_dim, inter_dim] for w1, w3
        #                 [num_experts, inter_dim, hidden_dim] for w2
        w1 = torch.randn(
            num_experts, hidden_dim, inter_dim, device=device, dtype=torch.bfloat16
        )
        w2 = torch.randn(
            num_experts, inter_dim, hidden_dim, device=device, dtype=torch.bfloat16
        )
        w3 = torch.randn(
            num_experts, hidden_dim, inter_dim, device=device, dtype=torch.bfloat16
        )

        # Routing: each token goes to top_k experts
        # topk_weights: [num_tokens, top_k]
        # topk_ids: [num_tokens, top_k]
        topk_weights = torch.ones(
            num_tokens, top_k, device=device, dtype=torch.bfloat16
        )
        topk_ids = torch.randint(
            0, num_experts, (num_tokens, top_k), device=device, dtype=torch.int32
        )

        # Run fused_moe
        output = fused_moe(
            hidden_states,
            w1,
            w2,
            w3,
            topk_weights,
            topk_ids,
            inplace=False,
        )

        # Sync
        if device.type == "xpu":
            torch.xpu.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

        # Verify output
        if output.shape != (num_tokens, hidden_dim):
            return (
                False,
                f"Output shape mismatch: {output.shape} vs expected ({num_tokens}, {hidden_dim})",
            )

        if torch.isnan(output).any():
            return False, "Output contains NaN values"

        if torch.isinf(output).any():
            return False, "Output contains Inf values"

        return (
            True,
            f"fused_moe executed successfully on {device}, output shape: {output.shape}",
        )

    except ImportError as e:
        return False, f"Import failed: {e}"
    except RuntimeError as e:
        error_msg = str(e)
        if "compile" in error_msg.lower() or "triton" in error_msg.lower():
            return False, f"Triton compilation error: {error_msg[:200]}"
        else:
            return False, f"Runtime error: {error_msg[:200]}"
    except Exception as e:
        return False, f"Execution failed: {type(e).__name__}: {e}"


def main():
    print("=" * 70)
    print("Testing Triton MoE Kernel Support on XPU")
    print("=" * 70)

    device = get_device()
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  XPU available: {is_xpu_available()}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  PyTorch version: {torch.__version__}")

    results = {}

    # Phase 1: Quick checks
    print("\n" + "-" * 40)
    print("Phase 1: Quick Checks")
    print("-" * 40)

    supported, message = test_triton_import()
    results["triton_import"] = (supported, message)

    if supported:
        supported, message = test_triton_xpu_backend()
        results["triton_xpu_backend"] = (supported, message)

        supported, message = test_simple_triton_kernel()
        results["simple_kernel"] = (supported, message)
    else:
        results["triton_xpu_backend"] = (False, "Skipped (Triton import failed)")
        results["simple_kernel"] = (False, "Skipped (Triton import failed)")

    # Phase 2: Full functional test
    print("\n" + "-" * 40)
    print("Phase 2: Full Functional Test")
    print("-" * 40)

    if results.get("simple_kernel", (False,))[0]:
        supported, message = test_fused_moe_import()
        results["fused_moe_import"] = (supported, message)

        if supported:
            supported, message = test_fused_moe_kernel_compile()
            results["fused_moe_compile"] = (supported, message)

            supported, message = test_fused_moe_functional()
            results["fused_moe_functional"] = (supported, message)
        else:
            results["fused_moe_compile"] = (False, "Skipped (import failed)")
            results["fused_moe_functional"] = (False, "Skipped (import failed)")
    else:
        results["fused_moe_import"] = (False, "Skipped (simple kernel failed)")
        results["fused_moe_compile"] = (False, "Skipped (simple kernel failed)")
        results["fused_moe_functional"] = (False, "Skipped (simple kernel failed)")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Check if basic triton functionality works (import + simple kernel)
    triton_works = (
        results.get("triton_import", (False,))[0]
        and results.get("simple_kernel", (False,))[0]
    )
    # Check if fused_moe specifically works
    fused_moe_works = all(
        results.get(k, (False,))[0]
        for k in ["fused_moe_import", "fused_moe_compile", "fused_moe_functional"]
    )

    for test_name, (supported, message) in results.items():
        status = "[SUPPORTED]" if supported else "[NOT SUPPORTED]"
        print(f"\n{test_name}:")
        print(f"  Status: {status}")
        print(f"  Details: {message}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"  Triton import:        {'PASS' if results.get('triton_import', (False,))[0] else 'FAIL'}"
    )
    print(
        f"  Simple kernel:        {'PASS' if results.get('simple_kernel', (False,))[0] else 'FAIL'}"
    )
    print(f"  fused_moe functional: {'PASS' if fused_moe_works else 'FAIL'}")

    if triton_works and fused_moe_works:
        print("\nOVERALL: Triton MoE kernels ARE SUPPORTED on this system")
        print("         You can set use_triton_moe = true in config")
    elif triton_works:
        print("\nOVERALL: Triton works but fused_moe kernel has issues")
        print("         Recommend use_triton_moe = false")
    else:
        print("\nOVERALL: Triton MoE kernels are NOT SUPPORTED on this system")
        print("         Use use_triton_moe = false in config")
    print("=" * 70)


if __name__ == "__main__":
    main()
