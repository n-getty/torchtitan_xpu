import os
import sys
import shutil
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass

# Add path to find torchtitan package
# Assumptions: script is in ./scripts/ and torchtitan package is in ./torchtitan/torchtitan
# We need to add ./torchtitan to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
# If script is in scripts/, parent is root. Root contains torchtitan/ (which contains package torchtitan)
# Wait, list_dir showed torchtitan/torchtitan structure.
# So we need to add root/torchtitan to path?
# Let's try adding root first.
project_root = os.path.abspath(os.path.join(script_dir, '..'))
torchtitan_outer = os.path.join(project_root, 'torchtitan')
if torchtitan_outer not in sys.path:
    sys.path.append(torchtitan_outer)

print(f"Added {torchtitan_outer} to sys.path")

try:
    from torchtitan.components.checkpoint import CheckpointManager
    from torchtitan.config.job_config import Checkpoint as CheckpointConfig
except ImportError as e:
    print(f"Failed to import torchtitan: {e}")
    print("sys.path:", sys.path)
    sys.exit(1)

class FakeOptimizersContainer:
    def __init__(self):
        self.state = {"step": 0}

    def state_dict(self):
        return {"optimizer_state": self.state}

    def load_state_dict(self, sd: dict):
        self.state = sd.get("optimizer_state", {})

    def init_cache_state_dict(self):
        pass

class FakeLRSchedulersContainer:
    def state_dict(self): return {}
    def load_state_dict(self, sd): pass

class FakeDataLoader(DataLoader):
    def __init__(self): 
        pass
    def state_dict(self): return {}
    def load_state_dict(self, sd): pass
    def __iter__(self): return iter([])

def run_test():
    # Init distributed
    backend = "gloo"
    
    print("Checking XPU availability...")
    try:
        import intel_extension_for_pytorch as ipex
        print(f"IPEX imported: {ipex.__version__}")
    except ImportError as e:
        print(f"Failed to import IPEX: {e}")
    except Exception as e:
        print(f"Error importing IPEX: {e}")

    if hasattr(torch, "xpu"):
        print(f"torch.xpu exists. is_available: {torch.xpu.is_available()}")
        print(f"device_count: {torch.xpu.device_count()}")
    else:
        print("torch.xpu does not exist")

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            import oneccl_bindings_for_pytorch
            backend = "ccl"
            print("Using ccl backend (oneccl imported)")
        except ImportError:
            print("oneccl_bindings_for_pytorch not found, falling back to gloo")
            pass
    
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
         dist.init_process_group(backend=backend)
    else:
         os.environ["MASTER_ADDR"] = "localhost"
         os.environ["MASTER_PORT"] = "29500"
         os.environ["RANK"] = "0"
         os.environ["WORLD_SIZE"] = "1"
         dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    print(f"Rank {rank} initialized with backend {backend}")

    # Setup matches device if available
    device = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Simple model
    model = nn.Linear(10, 10).to(device)
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(0.0)
    
    optimizer = FakeOptimizersContainer()
    scheduler = FakeLRSchedulersContainer()
    dataloader = FakeDataLoader()
    
    states = {"extra_state": torch.tensor([123], device=device)}
    
    ckpt_config = CheckpointConfig(
        enable=True,
        folder="checkpoint_verify_test",
        interval=1,
        async_mode="disabled", 
    )
    
    # Use a temp directory relative to current or a specific place
    # We use a unique name in current dir to be visible
    base_folder = os.path.abspath("verification_outputs")
    if rank == 0:
        os.makedirs(base_folder, exist_ok=True)
        # cleanup previous run
        shutil.rmtree(os.path.join(base_folder, "checkpoint_verify_test"), ignore_errors=True)

    dist.barrier()

    manager = CheckpointManager(
        dataloader=dataloader,
        model_parts=[model],
        optimizers=optimizer,
        lr_schedulers=scheduler,
        states=states,
        checkpoint_config=ckpt_config,
        sd_adapter=None,
        base_folder=base_folder
    )
    
    # Create dummy data for loss verification
    input_data = torch.randn(5, 10, device=device)
    target_data = torch.randn(5, 10, device=device)
    criterion = nn.MSELoss()
    
    # Compute initial loss
    with torch.no_grad():
        output_before = model(input_data)
        loss_before = criterion(output_before, target_data)
    print(f"Loss before save: {loss_before.item()}")

    # Save
    print("Saving checkpoint...")
    manager.save(curr_step=1)
    
    # Modify model to ensure we aren't just getting lucky
    with torch.no_grad():
        model.weight.fill_(2.0)
        states["extra_state"].fill_(999)
        
    print(f"Model modified. Weights now: {model.weight[0][0].item()}")
    
    # Load
    print("Loading checkpoint...")
    success = manager.load(step=1)
    if not success:
        print("Failed to load checkpoint!")
        sys.exit(1)
    
    # Verify
    print("Verifying...")
    # Check weight
    w_val = model.weight[0][0].item()
    s_val = states["extra_state"].item()

    failed = False
    if abs(w_val - 1.0) > 1e-5:
        print(f"FAIL: Weight mismatch. Expected 1.0, got {w_val}")
        failed = True
    else:
        print("PASS: Weights matched.")

    if abs(s_val - 123) > 1e-5:
         print(f"FAIL: State mismatch. Expected 123, got {s_val}")
         failed = True
    else:
         print("PASS: States matched.")

    # Compute loss after load
    with torch.no_grad():
        output_after = model(input_data)
        loss_after = criterion(output_after, target_data)
    print(f"Loss after load: {loss_after.item()}")
    
    if abs(loss_before.item() - loss_after.item()) > 1e-5:
        print(f"FAIL: Loss mismatch! Before: {loss_before.item()}, After: {loss_after.item()}")
        failed = True
    else:
        print("PASS: Loss matched.")
         
    manager.close()
    dist.destroy_process_group()
    
    if failed:
        sys.exit(1)
    else:
        print("Checkpoint verification SUCCESSFUL.")

if __name__ == "__main__":
    run_test()
