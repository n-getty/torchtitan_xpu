import torch
import intel_extension_for_pytorch
import torch.distributed as dist
import os

os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

print(f"Backend map: {torch.distributed.Backend.default_device_backend_map}")

print("Init PG ccl...")
try:
    dist.init_process_group("xccl")
    print("Init Successful")
    # Verify broadcast
    t = torch.tensor([1]).to("xpu:0")
    dist.broadcast(t, 0)
    print("Broadcast Successful")
except Exception as e:
    print(f"Init Failed: {e}")
