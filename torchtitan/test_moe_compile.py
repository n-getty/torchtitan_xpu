import torch
import torch.nn.functional as F
import time
import intel_extension_for_pytorch

device = "xpu"
dim = 256
hidden_dim = 1024
num_experts = 8

class MoELayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.randn(num_experts, dim, hidden_dim, device=device))
        self.w2 = torch.nn.Parameter(torch.randn(num_experts, hidden_dim, dim, device=device))
        self.w3 = torch.nn.Parameter(torch.randn(num_experts, dim, hidden_dim, device=device))

    def forward(self, x, splits):
        # x: (total_tokens, dim)
        # splits: tensor of ints
        x_split = torch.split(x, splits.tolist(), dim=0)
        outs = []
        for i, expert_input in enumerate(x_split):
            w1 = self.w1[i]
            w2 = self.w2[i]
            w3 = self.w3[i]
            h = F.silu(torch.matmul(expert_input, w1))
            h = h * torch.matmul(expert_input, w3)
            h = torch.matmul(h, w2)
            outs.append(h)
        return torch.cat(outs, dim=0)

model = MoELayer().to(device)
compiled_model = torch.compile(model, dynamic=True) # enabling dynamic shape

print("Warming up eager...")
x = torch.randn(32 * 512, dim, device=device)
splits = torch.tensor([2048]*8, device="cpu") # evenly split
model(x, splits)

print("Warming up compiled...")
compiled_model(x, splits)

print("Benchmarking...")
# Generate random splits summing to total
total = 32 * 512
import random

def get_random_splits(total, num):
    splits = [0]*num
    remaining = total
    for i in range(num-1):
        val = random.randint(0, remaining)
        splits[i] = val
        remaining -= val
    splits[-1] = remaining
    return torch.tensor(splits, device="cpu")

# Eager
torch.xpu.synchronize()
start = time.time()
for _ in range(10):
    splits = get_random_splits(total, num_experts)
    model(x, splits)
torch.xpu.synchronize()
print(f"Eager avg time: {(time.time()-start)/10:.4f}s")

# Compiled
torch.xpu.synchronize()
start = time.time()
for _ in range(10):
    splits = get_random_splits(total, num_experts)
    compiled_model(x, splits)
torch.xpu.synchronize()
print(f"Compiled avg time: {(time.time()-start)/10:.4f}s")
