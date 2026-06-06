import torch

print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
cap = torch.cuda.get_device_capability(0)
print(f"capability sm_{cap[0]}{cap[1]}")
a = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, 2048, device="cuda")
c = (a @ b).sum().item()
torch.cuda.synchronize()
print("kernel OK, matmul reduced:", round(c, 2))
