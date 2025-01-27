import torch

t = torch.rand(4, 4, 4, 4)

print(t)
print(t.transpose(2, 1))
print(t.transpose(-2, -1))

