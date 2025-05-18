import torch


ten = torch.tensor([1, 2, 3, 4, 5]).unsqueeze(0)
print(ten)
print(ten.shape)