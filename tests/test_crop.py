import torch
from padding import crop

X = torch.rand(1, 1, 5, 5)
R = torch.tensor(
        [[2, 2]], dtype=torch.int16)

x = crop(X.cuda(), R.cuda(), 7, 7)


