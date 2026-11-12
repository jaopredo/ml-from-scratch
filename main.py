from utils import im2col, col2im
import torch

x = torch.tensor([
    [
        [i+1+5*j for i in range(5)] for j in range(5)
    ] for _ in range(1)
], dtype=torch.float32)

print(x)
print(x.shape)
y = im2col(x, 2, 1)
print(col2im(y, (5,5), 2, 1))
