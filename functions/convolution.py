import torch
from core import Module


class Convolution(Module):
    def __init__(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        pass

    def backward(self, dout: torch.Tensor) -> torch.Tensor:
        pass

    def to(self, device: torch.device):
        pass
