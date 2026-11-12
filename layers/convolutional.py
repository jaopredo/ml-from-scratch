import torch
from core import Module, Parameter
from typing import Optional
from math import floor


class Kernel2D(Module):
    def __init__(
        self
    ):
        pass

    def forward(self):
        pass

    def backward(self, dout: torch.Tensor) -> torch.Tensor:
        pass

    def to(self, device: torch.device) -> "Kernel2D":
        return self


class ConvolutionalLayer(Module):
    def __init__(
        self
    ):
        pass

    def to(self, device: torch.device):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass                                                        

    def backward(self, dout):
        pass
