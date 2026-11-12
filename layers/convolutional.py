import torch
from core import Module, Parameter
from typing import Optional
from math import sqrt
from utils import im2col, apply_padding, col2im, remove_padding


class ConvolutionalHead(Module):
    def __init__(
        self,
        channels_in: int,
        kernel_size: int = 3,
        device: torch.device = torch.device("cpu")
    ) -> None:
        self.__device = device
        
        self.channels_in: int = channels_in  # Quantidade de canais que a imagem do forward terá
        self.kernel_size: int = kernel_size

        # Variável para guardar o cache da operação
        self.x: Optional[torch.Tensor] = None

        fan_in = (self.kernel_size**2)*self.channels_in

        self.kernel: Parameter = Parameter(
            torch.randn(
                fan_in
            ) * sqrt(2 / fan_in)
        )

    def to(self, device: torch.device) -> Module:
        self.__device = device
        
        self.x = self.x.to(self.__device) if self.x is not None else None
        self.kernel = self.kernel.to(self.__device)

        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2

        self.x = x

        return self.x.T @ self.kernel

    def backward(self, dout: torch.Tensor) -> torch.Tensor:
        assert dout.device.type == self.__device.type
        assert self.x is not None

        self.kernel.accumulate_grad(self.x @ dout)

        return torch.outer(self.kernel.data, dout)


class ConvolutionalLayer(Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        kernel_size: int=3,
        padding: int = 0,
        stride: int = 1,
        device: torch.device = torch.device("cpu")
    ):
        self.__device = device

        self.channels_in: int = channels_in
        self.channels_out: int = channels_out
        self.kernel_size: int = kernel_size
        self.padding = padding
        self.stride = stride

        self.x: Optional[torch.Tensor] = None
        self.original_padded_shape: Optional[tuple[int,int]] = None

        self.kernels: list[ConvolutionalHead] = [
            ConvolutionalHead(
                self.channels_in,
                self.kernel_size,
                self.__device
            ) for _ in range(self.channels_out)
        ]

    def to(self, device: torch.device):
        self.__device = device
        for kernel in self.kernels:
            kernel.to(self.__device)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3

        x_col = im2col(apply_padding(x, self.padding), self.kernel_size, self.stride)

        _, height, width = x.shape

        self.x = x_col
        self.original_padded_shape = (height + 2 * self.padding, width + 2 * self.padding)

    def backward(self, dout):
        pass
