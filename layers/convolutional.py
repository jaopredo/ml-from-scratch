import torch
from core import Module, Parameter
from typing import Optional
from math import floor


class ConvolutionalLayer(Module):
    def __init__(
        self,
        kernel_size: int,
        features: int,
        stride: int = 1,
        padding: int = 0,
        device: torch.device = torch.device("cpu")
    ):
        """Inicializa a camada de convolução.

        Args:
            kernel_size (int): Tamanho do kernel da convolução
            features (int): Número de features (canais) de saída da convolução
            stride (int): Passos que o filtro anda na feature map. Defauts to 1.
            padding (int): Quantos zeros adicionar na borda das imagens. Defaults to 0.
            device (torch.device, optional): Dispositivo onde os tensores serão armazenados. Defaults to torch.device("cpu").
        """
        self.__device: torch.device = device

        n_in = (kernel_size**2) * features

        self.__n_in = n_in
        self.__features = features
        self.__kernel_size = kernel_size
        self.__stride = stride
        self.__padding = padding

        # Inicializando o kernel como vetor coluna para posterior
        # aplicação no feature map transformado em imToCol
        self.kernel: Parameter = Parameter(torch.normal(
            mean=0,
            std=(2 / n_in) ** 0.5,
            size=(kernel_size**2, features),
            device=self.__device
        ))

        self.bias: Parameter = Parameter(
            torch.zeros(self.kernel.size())
        )

        self.x: Optional[torch.Tensor] = None

    def to(self, device: torch.device):
        self.__device = device

        self.kernel.to(self.__device)

        self.x = self.x.to(self.__device) if self.x is not None else None

        return self

    def _apply_padding(self, x: torch.Tensor) -> torch.Tensor:
        c, h, w = x.size()
        p = self.__padding
        if p > 0:
            x_padded = torch.concat([
                torch.zeros((c, p, w)),
                x,
                torch.zeros((c, p, w))
            ], dim=1)
            x_padded = torch.concat([
                torch.zeros((c, h+2*p, p)),
                x_padded,
                torch.zeros((c, h+2*p, p))
            ],dim=2)
            return x_padded
        return x

    def _im2col(self, x: torch.Tensor) -> torch.Tensor:
        c, h_in, w_in = x.size()

        h_out = floor((h_in - self.__kernel_size)/self.__stride) + 1
        w_out = floor((w_in - self.__kernel_size)/self.__stride) + 1

        # Versão ingênua
        x_col = torch.zeros(
            (c * self.__kernel_size**2, h_out * w_out),
        )

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.__stride
                h_end = h_start + self.__kernel_size
                w_start = j * self.__stride
                w_end = w_start + self.__kernel_size

                patch = x[:, h_start:h_end, w_start:w_end]
                x_col[:, i*w_out + j] = patch.flatten()

        return x_col

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.x = self._apply_padding(x)
        self.x = self._im2col(self.x)
        print(x)
        print(self.x)
        # print(x)
        # print()
        # print(self._im2col(x))

    def backward(self, dout):
        pass
