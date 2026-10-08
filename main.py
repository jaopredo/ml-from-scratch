from PIL import Image
from torchvision.transforms import ToTensor
from layers.convolutional import ConvolutionalLayer
import torch

layer = ConvolutionalLayer(
    kernel_size=3,
    features=1,
    padding=2
)

x = torch.tensor([
    [1,2,3],
    [4,5,6],
    [7,8,9]
], dtype=torch.float32)
x = x.unsqueeze(0)  # Adiciona a dimensão do canal
x = torch.concat([x, x, x], dim=0)  # Adiciona mais canais para simular uma imagem RGB
print(x.size())

# img = Image.open("imagem.png")
# x = ToTensor()(img)

# print(x.size())

layer.forward(x)