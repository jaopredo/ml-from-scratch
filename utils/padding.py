import torch


def apply_padding(X: torch.Tensor, padding: int) -> torch.Tensor:
    """Receives an tensor torch and adds zeros to the borders. The amount of zeros
    is equal to the `padding` parameter

    Args:
        X (torch.Tensor): Image that will be padded. The image needs, at least, 2 dimensions and at most, 3 dimensions
        padding (int): Size of the padding to be added

    Returns:
        torch.Tensor: Image with `padding` zeros in the borders
    
    Examples:
>>> image = torch.tensor([
    [1,2,3],
    [4,5,6],
    [7,8,9],
])

>>> transformed = apply_padding(image, 1)
>>> transformed
[
    [0,0,0,0,0]
    [0,1,2,3,0]
    [0,4,5,6,0]
    [0,7,8,9,0]
    [0,0,0,0,0]
]
    """
    if len(X.shape) not in (2,3):
        raise ValueError(f"The passed image do not have the required dimension. Dimension must be 2 or three, instead, got {X.dim()}")
    
    if padding == 0:
        return X

    # Desempacotando as dimensões da imagem
    shape = X.shape

    if len(shape) == 2:
        height, width = shape

        left_right_dim = (height, padding)
        top_down_dim = (padding,width+2*padding)

        height_dim=1
        width_dim=0
    else:
        depth, height, width = shape

        left_right_dim = (depth, height, padding)
        top_down_dim = (depth, padding, width+2*padding)

        height_dim=2
        width_dim=1
    
    left_right = torch.zeros(left_right_dim)
    top_down = torch.zeros(top_down_dim)

    padded_sideways = torch.cat((left_right, X, left_right), dim=height_dim)
    
    return torch.cat((top_down, padded_sideways, top_down), dim=width_dim)


def remove_padding(X: torch.Tensor, padding: int) -> torch.Tensor:
    """Receives a tensor and cuts `padding` columns and `padding` rows in each side of it. Be awared to not apply it in
    integral images, the `apply_padding` function adds zeros on the margins, but this function do not check if the margins
    are composed of only zeros

    Args:
        X (torch.Tensor): Image that will be cut. Image must have 2 dimensions or 3 (with the dimension of channels)
        padding (int): How many of the margins will be cut

    Returns:
        torch.Tensor: Tensor with its margins cutted
    
    Examples:
>>> image = torch.tensor([
[0,0,0,0,0]
[0,1,2,3,0]
[0,4,5,6,0]
[0,7,8,9,0]
[0,0,0,0,0]
])

>>> transformed = remove_padding(image, 1)
>>> transformed
[
    [1,2,3],
    [4,5,6],
    [7,8,9],
]
    """
    if len(X.shape) not in (2,3):
        raise ValueError(f"The passed image do not have the required dimension. Dimension must be 2 or three, instead, got {X.dim()}")
        
    if padding == 0:
        return X

    shape = X.shape

    if len(shape) == 2:
        height, width = shape
        return X[padding-1:height+padding-1,padding-1:width+padding-1]
    else:
        _, height, width = shape
        return X[:, padding-1:height+padding-1,padding-1:width+padding-1]
