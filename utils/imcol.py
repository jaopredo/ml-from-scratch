import torch
from math import floor
from typing import Optional


def im2col(X: torch.Tensor, k: int, s: int) -> torch.Tensor:
    if X.dim() != 3:
        raise ValueError(f"The data must have 3 dimensions: (channels, height and width), but the function got a tensor with {X.dim()} dimensions")

    X_col: Optional[torch.Tensor] = None

    _, heigth, width = X.shape

    width_limit = floor((width-k)/s + 1)
    heigth_limit = floor((heigth-k)/s + 1)

    ellapsed_columns: list[torch.Tensor] = []

    for j in range(0, heigth_limit):      
        for i in range(0, width_limit):
            image_window = X[:, s * j : s * j + k, s * i : s * i + k]

            image_window = image_window.reshape(-1,1)

            ellapsed_columns.append(image_window)

    X_col = torch.cat(ellapsed_columns, dim=1)

    return X_col


def col2im(X_col: torch.Tensor, output_size: tuple[int, int], k: int, s: int) -> torch.Tensor:
    if X_col.dim() != 2:
        raise ValueError(f"Expected a 2D tensor, got {X_col.dim()} dimensions")

    height, width = output_size
    rows, n_columns = X_col.shape

    if rows % (k * k) != 0:
        raise ValueError(f"Number of rows ({rows}) is not a multiple of k*k ({k * k})")

    channels = rows // (k * k)

    height_limit = floor((height - k) / s) + 1
    width_limit = floor((width - k) / s) + 1

    if n_columns != height_limit * width_limit:
        raise ValueError(
            f"Expected {height_limit * width_limit} columns for output_size={output_size}, got {n_columns}"
        )

    X = torch.zeros((channels, height, width), dtype=X_col.dtype, device=X_col.device)

    column = 0
    for j in range(height_limit):
        for i in range(width_limit):
            window = X_col[:, column].reshape(channels, k, k)
            X[:, s * j : s * j + k, s * i : s * i + k] += window
            column += 1

    return X
