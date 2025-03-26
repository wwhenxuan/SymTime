# -*- coding: utf-8 -*-
"""
Created on 2024/9/16 10:29
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import torch
from torch import nn
from torch import Tensor
from typing import Tuple


class Transpose(nn.Module):
    """Transpose the dimensions of the input tensor"""

    def __init__(self, *dims, contiguous=False) -> None:
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x: Tensor) -> Tensor:
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


def get_batch_norm(d_model: int) -> nn.Module:
    """Get the BatchNorm module for processing the attention mechanism"""
    return nn.Sequential(
        Transpose(1, 2),
        nn.BatchNorm1d(d_model),
        Transpose(1, 2)
    )


def get_activation_fn(activation: str) -> nn.Module:
    """选择使用的激活函数"""
    if callable(activation):
        return activation()
    elif activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')


class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series"""

    def __init__(self, kernel_size: int = 25, stride: int = 1) -> None:
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """Series decomposition instance block"""

    def __init__(self, kernel_size: int) -> None:
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

