# -*- coding: utf-8 -*-
"""
Get the interface module of the loss function.

Created on 2024/9/23 17:00
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/SymTime
"""
from torch import nn
from typing import Callable


def get_criterion(name: str = "MSE") -> Callable:
    """Get the interface configuration of the neural network loss function"""
    if name == "MSE":
        return nn.MSELoss
    elif name == "MAE":
        return nn.L1Loss
    elif name == "CEL":
        return nn.CrossEntropyLoss
    elif name == "Huber":
        return nn.SmoothL1Loss
    elif name == "Cos":
        return nn.CosineEmbeddingLoss
    else:
        raise ValueError("The loss function name is incorrect.!")
