# -*- coding: utf-8 -*-
"""
Created on 2024/9/23 17:00
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
获取使用的损失函数的接口模块
"""
from torch import nn
from typing import Callable


def get_criterion(name: str = "MSE") -> Callable:
    """获取神经网络损失函数的接口配置"""
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
        raise ValueError("损失函数名称填写错误!")
