# -*- coding: utf-8 -*-
"""
Created on 2024/11/4 11:07
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import torch
from torch import nn
from torch import Tensor
from typing import Optional


class Flatten_Heads(nn.Module):
    """Integrate the final output of the time series encoder"""

    def __init__(self,
                 individual: bool,
                 n_vars: int,
                 nf: int,
                 patch_num: int,
                 targets_window: int,
                 head_dropout: int = 0,
                 cls_token: Optional[bool] = True) -> None:
        super().__init__()
        # Whether to output in a channel-independent manner
        self.individual = individual
        self.n_vars = n_vars
        self.patch_num = patch_num
        # Whether to take the [CLS] Token
        if cls_token is True:
            self.patch_num += 1

        if self.individual is True:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.linears.append(nn.Linear(nf * self.patch_num, targets_window))
                self.dropouts.append(nn.Dropout(head_dropout))
                self.flattens.append(nn.Flatten(start_dim=-2))
        else:
            self.linear = nn.Linear(nf * self.patch_num, targets_window)
            self.dropout = nn.Dropout(head_dropout)
            self.flatten = nn.Flatten(start_dim=-2)

    def forward(self, x: Tensor) -> Tensor:  # [batch_size, n_vars, d_model, patch_num]
        if self.individual is True:
            x_out = []
            for i in range(self.n_vars):
                # 将某一通道的维数展平
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x
