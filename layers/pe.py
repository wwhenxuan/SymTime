# -*- coding: utf-8 -*-
"""
Created on 2024/9/16 12:31
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import torch
from torch import nn
from torch import Tensor
import math


class PositionalEmbedding(nn.Module):
    """Adding the positional encoding to the input for Transformer"""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super(PositionalEmbedding, self).__init__()
        # Calculate the positional encoding once in the logarithmic space.
        pe = torch.zeros(
            max_len, d_model
        ).float()  # Initialize a tensor of zeros with shape (max_len, d_model) to store positional encodings
        pe.requires_grad = (
            False  # Positional encodings do not require gradients as they are fixed
        )

        position = (
            torch.arange(0, max_len).float().unsqueeze(1)
        )  # Generate a sequence from 0 to max_len-1 and add a dimension at the 1st axis
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()  # Calculate the divisor term in the positional encoding formula

        pe[:, 0::2] = torch.sin(
            position * div_term
        )  # Apply the sine function to the even columns of the positional encoding matrix
        pe[:, 1::2] = torch.cos(
            position * div_term
        )  # Apply the cosine function to the odd columns of the positional encoding matrix

        pe = pe.unsqueeze(
            0
        )  # Add a batch dimension, changing the shape to (1, max_len, d_model)
        self.register_buffer(
            "pe", pe
        )  # Register the positional encodings as a buffer, which will not be updated as model parameters

    def forward(self, x: Tensor) -> Tensor:
        # Return the first max_len positional encodings that match the length of input x
        return x + self.pe[:, : x.size(1)]
