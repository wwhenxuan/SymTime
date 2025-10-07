# -*- coding: utf-8 -*-
"""
Created on 2024/10/13 10:04
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/SymTime
"""
from os import path
import yaml

from torch import nn
from typing import List
from models import SymTime_pretrain as SymTime
from colorama import Fore, Style


class ModelInterface(object):
    """
    A general interface for loading models,
    including model pre-training and model fine-tuning
    """

    def __init__(self, args, accelerator) -> None:
        self.args = args

        # Accelerator object used
        self.accelerator = accelerator

        # Determine whether to pre-train the model
        self.is_pretrain = args.is_pretrain

        # Determine the model to use
        self.model_type = args.model
        self.model = self.load_pretrain()

    def load_pretrain(self) -> nn.Module:
        """Load the initialized model for pre-training"""
        self.accelerator.print(
            Fore.RED + "Now is loading model" + Style.RESET_ALL, end=" -> "
        )

        # Get the address of the configuration file
        configs_path = path.join("configs", f"SymTime_{self.model_type}.yaml")
        with open(configs_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        model = SymTime(
            config,
            context_window=self.args.context_window,
            time_mask_ratio=self.args.time_mask_ratio,
            sym_mask_ratio=self.args.sym_mask_ratio,
        )
        self.accelerator.print(Fore.GREEN + "successfully loaded!" + Style.RESET_ALL)
        return model

    def trainable_params(self) -> List:
        """Get trainable model parameters"""
        train_params = []
        for params in self.model.parameters():
            if params.requires_grad is True:
                # Parameters for which gradients are not calculated are frozen parameters
                train_params.append(params)
        return train_params
