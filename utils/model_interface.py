# -*- coding: utf-8 -*-
"""
Created on 2024/10/13 10:04
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/SymTime
获取模型的接口
"""
from os import path
import yaml

from torch import nn
from typing import List
from models import SymTime_pretrain as SymTime
from colorama import Fore, Style


class ModelInterface(object):
    """加载模型的通用接口，包括模型预训练和模型微调"""

    def __init__(self, args, accelerator) -> None:
        self.args = args
        # 使用的Accelerator对象
        self.accelerator = accelerator
        # 判断是否进行模型的预训练
        self.is_pretrain = args.is_pretrain
        # 判断使用模型的型号
        self.model_type = args.model
        self.model = self.load_pretrain()

    def load_pretrain(self) -> nn.Module:
        """加载初始化的模型进行预训练"""
        self.accelerator.print(
            Fore.RED + "Now is loading model" + Style.RESET_ALL, end=" -> "
        )
        # 获取配置文件的地址
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
        """获取可训练的模型参数"""
        train_params = []
        for params in self.model.parameters():
            if params.requires_grad is True:
                # 不计算梯度的参数为冻结参数
                train_params.append(params)
        return train_params
