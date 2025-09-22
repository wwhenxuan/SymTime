# -*- coding: utf-8 -*-
"""
Created on 2024/9/23 16:39
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
加载优化器的模块, 包括学习率预热和动态学习率调整
"""
from torch import Tensor
from torch import optim
from torch.optim import Optimizer
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LRScheduler

from colorama import Fore, Style

from typing import Optional, List


class OptimInterface(object):
    """获得优化器的接口"""

    def __init__(self, args, accelerator) -> None:
        self.accelerator = accelerator
        # 获取使用的优化器
        self.optimizer = args.optimizer
        # 获取预测和动态学习率调整方法
        self.warmup, self.scheduler = args.warmup, args.scheduler
        # 获取预热轮数和总训练轮数
        self.num_epochs, self.warmup_epochs = args.num_epochs, args.warmup_epochs
        self.pct_start = self.warmup_epochs / self.num_epochs

        # 获取优化器配置的参数
        self.learning_rate = args.learning_rate
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.beta1, self.beta2 = args.beta1, args.beta2
        self.eps = args.eps
        self.amsgrad = args.amsgrad

        # 动态学习率调整的参数
        self.step_size = args.step_size
        self.gamma = args.gamma
        self.cycle_momentum = args.cycle_momentum
        self.base_momentum = args.base_momentum
        self.max_momentum = args.max_momentum
        self.anneal_strategy = args.anneal_strategy

    def load_optimizer(self, parameters: Optional[Tensor | List]) -> Optimizer:
        """获取优化器的方法"""
        self.accelerator.print(
            Fore.RED
            + f"Now is loading the optimizer: {self.optimizer}"
            + Style.RESET_ALL,
            end=" -> ",
        )
        if self.optimizer == "SGD":
            # 使用随机梯度下降法
            return self.load_SGD(parameters)
        elif self.optimizer == "Adam":
            # 使用Adam优化器
            return self.load_Adam(parameters)
        elif self.optimizer == "AdamW":
            # 使用AdamW优化器
            return self.load_AdamW(parameters)
        else:
            raise ValueError("args.optimizer填写错误")

    def load_scheduler(
        self, optimizer: Optimizer, loader_len: int = None
    ) -> LRScheduler:
        """获得动态学习率调整的方法"""
        self.accelerator.print(
            Fore.RED
            + f"Now is loading the scheduler: {self.scheduler}"
            + Style.RESET_ALL,
            end=" -> ",
        )
        # 如果使用的是OneCycle则自带学习率预热过程
        if self.scheduler == "OneCycle":
            return self.load_OneCycleLR(optimizer, loader_len)
        # 先加载学习率预热的方法
        warmup_scheduler = self.load_warmup(optimizer)
        # 再加载动态学习率调整的方法
        if self.scheduler == "StepLR":
            dynamic_scheduler = self.load_StepLR(optimizer)
        elif self.scheduler == "ExponLR":
            dynamic_scheduler = self.load_ExponentialLR(optimizer)
        else:
            raise ValueError("args.scheduler填写错误")
        # 将学习率预热和动态学习率调整两部分结合
        return lr_scheduler.SequentialLR(
            optimizer,
            [warmup_scheduler, dynamic_scheduler],
            milestones=[self.warmup_epochs, self.num_epochs],
        )

    def load_warmup(self, optimizer: Optimizer) -> LRScheduler:
        """获得学习率预热的调整方法"""
        if self.warmup == "LinearLR":
            # 使用线性学习率增长
            scheduler = lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.0,
                end_factor=1.0,
                total_iters=self.warmup_epochs,
            )
            self.load_successfully()
            return scheduler
        else:
            raise ValueError("args.warmup填写错误")

    def load_SGD(self, parameters: Tensor) -> Optimizer:
        """获得随机梯度下降优化器的方法"""
        optimizer = optim.SGD(parameters, lr=self.learning_rate, momentum=self.momentum)
        self.load_successfully()
        return optimizer

    def load_Adam(self, parameters: Tensor) -> Optimizer:
        """获得Adam优化器的方法"""
        optimizer = optim.Adam(
            parameters,
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay,
            eps=self.eps,
            amsgrad=self.amsgrad,
        )
        self.load_successfully()
        return optimizer

    def load_AdamW(self, parameters: Tensor) -> Optimizer:
        """获得AdamW优化器的接口"""
        optimizer = optim.AdamW(
            parameters,
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay,
            eps=self.eps,
            amsgrad=self.amsgrad,
        )
        self.load_successfully()
        return optimizer

    def load_ExponentialLR(self, optimizer: Optimizer) -> LRScheduler:
        """获得学习率指数衰减因子"""
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        self.load_successfully()
        return scheduler

    def load_StepLR(self, optimizer: Optimizer) -> LRScheduler:
        """获得StepLR每个一定的Epochs数目进行动态学习率衰减的方法"""
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma
        )
        self.load_successfully()
        return scheduler

    def load_OneCycleLR(
        self, optimizer: Optimizer, loader_len: int = None
    ) -> LRScheduler:
        """获得周期性循环动态学习率调整方法"""
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=loader_len * self.num_epochs,
            pct_start=self.pct_start,
            anneal_strategy=self.anneal_strategy,
            cycle_momentum=self.cycle_momentum,
            base_momentum=self.base_momentum,
            max_momentum=self.max_momentum,
        )
        self.load_successfully()
        return scheduler

    def load_successfully(self) -> None:
        """note that the optimizer / scheduler has been loaded successfully"""
        self.accelerator.print(Fore.GREEN + "successfully loaded!" + Style.RESET_ALL)
