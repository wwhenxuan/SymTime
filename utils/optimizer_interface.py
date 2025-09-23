# -*- coding: utf-8 -*-
"""
Load the optimizer module,
including learning rate warmup and dynamic learning rate adjustment

Created on 2024/9/23 16:39
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/SymTime
"""
from torch import Tensor
from torch import optim
from torch.optim import Optimizer
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LRScheduler

from colorama import Fore, Style

from typing import Optional, List


class OptimInterface(object):
    """
    The General Interface for Loading Optimizers,
    including Learning Rate Warmup and Dynamic Learning Rate Adjustment
    """

    def __init__(self, args, accelerator) -> None:
        self.accelerator = accelerator
        # Get the optimizer used
        self.optimizer = args.optimizer

        # Methods for obtaining predictions and dynamic learning rate adjustment
        self.warmup, self.scheduler = args.warmup, args.scheduler

        # Get the number of warm-up rounds and the total number of training rounds
        self.num_epochs, self.warmup_epochs = args.num_epochs, args.warmup_epochs
        self.pct_start = self.warmup_epochs / self.num_epochs

        # Get optimizer configuration parameters
        self.learning_rate = args.learning_rate
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.beta1, self.beta2 = args.beta1, args.beta2
        self.eps = args.eps
        self.amsgrad = args.amsgrad

        # Parameters for dynamic learning rate adjustment
        self.step_size = args.step_size
        self.gamma = args.gamma
        self.cycle_momentum = args.cycle_momentum
        self.base_momentum = args.base_momentum
        self.max_momentum = args.max_momentum
        self.anneal_strategy = args.anneal_strategy

    def load_optimizer(self, parameters: Optional[Tensor | List]) -> Optimizer:
        """How to get the optimizer"""
        self.accelerator.print(
            Fore.RED
            + f"Now is loading the optimizer: {self.optimizer}"
            + Style.RESET_ALL,
            end=" -> ",
        )
        if self.optimizer == "SGD":
            # Using stochastic gradient descent
            return self.load_SGD(parameters)

        elif self.optimizer == "Adam":
            # Using Adam optimizer
            return self.load_Adam(parameters)

        elif self.optimizer == "AdamW":
            # Using the AdamW optimizer
            return self.load_AdamW(parameters)

        else:
            raise ValueError("args.optimizer inputs error!")

    def load_scheduler(
        self, optimizer: Optimizer, loader_len: int = None
    ) -> LRScheduler:
        """Methods for obtaining dynamic learning rate adjustments"""
        self.accelerator.print(
            Fore.RED
            + f"Now is loading the scheduler: {self.scheduler}"
            + Style.RESET_ALL,
            end=" -> ",
        )
        # If OneCycle is used, it comes with a learning rate warm-up process
        if self.scheduler == "OneCycle":
            return self.load_OneCycleLR(optimizer, loader_len)

        # First load the learning rate warm-up method
        warmup_scheduler = self.load_warmup(optimizer)

        # Reloading dynamic learning rate adjustment method
        if self.scheduler == "StepLR":
            dynamic_scheduler = self.load_StepLR(optimizer)
        elif self.scheduler == "ExponLR":
            dynamic_scheduler = self.load_ExponentialLR(optimizer)
        else:
            raise ValueError("args.scheduler inputs error!")

        # Combining learning rate warmup and dynamic learning rate adjustment
        return lr_scheduler.SequentialLR(
            optimizer,
            [warmup_scheduler, dynamic_scheduler],
            milestones=[self.warmup_epochs, self.num_epochs],
        )

    def load_warmup(self, optimizer: Optimizer) -> LRScheduler:
        """Get the adjustment method of learning rate warm-up"""
        if self.warmup == "LinearLR":
            # Use linear learning rate growth
            scheduler = lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.0,
                end_factor=1.0,
                total_iters=self.warmup_epochs,
            )
            self.load_successfully()
            return scheduler
        else:
            raise ValueError("args.warmup fill in error")

    def load_SGD(self, parameters: Tensor) -> Optimizer:
        """Methods for obtaining a stochastic gradient descent optimizer"""
        optimizer = optim.SGD(parameters, lr=self.learning_rate, momentum=self.momentum)
        self.load_successfully()
        return optimizer

    def load_Adam(self, parameters: Tensor) -> Optimizer:
        """The Interface to Get the Adam optimizer"""
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
        """The Interface to Get the AdamW optimizer"""
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
        """Get the learning rate exponential decay factor"""
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        self.load_successfully()
        return scheduler

    def load_StepLR(self, optimizer: Optimizer) -> LRScheduler:
        """A method for obtaining dynamic learning rate attenuation for each certain number of Epochs in StepLR"""
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma
        )
        self.load_successfully()
        return scheduler

    def load_OneCycleLR(
        self, optimizer: Optimizer, loader_len: int = None
    ) -> LRScheduler:
        """Obtaining a periodic cyclic dynamic learning rate adjustment method"""
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
