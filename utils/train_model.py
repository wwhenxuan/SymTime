# -*- coding: utf-8 -*-
"""
Created on 2024/9/23 15:56
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
用于模型训练的代码
"""
import os
from os import path
import sys
from time import sleep

import torch
from torch import nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from accelerate import Accelerator
from typing import Callable, Tuple, Dict
from colorama import Fore, Style
from tqdm import tqdm

from .tools import makedir
from .logging import Logging


class PreTrainer(object):
    """用于模型预训练的接口"""

    def __init__(
        self,
        args,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: Callable,
        scheduler: LRScheduler,
        accelerator: Accelerator,
        data_interface,
    ):
        self.args = args
        # 获取训练轮数
        self.num_epochs = args.num_epochs
        self.save_epochs = args.save_epochs
        # 获取模型和训练数据集
        self.model = model
        # 获取神经网络的优化器
        self.optimizer = optimizer
        # 获取损失函数
        self.criterion = criterion
        # 获取动态调整学习率
        self.scheduler = scheduler
        # 获取协同加速器
        self.accelerator = accelerator
        # 记录当前的进程号
        self.process_index = self.accelerator.process_index
        # 获取训练集和验证集
        self.data_interface = data_interface
        # 获取当前训练设备
        self.device = self.accelerator.device
        if self.process_index == 0:  # 只有主进程记录损失并保存模型参数
            # 获取保存模型和参数的地址
            self.main_path, self.params_path = self.init_path()
            # 创建模型训练的Logging模块
            self.logging = Logging(
                is_pretrain=True, logging_path=self.main_path, datasets=[]
            )

    def fit(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """训练模型拟合数据"""
        self.accelerator.print(
            Fore.GREEN + "Starting SymTime Model Pretraining..." + Style.RESET_ALL
        )
        train_loss = torch.zeros(self.num_epochs, device=self.device)
        train_loss_mtm = torch.zeros(self.num_epochs, device=self.device)
        train_loss_mlm = torch.zeros(self.num_epochs, device=self.device)
        train_loss_t2s = torch.zeros(self.num_epochs, device=self.device)
        train_loss_s2t = torch.zeros(self.num_epochs, device=self.device)
        for idx, epoch in enumerate(range(1, self.num_epochs + 1), 0):
            """这里是开始了一个Epoch"""
            num_samples = 0  # 这一个Epoch中遍历的累计样本数目
            for ii in range(1, len(self.data_interface) + 1):
                """在一个Epoch中要遍历读取完所有的数据"""
                self.accelerator.print(
                    Fore.RED + "Now is loading pretraining data" + Style.RESET_ALL,
                    end=" -> ",
                )
                train_loader = self.data_interface.get_dataloader()
                train_loader = self.accelerator.prepare_data_loader(
                    train_loader, device_placement=True
                )
                sleep(2)
                self.accelerator.print(
                    Fore.GREEN + "successfully loaded!" + Style.RESET_ALL
                )
                self.model.train()
                data_loader = tqdm(train_loader, file=sys.stdout)
                for step, (time, time_mask, sym_ids, sym_mask) in enumerate(
                    data_loader, 1
                ):
                    self.optimizer.zero_grad()
                    num_samples += time.shape[0]
                    # 直接在模型正向传播的过程中获得损失
                    loss_mtm, loss_mlm, loss_t2s, loss_s2t = self.model(
                        time, time_mask, sym_ids, sym_mask
                    )
                    # 获取和整合误差
                    loss = loss_mtm + loss_mlm + (loss_t2s + loss_s2t) / 2
                    # 误差的反向传播
                    self.accelerator.backward(loss)
                    # 参数的更新
                    self.optimizer.step()
                    # 检查模型损失
                    check_loss(loss, train_type="Pretrain")
                    # 计算这个epoch的累计损失
                    train_loss[idx] += loss.item()
                    train_loss_mtm[idx] += loss_mtm.item()
                    train_loss_mlm[idx] += loss_mlm.item()
                    train_loss_t2s[idx] += loss_t2s.item()
                    train_loss_s2t[idx] += loss_s2t.item()
                    data_loader.desc = (
                        "["
                        + Fore.GREEN
                        + f"Epoch {epoch}"
                        + Style.RESET_ALL
                        + "] "
                        + "Loss="
                        + Fore.GREEN
                        + f"{round(train_loss[idx].item() / num_samples, 6)}"
                        + Style.RESET_ALL
                        + f" loss_mtm: {round(train_loss_mtm[idx].item() / num_samples, 6)}, loss_mlm: {round(train_loss_mlm[idx].item() / num_samples, 6)}, "
                        f"loss_t2s: {round(train_loss_t2s[idx].item() / num_samples, 6)}, loss_s2t: {round(train_loss_s2t[idx].item() / num_samples, 6)}"
                    )
                    # 动态调整学习率
                    self.scheduler.step()
                # 释放训练优化器的内存
                self.accelerator.clear(train_loader)
            # 记录最终损失的变化
            train_loss[idx] = train_loss[idx] / num_samples
            train_loss_mtm[idx] = train_loss_mtm[idx] / num_samples
            train_loss_mlm[idx] = train_loss_mlm[idx] / num_samples
            train_loss_t2s[idx] = train_loss_t2s[idx] / num_samples
            train_loss_s2t[idx] = train_loss_s2t[idx] / num_samples
            if epoch % self.save_epochs == 0:
                # 保存一次预训练模型的参数
                self.save_model(loss=train_loss[idx], epoch=epoch)
            # Logging训练过程 登记当前的epoch和最后的损失
            self.logging_epoch(
                epoch,
                train_loss[idx],
                train_loss_mtm[idx],
                train_loss_mlm[idx],
                train_loss_t2s[idx],
                train_loss_s2t[idx],
            )
        # """这部分可以调整一下专门写一个函数来执行"""
        # # 记录logging结果
        # self.logging.dict2csv()
        # self.logging.plot_results()
        return (
            train_loss,
            train_loss_mtm,
            train_loss_mlm,
            train_loss_t2s,
            train_loss_s2t,
        )

    def init_path(self) -> Tuple:
        """获取本次预训练保存模型和logging的地址"""
        # 保存模型的目录
        save_path = self.args.save_path
        # 判断保存目录下有多少个文件
        num_folder = len(os.listdir(save_path))
        # 创建本次保存模型的文件夹
        folder_name = f"exp{num_folder + 1}"
        makedir(save_path, folder_name)
        # 更新保存目录的主要地址
        main_path = path.join(save_path, folder_name)
        # 创建保存模型参数的文件夹
        makedir(main_path, "params")
        params_path = path.join(main_path, "params")
        print(f"Attention the logging path is {main_path}")
        return main_path, params_path

    def save_model(self, epoch: int, loss: Tensor) -> None:
        """保存模型的参数"""
        if self.process_index == 0:
            self.accelerator.print(
                Fore.RED + "Now is saving the pretrained params" + Style.RESET_ALL,
                end=" -> ",
            )
            save_name = f"{epoch}_{round(loss.item(), 4)}.pth"
            torch.save(
                self.model.time_encoder.state_dict(),
                path.join(self.params_path, save_name),
            )
            self.accelerator.print(Fore.GREEN + "successfully saved!" + Style.RESET_ALL)

    def logging_epoch(
        self,
        epoch: int,
        train_loss: Tensor,
        train_loss_mtm: Tensor,
        train_loss_mlm: Tensor,
        train_loss_t2s: Tensor,
        train_loss_s2t: Tensor,
    ) -> None:
        """记录一个Epoch的训练损失变化情况"""
        gather_train_loss = self.accelerator.gather(train_loss).mean().item()
        gather_train_loss_mtm = self.accelerator.gather(train_loss_mtm).mean().item()
        gather_train_loss_mlm = self.accelerator.gather(train_loss_mlm).mean().item()
        gather_train_loss_t2s = self.accelerator.gather(train_loss_t2s).mean().item()
        gather_train_loss_s2t = self.accelerator.gather(train_loss_s2t).mean().item()
        if self.process_index == 0:
            # 记录一个Epoch下的所有进程的平均损失
            self.logging.logging_epoch(
                epoch,
                gather_train_loss,
                gather_train_loss_mtm,
                gather_train_loss_mlm,
                gather_train_loss_t2s,
                gather_train_loss_s2t,
            )


def init_path(save_path) -> str:
    """获取本次预训练保存模型和logging的地址"""
    # 判断保存目录下有多少个文件
    num_folder = len(os.listdir(save_path))
    # 创建本次保存模型的文件夹
    folder_name = f"exp{num_folder + 1}"
    makedir(save_path, folder_name)
    # 更新保存目录的主要地址
    main_path = path.join(save_path, folder_name)
    # 创建保存模型参数的文件夹
    return main_path


def check_loss(loss: Tensor, train_type: str) -> None:
    """检查训练和验证的损失避免梯度爆炸"""
    if not torch.isfinite(loss):
        print(
            Fore.RED + f"{train_type} now occurs ERROR: non-finite loss, end training!"
        )
        sys.exit(1)
