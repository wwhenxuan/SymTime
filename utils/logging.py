# -*- coding: utf-8 -*-
"""
Created on 2024/9/28 10:10
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
用于logging实验结果的mok
"""
import os
from os import path
import pandas as pd
from matplotlib import pyplot as plt

from .tools import time_now

from typing import Tuple, Callable, Dict, List


class Logging(object):
    """logging实验结果的模块接口"""

    def __init__(self, is_pretrain: bool, logging_path: str, datasets: List) -> None:
        # 判断是否是预训练
        self.is_pretrain = is_pretrain
        # 排除或是使用的数据集
        self.datasets = datasets
        # 进行logging的地址
        self.logging_path = logging_path
        # 获得logging的数据字典和具体的方法
        self.dict, self.logging_epoch = self.init_logging()
        # 创建一个可以写入的TXT文件
        self.text = create_txt_file(
            file_path=self.logging_path, file_name="pretrain.txt"
        )

    def init_logging(self) -> Tuple[Dict, Callable]:
        """根据训练类型返回对应形式的字典和方法"""
        # 如果是预训练
        return {
            "time": [],
            "epoch": [],
            "loss": [],
            "loss_mtm": [],
            "loss_mlm": [],
            "loss_t2s": [],
            "loss_s2t": [],
        }, self.logging_pretrain

    def logging_pretrain(
        self, epoch: int, loss: float, loss_mtm, loss_mlm, loss_t2s, loss_s2t
    ) -> None:
        """logging预训练模型的训练过程"""
        self.dict["time"].append(time_now())  # 获取当前时间
        self.dict["epoch"].append(epoch)  # 添加当前训练的Epoch
        self.dict["loss"].append(loss)  # 获取当前无监督预训练的损失
        self.dict["loss_mtm"].append(loss_mtm)
        self.dict["loss_mlm"].append(loss_mlm)
        self.dict["loss_t2s"].append(loss_t2s)
        self.dict["loss_s2t"].append(loss_s2t)
        self.logging_txt(epoch, loss, loss_mtm, loss_mlm, loss_t2s, loss_s2t)

    def logging_txt(
        self, epoch: int, loss: float, loss_mtm, loss_mlm, loss_t2s, loss_s2t
    ) -> None:
        """写入txt"""
        content = f"epoch={epoch}, loss={loss}, loss_mtm={loss_mtm}, loss_mlm={loss_mlm}, loss_t2s={loss_t2s}, loss_s2t={loss_s2t}"
        write_to_txt(file_path=self.text, content=content)

    def dict2csv(self) -> None:
        """将logging得到的字典写入csv文件中"""
        df = pd.DataFrame(self.dict)
        df.to_csv(path.join(self.logging_path, "logging.csv"), index=False)

    def plot_results(self) -> None:
        """实验结果可视化的函数"""
        fig, ax = plt.subplots(figsize=(10, 4))
        if self.is_pretrain is True:
            # ax.plot(self.dict["epoch"], self.dict["loss"], color='royalblue', label='loss')
            ax.plot(
                self.dict["epoch"],
                self.dict["loss_mtm"],
                color="tomato",
                label="loss_mtm",
            )
            ax.plot(
                self.dict["epoch"],
                self.dict["loss_mlm"],
                color="royalblue",
                label="loss_mlm",
            )
            ax.plot(
                self.dict["epoch"],
                self.dict["loss_t2s"],
                color="#6FAE45",
                label="loss_t2s",
            )
            ax.plot(
                self.dict["epoch"],
                self.dict["loss_s2t"],
                color="darkorange",
                label="loss_s2t",
            )
            ax.set_xlabel("num_epoch", fontsize=16)
            ax.set_ylabel("loss", fontsize=16)
            ax.legend(loc="best", fontsize=15)
        else:
            ax_twinx = ax.twinx()
            ax.set_xlabel("num_epoch", fontsize=16)
            ax.set_ylabel("loss", fontsize=16)
            ax_twinx.set_ylabel("metric", fontsize=16)
            ax.plot(
                self.dict["epoch"],
                self.dict["train_loss"],
                color="royalblue",
                label="Train Loss",
            )
            ax.plot(
                self.dict["epoch"],
                self.dict["test_loss"],
                color="tomato",
                label="Test Loss",
            )
            ax.legend(loc="best", fontsize=15)
            ax_twinx.plot(
                self.dict["epoch"],
                self.dict["train_metric"],
                color="royalblue",
                label="Train Metric",
            )
            ax_twinx.plot(
                self.dict["epoch"],
                self.dict["test_metric"],
                color="tomato",
                label="Test Metric",
            )
        fig.savefig(
            path.join(self.logging_path, "plot.jpg"), bbox_inches="tight", dpi=900
        )


def create_txt_file(file_path, file_name):
    """
    在指定目录下创建一个Txt文件。
    :param file_path: 要创建文件的目录路径
    :param file_name: 要创建的文件名（包括.txt扩展名）
    """
    if not os.path.exists(file_path):
        assert OSError
    # 完整的文件路径
    full_file_path = os.path.join(file_path, file_name)
    # 创建并打开文件
    with open(full_file_path, "w", encoding="utf-8") as file:
        pass  # 创建文件，不需要写入任何内容
    return full_file_path


def write_to_txt(file_path, content, mode="a"):
    """
    根据传入的参数内容向Txt文件中写入内容。
    :param file_path: 文件的完整路径
    :param content: 要写入的内容
    :param mode: 写入模式，默认为'a'（追加模式），如果为'w'则覆盖原有内容
    """
    with open(file_path, mode, encoding="utf-8") as file:
        file.write(content + "\n")  # 写入内容，并在末尾添加换行符
