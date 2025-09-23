# -*- coding: utf-8 -*-
"""
Created on 2024/9/28 10:10
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/SymTime
"""
import os
from os import path

import torch
import pandas as pd
from matplotlib import pyplot as plt

from .tools import time_now

from typing import Tuple, Union, Callable, Dict, List


class Logging(object):
    """The interface for logging experimental results"""

    def __init__(self, is_pretrain: bool, logging_path: str, datasets: List) -> None:
        # Determine whether it is pre-training
        self.is_pretrain = is_pretrain

        # Datasets excluded or used
        self.datasets = datasets

        # The address where the recording is performed
        self.logging_path = logging_path

        # Get the data dictionary and specific methods of recording
        self.dict, self.logging_epoch = self.init_logging()

        # Create a TXT file that can be written
        self.text = create_txt_file(
            file_path=self.logging_path, file_name="pretrain.txt"
        )

    def init_logging(self) -> Tuple[Dict, Callable]:
        """Returns a dictionary and method of the corresponding form according to the training type"""
        # If it is pre-training
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
        self,
        epoch: int,
        loss: Union[float, torch.Tensor],
        loss_mtm: Union[float, torch.Tensor],
        loss_mlm: Union[float, torch.Tensor],
        loss_t2s: Union[float, torch.Tensor],
        loss_s2t: Union[float, torch.Tensor],
    ) -> None:
        """Logging the training process of the pre-trained model"""
        self.dict["time"].append(time_now())  # Get the current time
        self.dict["epoch"].append(epoch)  # Add the current training Epoch
        self.dict["loss"].append(loss)  # Get the current unsupervised pre-training loss
        self.dict["loss_mtm"].append(loss_mtm)
        self.dict["loss_mlm"].append(loss_mlm)
        self.dict["loss_t2s"].append(loss_t2s)
        self.dict["loss_s2t"].append(loss_s2t)
        self.logging_txt(epoch, loss, loss_mtm, loss_mlm, loss_t2s, loss_s2t)

    def logging_txt(
        self,
        epoch: int,
        loss: Union[float, torch.Tensor],
        loss_mtm: Union[float, torch.Tensor],
        loss_mlm: Union[float, torch.Tensor],
        loss_t2s: Union[float, torch.Tensor],
        loss_s2t: Union[float, torch.Tensor],
    ) -> None:
        """Write the results to txt file"""
        content = f"epoch={epoch}, loss={loss}, loss_mtm={loss_mtm}, loss_mlm={loss_mlm}, loss_t2s={loss_t2s}, loss_s2t={loss_s2t}"
        write_to_txt(file_path=self.text, content=content)

    def dict2csv(self) -> None:
        """Write the recorded dictionary into a csv file"""
        df = pd.DataFrame(self.dict)
        df.to_csv(path.join(self.logging_path, "logging.csv"), index=False)

    def plot_results(self) -> None:
        """Function for visualizing experimental results"""

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


def create_txt_file(file_path: str, file_name: str) -> str:
    """
    Creates a TXT file in the specified directory.

    :param file_path: The directory path where the file is to be created.
    :param file_name: The name of the file to be created (including the .txt extension).
    """
    if not os.path.exists(file_path):
        assert OSError
    # Full file path
    full_file_path = os.path.join(file_path, file_name)
    # Create and open a file
    with open(full_file_path, "w", encoding="utf-8") as file:
        pass  # Create a file without writing anything
    return full_file_path


def write_to_txt(file_path: str, content: str, mode: str = "a") -> None:
    """
    Writes content to a TXT file based on the passed parameters.

    :param file_path: The full path to the file
    :param content: The content to be written
    :param mode: The write mode. Defaults to 'a' (append mode). 'w' overwrites the existing content.
    """
    with open(file_path, mode, encoding="utf-8") as file:
        file.write(content + "\n")  # Write the content and add a newline at the end
