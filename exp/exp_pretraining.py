# -*- coding: utf-8 -*-
"""
Created on 2024/9/23 15:56
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/SymTime
"""
import os
from os import path
import sys
from time import sleep

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from accelerate import Accelerator
from typing import Callable, Tuple, Dict
from colorama import Fore, Style
from tqdm import tqdm

from utils.tools import makedir
from utils.logging import Logging


class Exp_PreTraining(object):
    """The Interface for the pre-training of SymTime"""

    def __init__(
        self,
        args,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: Callable,
        scheduler: LRScheduler,
        accelerator: Accelerator,
        data_interface,
    ) -> None:
        self.args = args
        # Get the number of training rounds
        self.num_epochs = args.num_epochs
        self.save_epochs = args.save_epochs

        # Get the model and training dataset
        self.model = model

        # Get the optimizer for the neural network
        self.optimizer = optimizer

        # Get the loss function
        self.criterion = criterion

        # Get dynamically adjusted learning rate
        self.scheduler = scheduler

        # Get the Synergy Accelerator
        self.accelerator = accelerator

        # Record the current process ID
        self.process_index = self.accelerator.process_index

        # Get the training set and validation set
        self.data_interface = data_interface
        # Get the current training device
        self.device = self.accelerator.device

        if (
            self.process_index == 0
        ):  # Only the main process records losses and saves model parameters
            # Get the address where the model and parameters are saved
            self.main_path, self.params_path = self.init_path()
            # Create a Logging module for model training
            self.logging = Logging(
                is_pretrain=True, logging_path=self.main_path, datasets=[]
            )

    def fit(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training model to fit data"""
        self.accelerator.print(
            Fore.GREEN + "Starting SymTime Model Pre-training..." + Style.RESET_ALL
        )

        # Record the four loss functions used by SymTime in self-supervised pre-training
        train_loss = torch.zeros(self.num_epochs, device=self.device)
        train_loss_mtm = torch.zeros(self.num_epochs, device=self.device)
        train_loss_mlm = torch.zeros(self.num_epochs, device=self.device)
        train_loss_t2s = torch.zeros(self.num_epochs, device=self.device)
        train_loss_s2t = torch.zeros(self.num_epochs, device=self.device)

        for idx, epoch in enumerate(range(1, self.num_epochs + 1), 0):
            """Here begins an Epoch"""
            num_samples = 0
            # The cumulative number of samples traversed in this Epoch

            for ii in range(1, len(self.data_interface) + 1):
                """In one Epoch, all data must be traversed and read"""
                self.accelerator.print(
                    Fore.RED + "Now is loading pre-training data" + Style.RESET_ALL,
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
                    # Obtain the loss directly during the forward propagation of the model
                    loss_mtm, loss_mlm, loss_t2s, loss_s2t = self.model(
                        time, time_mask, sym_ids, sym_mask
                    )

                    # Acquiring and integrating errors
                    loss = loss_mtm + loss_mlm + (loss_t2s + loss_s2t) / 2
                    # Back propagation of error
                    self.accelerator.backward(loss)

                    # Parameter update
                    self.optimizer.step()
                    # Checking model loss
                    check_loss(loss, train_type="Pretrain")

                    # Calculate the cumulative loss of this epoch
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
                    # Dynamically adjust learning rate
                    self.scheduler.step()
                # Freeing up memory for training optimizers
                self.accelerator.clear(train_loader)

            # Record changes in final loss
            train_loss[idx] = train_loss[idx] / num_samples
            train_loss_mtm[idx] = train_loss_mtm[idx] / num_samples
            train_loss_mlm[idx] = train_loss_mlm[idx] / num_samples
            train_loss_t2s[idx] = train_loss_t2s[idx] / num_samples
            train_loss_s2t[idx] = train_loss_s2t[idx] / num_samples

            if epoch % self.save_epochs == 0:
                # Save the parameters of a pre-trained model
                self.save_model(loss=train_loss[idx], epoch=epoch)

            # Logging training process to register the current epoch and the final loss
            self.logging_epoch(
                epoch,
                train_loss[idx],
                train_loss_mtm[idx],
                train_loss_mlm[idx],
                train_loss_t2s[idx],
                train_loss_s2t[idx],
            )

        return (
            train_loss,
            train_loss_mtm,
            train_loss_mlm,
            train_loss_t2s,
            train_loss_s2t,
        )

    def init_path(self) -> Tuple[str, str]:
        """Get the address of the pre-training saved model and logging"""

        # Directory where the model is saved
        save_path = self.args.save_path

        # Determine how many files are in the save directory
        num_folder = len(os.listdir(save_path))

        # Create a folder to save the model
        folder_name = f"exp{num_folder + 1}"
        makedir(save_path, folder_name)

        # Update the main address of the save directory
        main_path = path.join(save_path, folder_name)

        # Create a folder to save model parameters
        makedir(main_path, "params")
        params_path = path.join(main_path, "params")

        print(f"Attention the logging path is {main_path}")
        return main_path, params_path

    def save_model(self, epoch: int, loss: torch.Tensor) -> None:
        """Save the parameters of SymTime time encoder in pre-training"""
        if self.process_index == 0:
            self.accelerator.print(
                Fore.RED + "Now is saving the pre-trained params" + Style.RESET_ALL,
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
        train_loss: torch.Tensor,
        train_loss_mtm: torch.Tensor,
        train_loss_mlm: torch.Tensor,
        train_loss_t2s: torch.Tensor,
        train_loss_s2t: torch.Tensor,
    ) -> None:
        """Record the changes in training loss of an Epoch"""
        gather_train_loss = self.accelerator.gather(train_loss).mean().item()
        gather_train_loss_mtm = self.accelerator.gather(train_loss_mtm).mean().item()
        gather_train_loss_mlm = self.accelerator.gather(train_loss_mlm).mean().item()
        gather_train_loss_t2s = self.accelerator.gather(train_loss_t2s).mean().item()
        gather_train_loss_s2t = self.accelerator.gather(train_loss_s2t).mean().item()
        if self.process_index == 0:
            # Record the average loss of all processes under an Epoch
            self.logging.logging_epoch(
                epoch,
                gather_train_loss,
                gather_train_loss_mtm,
                gather_train_loss_mlm,
                gather_train_loss_t2s,
                gather_train_loss_s2t,
            )


def init_path(save_path) -> str:
    """Get the address of the pre-training saved model and logging"""
    # Determine how many files are in the save directory
    num_folder = len(os.listdir(save_path))

    # Create a folder to save the model
    folder_name = f"exp{num_folder + 1}"
    makedir(save_path, folder_name)

    # Update the main address of the save directory
    main_path = path.join(save_path, folder_name)

    # Create a folder to save model parameters
    return main_path


def check_loss(loss: torch.Tensor, train_type: str) -> None:
    """Check training and validation losses to avoid exploding gradients"""
    if not torch.isfinite(loss):
        print(
            Fore.RED + f"{train_type} now occurs ERROR: non-finite loss, end training!"
        )
        sys.exit(1)
