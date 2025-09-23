# -*- coding: utf-8 -*-
"""
For the pre-training of SymTime using self-supervised learning.

Created on 2024/10/9 17:20
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/SymTime
"""
import random
import os
import argparse

import numpy as np
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs

from data_provider import PreTrainDataLoader
from exp import Exp_Pretraining
from utils import ModelInterface, OptimInterface, get_criterion

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

parser = argparse.ArgumentParser(description="SymTime-pretrain")

# basic config
parser.add_argument(
    "--is_pretrain",
    type=bool,
    default=True,
    help="Whether to perform model pre - training",
)
parser.add_argument(
    "--model", type=str, default="base", help="Model type used: small, base, large"
)
parser.add_argument(
    "--context_window", type=int, default=256, help="Original length of data loaded"
)
parser.add_argument(
    "--time_mask_ratio",
    type=float,
    default=0.40,
    help="Masking ratio of signal patches",
)
parser.add_argument(
    "--sym_mask_ratio",
    type=float,
    default=0.15,
    help="Masking ratio of natural language symbols",
)
parser.add_argument(
    "--patch_len",
    type=int,
    default=16,
    help="Length of each patch for data embedding in Transformer",
)
parser.add_argument(
    "--stride",
    type=int,
    default=None,
    help="Stride size for patching using the sliding windows. If None, non-overlapping patches are used",
)

# Data path related parameters
parser.add_argument(
    "--data_path",
    type=str,
    default=r"./datasets/pretrain_data/",
    help="Path to store training data",
)
parser.add_argument(
    "--save_path", type=str, default="./logging", help="Path to save the model"
)
parser.add_argument(
    "--number", type=int, default=2, help="Number of data read per round"
)
parser.add_argument(
    "--llm_name", type=str, default="DistilBert", help="Large - language model used"
)

# Parameters related to model optimization
parser.add_argument(
    "--num_epochs",
    type=int,
    default=1000,
    help="Number of rounds for model pre - training",
)
parser.add_argument(
    "--warmup_epochs",
    type=int,
    default=50,
    help="Number of rounds for learning rate warm - up",
)
parser.add_argument(
    "--save_epochs",
    type=int,
    default=1,
    help="Save the model once every few rounds of training",
)
parser.add_argument(
    "--batch_size", type=int, default=48, help="Batch size used for training"
)
parser.add_argument(
    "--num_workers", type=int, default=1, help="Number of workers for data loader"
)
parser.add_argument(
    "--shuffle",
    type=bool,
    default=True,
    help="Whether to shuffle the order of training samples during training",
)
parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer used")
parser.add_argument("--criterion", type=str, default="MSE", help="Loss function used")
parser.add_argument(
    "--warmup", type=str, default="LinearLR", help="Learning rate warm - up method used"
)
parser.add_argument(
    "--scheduler",
    type=str,
    default="OneCycle",
    help="Dynamic learning rate adjustment method used",
)
parser.add_argument(
    "--learning_rate", type=float, default=1e-5, help="Training learning rate"
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    help="Momentum size used in stochastic gradient descent",
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=1e-4,
    help="L2 regularization strength suitable for Adam",
)
parser.add_argument(
    "--beta1",
    type=float,
    default=0.9,
    help="Decay rate of first - order moment estimate, degree of retention of historical gradients, default 0.9",
)
parser.add_argument(
    "--beta2",
    type=float,
    default=0.999,
    help="Decay rate of second - order moment estimate, conducive to improving stability, default 0.999",
)
parser.add_argument(
    "--eps", type=float, default=1e-8, help="Constant to prevent division by zero"
)
parser.add_argument(
    "--amsgrad", type=bool, default=False, help="Whether to use the AMSgrad variant"
)
parser.add_argument(
    "--step_size",
    type=int,
    default=10,
    help="Number of Epochs in StepLR that multiply the learning rate by gamma at regular intervals",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
    help="Learning rate decay multiplier for StepLR and ExponLR",
)
parser.add_argument(
    "--cycle_momentum",
    type=bool,
    default=True,
    help="Whether to use periodic momentum adjustment strategy in OneCycle",
)
parser.add_argument(
    "--base_momentum",
    type=float,
    default=0.85,
    help="Base momentum value set during learning rate adjustment",
)
parser.add_argument(
    "--max_momentum",
    type=float,
    default=0.95,
    help="Momentum value set when learning rate reaches maximum",
)
parser.add_argument(
    "--anneal_strategy",
    type=str,
    default="cos",
    help="Learning rate decay strategy used: cos or linear",
)

args = parser.parse_args()

fix_seed = 2025
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

if __name__ == "__main__":
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config="./configs/ds_config.json")
    accelerator = Accelerator(
        device_placement=True,
        gradient_accumulation_steps=1,
        cpu=False,
        # kwargs_handlers=[ddp_kwargs],
        # deepspeed_plugin=deepspeed_plugin
    )
    interface = {
        "data": PreTrainDataLoader(args),
        "model": ModelInterface(args, accelerator),
        "criterion": get_criterion(name=args.criterion),
        "optimizer": OptimInterface(args, accelerator),
    }
    train_data = interface["data"]
    data_loader = train_data.get_dataloader()
    train_data.pointer = 0
    model = interface["model"].model
    train_params = interface["model"].trainable_params()
    criterion = interface["criterion"]
    optimizer = interface["optimizer"].load_optimizer(parameters=train_params)
    scheduler = interface["optimizer"].load_scheduler(
        optimizer, loader_len=len(data_loader)
    )

    model, optimizer, scheduler, data_loader = accelerator.prepare(
        model, optimizer, scheduler, data_loader
    )

    trainer = Exp_Pretraining(
        args, model, optimizer, criterion, scheduler, accelerator, train_data
    )
    (
        train_loss,
        train_loss_mtm,
        train_loss_mlm,
        train_loss_t2s,
        train_loss_s2t,
    ) = trainer.fit()
