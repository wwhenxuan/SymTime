# -*- coding: utf-8 -*-
"""
Created on 2024/10/20 10:59
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/SymTime
"""
import os
import argparse
import torch
from exp import Exp_Short_Term_Forecast
from utils.print_args import print_args
import random
import numpy as np

parser = argparse.ArgumentParser(description="SymTime-Short_Term_Forecasting")

# basic config
parser.add_argument("--task_name", type=str, default="short_term_forecast")
parser.add_argument("--is_training", type=int, default=1, help="status")
parser.add_argument("--dataset_name", type=str, default=f"m4", help="model id")
parser.add_argument("--model", type=str, default=f"SymTime")
parser.add_argument("--model_id", type=str, default=f"ETTh1")
parser.add_argument(
    "--pretrain_path", type=str, default="./models/params/finetuning.pth"
)
parser.add_argument("--pretrain_id", type=str, default="zero")

# data loader
parser.add_argument("--data", type=str, default="m4", help="dataset type")
parser.add_argument(
    "--root_path", type=str, default="./datasets/m4/", help="root path of the data file"
)
parser.add_argument("--data_path", type=str, default="m4", help="data file")
parser.add_argument(
    "--features",
    type=str,
    default="M",
    help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
)
parser.add_argument(
    "--target", type=str, default="OT", help="target feature in S or MS task"
)
parser.add_argument(
    "--freq",
    type=str,
    default="m",
    help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
)
parser.add_argument(
    "--checkpoints",
    type=str,
    default="./checkpoints/",
    help="location of model checkpoints",
)

# do patching
parser.add_argument(
    "--forward_layers", type=int, default=3, help="the feed forward layers numbers"
)
parser.add_argument("--patch_len", type=int, default=16, help="patching length")
parser.add_argument("--stride", type=int, default=4, help="patching stride")
parser.add_argument(
    "--padding_patch", type=bool, default=True, help="padding the last patching"
)
parser.add_argument("--out_dropout", type=float, default=0.1, help="the output dropout")
parser.add_argument(
    "--use_avg", type=bool, default=True, help="use moving average decomposition"
)
parser.add_argument(
    "--moving_avg", type=int, default=25, help="window size of moving average"
)

# forecasting task
parser.add_argument(
    "--seasonal_patterns", type=str, default="Yearly", help="subset for M4"
)
parser.add_argument(
    "--inverse", action="store_true", help="inverse output data", default=False
)

# model define
parser.add_argument("--enc_in", type=int, default=1, help="encoder input size")
parser.add_argument("--dec_in", type=int, default=1, help="decoder input size")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
parser.add_argument(
    "--embed",
    type=str,
    default="timeF",
    help="time features encoding, options:[timeF, fixed, learned]",
)
parser.add_argument("--activation", type=str, default="gelu", help="activation")
parser.add_argument("--individual", type=bool, default=False)

# optimization
parser.add_argument(
    "--num_workers", type=int, default=1, help="data loader num workers"
)
parser.add_argument("--itr", type=int, default=1, help="experiments times")
parser.add_argument("--train_epochs", type=int, default=12, help="train epochs")
parser.add_argument(
    "--batch_size", type=int, default=8, help="batch size of train input data"
)
parser.add_argument("--patience", type=int, default=5, help="early stopping patience")
parser.add_argument(
    "--learning_rate", type=float, default=0.0002, help="optimizer learning rate"
)
parser.add_argument("--des", type=str, default="test", help="exp description")
parser.add_argument("--loss", type=str, default="SMAPE", help="loss function")
parser.add_argument("--lradj", type=str, default="type2", help="adjust learning rate")
parser.add_argument(
    "--use_amp",
    action="store_true",
    help="use automatic mixed precision training",
    default=False,
)

# GPU
parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
parser.add_argument("--gpu", type=int, default=0, help="gpu")
parser.add_argument(
    "--use_multi_gpu", action="store_true", help="use multiple gpus", default=False
)
parser.add_argument(
    "--devices", type=str, default="0,1,2,3", help="device ids of multile gpus"
)

# metrics (dtw)
parser.add_argument(
    "--use_dtw", type=bool, default=False, help="the controller of using dtw metric"
)

# Augmentation
parser.add_argument(
    "--augmentation_ratio", type=int, default=0, help="How many times to augment"
)
parser.add_argument("--seed", type=int, default=2025, help="Randomization seed")

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() else False

# Set the random seed for reproducibility
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(" ", "")
    device_ids = args.devices.split(",")
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print("Args in experiment:")
print_args(args)

# setting record of experiments
exp = Exp_Short_Term_Forecast(args)  # set experiments
setting = "{}_{}_{}_{}_moving_avg{}_patch_len{}_stride{}_batch_size{}_learning_rate{}_lradj{}_seed{}_{}".format(
    args.task_name,
    args.model_id,
    args.model,
    args.data,
    args.moving_avg,
    args.patch_len,
    args.stride,
    args.batch_size,
    args.learning_rate,
    args.lradj,
    args.seed,
    args.pretrain_id,
)

print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
exp.train(setting)

print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
exp.test(setting)
torch.cuda.empty_cache()
