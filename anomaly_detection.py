# -*- coding: utf-8 -*-
"""
Created on 2024/10/19 21:20
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/SymTime
"""
import argparse
import torch
from exp import Exp_Anomaly_Detection
from utils.print_args import print_args
import random
import numpy as np


parser = argparse.ArgumentParser(description="SymTime-Anomaly_Detection")

# basic config
parser.add_argument("--task_name", type=str, default="anomaly_detection")
parser.add_argument("--is_training", type=int, default=1, help="status")
parser.add_argument("--dataset_name", type=str, default="SWAT", help="model id")
parser.add_argument("--model", type=str, default="SymTime")
parser.add_argument(
    "--pretrain_path", type=str, default="./models/params/finetuning.pth"
)
parser.add_argument("--pretrain_id", type=str, default="zero")

# data loader
parser.add_argument("--data", type=str, default="SWAT", help="datasets type")
parser.add_argument(
    "--root_path",
    type=str,
    default="./datasets/SWaT/",
    help="root path of the data file",
)
parser.add_argument(
    "--data_path", type=str, default="electricity.csv", help="data file"
)
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
    default="h",
    help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
)
parser.add_argument(
    "--checkpoints",
    type=str,
    default="./checkpoints/",
    help="location of model checkpoints",
)
parser.add_argument("--individual", type=bool, default=False)

parser.add_argument("--anomaly_ratio", type=float, default=0.5)
# do patching
parser.add_argument(
    "--forward_layers", type=int, default=3, help="the feed forward layers numbers"
)
parser.add_argument(
    "--patch_len", type=int, default=16, help="Divide the length of the patch"
)
parser.add_argument(
    "--stride", type=int, default=1, help="The division step size of patching"
)
parser.add_argument(
    "--padding_patch", type=bool, default=True, help="Whether to fill the last Patch"
)
parser.add_argument(
    "--out_dropout",
    type=float,
    default=0.1,
    help="Dropout of the final output of the model",
)
parser.add_argument(
    "--use_avg", type=bool, default=True, help="use moving average decomposition"
)
parser.add_argument(
    "--moving_avg", type=int, default=25, help="window size of moving average"
)
# forecasting task
parser.add_argument("--seq_len", type=int, default=100, help="input sequence length")
parser.add_argument("--label_len", type=int, default=0, help="start token length")
parser.add_argument(
    "--pred_len", type=int, default=0, help="prediction sequence length"
)
parser.add_argument(
    "--seasonal_patterns", type=str, default="Monthly", help="subset for M4"
)
parser.add_argument(
    "--inverse", action="store_true", help="inverse output data", default=False
)

# model define
parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
parser.add_argument(
    "--embed",
    type=str,
    default="timeF",
    help="time features encoding, options:[timeF, fixed, learned]",
)
parser.add_argument("--activation", type=str, default="gelu", help="activation")
parser.add_argument(
    "--channel_independence",
    type=int,
    default=1,
    help="0: channel dependence 1: channel independence for FreTS model",
)
parser.add_argument(
    "--use_norm", type=int, default=1, help="whether to use normalize; True 1 False 0"
)

# optimization
parser.add_argument("--enc_in", type=int, default=51, help="encoder input size")
parser.add_argument(
    "--num_workers", type=int, default=5, help="data loader num workers"
)
parser.add_argument("--itr", type=int, default=1, help="experiments times")
parser.add_argument("--train_epochs", type=int, default=5, help="train epochs")
parser.add_argument(
    "--batch_size", type=int, default=4, help="batch size of train input data"
)
parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
parser.add_argument(
    "--learning_rate", type=float, default=0.0001, help="optimizer learning rate"
)
parser.add_argument("--des", type=str, default="test", help="exp description")
parser.add_argument("--loss", type=str, default="MSE", help="loss function")
parser.add_argument("--lradj", type=str, default="type1", help="adjust learning rate")
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

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

if __name__ == "__main__":
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
        torch.cuda.set_device(args.gpu)

    print("Args in experiment:")
    print_args(args)

    exp = Exp_Anomaly_Detection(args)  # set experiments
    setting = "{}_{}_{}_{}_seq_len{}_moving_avg{}_forward_layers{}_patch_len{}_stride{}_batch_size{}_learning_rate{}_lradj{}_seed{}_{}".format(
        args.task_name,
        args.dataset_name,
        args.model,
        args.data,
        args.seq_len,
        args.use_avg,
        args.forward_layers,
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
