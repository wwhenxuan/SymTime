# -*- coding: utf-8 -*-
"""
Created on 2024/9/30 21:27
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/SymTime
"""
# from functools import partial
import numpy as np
import torch
from torch import nn
from layers import TSTEncoder
from layers import Flatten_Heads
from layers import series_decomp


class SymTime(nn.Module):
    """Network architecture used for fine-tuning downstream tasks"""

    def __init__(self, args, configs) -> None:
        super().__init__()
        # Downstream tasks to be completed
        self.task_name = args.task_name
        self.patch_len = args.patch_len
        self.stride = args.stride
        self.padding_patch = args.padding_patch

        # Calculate the number of patches that can be divided.
        self.patch_num = int((args.seq_len - self.patch_len) / self.stride + 1)
        if self.padding_patch is True:
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
            self.patch_num += 1

        # input and output sequence length
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len

        self.n_layers = configs["time_layers"]
        self.forward_layers = args.forward_layers
        self.d_model = configs["d_model"]
        self.n_heads = configs["n_heads"]
        self.d_ff = configs["d_ff"]

        # individual output for the final forecasting heads
        self.individual = args.individual

        self.pretrain_path = args.pretrain_path

        # the dropout for finally outputs
        self.out_dropout = args.out_dropout

        # An encoder for creating time series data
        self.time_encoder = TSTEncoder(
            patch_len=self.patch_len,
            n_layers=self.n_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            norm=configs["norm"],
            attn_dropout=configs["attn_dropout"],
            dropout=configs["dropout"],
            act=configs["act"],
            pre_norm=configs["pre_norm"],
            forward_layers=self.forward_layers,
        )
        # load the pre-training params
        self.load_pretrained()

        # freeze some Transformer layers in time encoder
        for name, param in self.time_encoder.named_parameters():
            # traverse the number of layers that need to be frozen
            for index in range(self.forward_layers, self.n_layers):
                if f"layers.{index}" in name:
                    param.requires_grad = False

        # time series seasonal decompsition
        self.use_avg = args.use_avg
        if self.use_avg is True:
            self.decompsition = series_decomp(kernel_size=args.moving_avg)
            # trend projection alone
            self.projection_trend = nn.Linear(
                in_features=self.seq_len,
                out_features=(
                    args.pred_len if "forecast" in self.task_name else self.seq_len
                ),
            )

        # Develop an interface module for handling downstream tasks
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            self.flatten_head = Flatten_Heads(
                individual=self.individual,
                n_vars=args.enc_in,
                patch_num=self.patch_num,
                nf=self.d_model,
                targets_window=args.pred_len,
                head_dropout=self.out_dropout,
            )
        elif self.task_name == "classification":
            if args.conv1d is True:
                self.conv1d = nn.Conv1d(
                    in_channels=args.enc_in,
                    out_channels=args.out_channels,
                    kernel_size=(3,),
                    stride=(1,),
                    padding=1,
                )
                args.enc_in = args.out_channels
                self.use_conv1d = True
            else:
                self.use_conv1d = False
            self.act = nn.GELU()
            self.ln_proj = nn.LayerNorm(
                self.d_model * (self.patch_num * args.enc_in + 1)
            )
            self.classifier = nn.Linear(
                in_features=self.d_model * (self.patch_num * args.enc_in + 1),
                out_features=args.num_classes,
            )
        elif self.task_name == "anomaly_detection":
            self.flatten_head = Flatten_Heads(
                individual=self.individual,
                n_vars=args.enc_in,
                patch_num=self.patch_num,
                nf=self.d_model,
                targets_window=self.seq_len,
                head_dropout=self.out_dropout,
            )
        elif self.task_name == "imputation":
            self.flatten_head = Flatten_Heads(
                individual=self.individual,
                n_vars=args.enc_in,
                patch_num=self.patch_num,
                nf=self.d_model,
                targets_window=self.seq_len,
                head_dropout=self.out_dropout,
                cls_token=True,
            )
        else:
            raise ValueError("task name wrong!")

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            x_dec = self.forcast(x_enc=x_enc)
        elif self.task_name == "classification":
            x_dec = self.classification(x_enc=x_enc)
        elif self.task_name == "imputation":
            x_dec = self.imputation(x_enc=x_enc)
        else:
            x_dec = self.anomaly_detection(x_enc=x_enc)
        return x_dec

    def forcast(self, x_enc: torch.Tensor) -> torch.Tensor:
        """Forward for long and short term forecasting"""

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # time series decompsition
        if self.use_avg is True:
            seasonal_part, trend_part = self.decompsition(x_enc)
            x_enc = seasonal_part.permute(0, 2, 1)
            # Mapping trend part to target length
            trend_part = trend_part.permute(0, 2, 1)
            trend_part = self.projection_trend(trend_part)
            trend_part = trend_part.permute(0, 2, 1)
        else:
            x_enc = x_enc.permute(0, 2, 1)

        # do patching
        x_enc = self.patching(ts=x_enc)  # [batch_size, num_vars, patch_num, patch_len]
        batch_size, num_vars, patch_num, patch_len = x_enc.size()

        x_enc = torch.reshape(x_enc, [batch_size * num_vars, patch_num, patch_len])
        x_dec = self.time_encoder(x_enc)

        # Restore the original input form independently from the channel
        x_dec = torch.reshape(
            x_dec, [batch_size, num_vars, x_dec.shape[-2], x_dec.shape[-1]]
        )
        x_dec = self.flatten_head(x_dec).permute(
            0, 2, 1
        )  # [batch_size, pred_len, num_vars]

        # add the trend part of the decompsition
        if self.use_avg is True:
            x_dec = x_dec + trend_part

        # De-Normalization from Non-stationary Transformer
        x_dec = x_dec * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        x_dec = x_dec + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return x_dec

    def classification(self, x_enc: torch.Tensor) -> torch.Tensor:
        """Forward for classification task"""

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        x_enc = x_enc.permute(0, 2, 1)  # [batch_size, num_vars, seq_len]

        # Adjusting the input channels through Conv1d
        if self.use_conv1d is True:
            x_enc = self.conv1d(x_enc)  # [batch_size, out_channels, seq_len]

        # do patching and reshape
        x_enc = self.patching(ts=x_enc)  # [batch_size, num_vars, patch_num, patch_len]
        batch_size, num_vars, patch_num, patch_len = x_enc.size()

        # Learning feature through the backbone of Transformer
        x_enc = torch.reshape(
            x_enc, shape=(batch_size, num_vars * patch_num, patch_len)
        )
        x_dec = self.time_encoder(x_enc)

        # Output processing
        x_dec = self.act(x_dec)
        x_dec = torch.reshape(x_dec, shape=(batch_size, -1))
        x_dec = self.ln_proj(x_dec)
        outputs = self.classifier(x_dec)

        return outputs

    def imputation(self, x_enc: torch.Tensor) -> torch.Tensor:
        """The interface for performing time series imputation tasks"""

        #  pre-interpolation from Peri-midFormer
        x_enc_np = x_enc.detach().cpu().numpy()
        zero_indices = np.where(x_enc_np[:, :, :] == 0)
        interpolated_x_enc = np.copy(x_enc_np)
        for sample_idx, time_idx, channel_idx in zip(*zero_indices):
            non_zero_indices = np.nonzero(x_enc_np[sample_idx, :, channel_idx])[0]
            before_non_zero_idx = (
                non_zero_indices[non_zero_indices < time_idx][-1]
                if len(non_zero_indices[non_zero_indices < time_idx]) > 0
                else None
            )
            after_non_zero_idx = (
                non_zero_indices[non_zero_indices > time_idx][0]
                if len(non_zero_indices[non_zero_indices > time_idx]) > 0
                else None
            )
            if before_non_zero_idx is not None and after_non_zero_idx is not None:
                interpolated_value = (
                    x_enc_np[sample_idx, before_non_zero_idx, channel_idx]
                    + x_enc_np[sample_idx, after_non_zero_idx, channel_idx]
                ) / 2
            elif before_non_zero_idx is None:
                interpolated_value = x_enc_np[
                    sample_idx, after_non_zero_idx, channel_idx
                ]
            elif after_non_zero_idx is None:
                interpolated_value = x_enc_np[
                    sample_idx, before_non_zero_idx, channel_idx
                ]
            interpolated_x_enc[sample_idx, time_idx, channel_idx] = interpolated_value

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # time series decompsition
        if self.use_avg is True:
            seasonal_part, trend_part = self.decompsition(x_enc)
            x_enc = seasonal_part.permute(0, 2, 1)

            # Mapping trend part to target length
            trend_part = trend_part.permute(0, 2, 1)
            trend_part = self.projection_trend(trend_part)
            trend_part = trend_part.permute(0, 2, 1)
        else:
            x_enc = x_enc.permute(0, 2, 1)

        # do patching and reshape
        x_enc = self.patching(ts=x_enc)  # [batch_size, n_vars, patch_num, patch_len]
        batch_size, n_vars, patch_num, patch_len = x_enc.size()

        # Process data in a channel-independent manner
        x_enc = torch.reshape(x_enc, shape=(batch_size * n_vars, patch_num, patch_len))

        # After the large model forward propagation part
        x_dec = self.time_encoder(x_enc)  # [batch_size * n_vars, patch_num, d_model]
        x_dec = torch.reshape(
            x_dec, shape=(batch_size, n_vars, x_dec.size(-2), self.d_model)
        )

        # Restore the original output dimension of the model
        x_dec = self.flatten_head(x_dec).permute(
            0, 2, 1
        )  # [batch_size, pred_len, num_vars]

        # add the trend part of the decompsition
        if self.use_avg is True:
            x_dec = x_dec + trend_part

        # De-Normalization from Non-stationary Transformer
        x_dec = x_dec * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        x_dec = x_dec + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))

        return x_dec

    def anomaly_detection(self, x_enc: torch.Tensor) -> torch.Tensor:
        """The interface for time series anomaly detection"""

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # time series decompsition
        if self.use_avg is True:
            seasonal_part, trend_part = self.decompsition(x_enc)
            x_enc = seasonal_part.permute(0, 2, 1)
            # Mapping trend part to target length
            trend_part = trend_part.permute(0, 2, 1)
            trend_part = self.projection_trend(trend_part)
            trend_part = trend_part.permute(0, 2, 1)
        else:
            x_enc = x_enc.permute(0, 2, 1)

        # do patching and reshape
        x_enc = self.patching(ts=x_enc)  # [batch_size, n_vars, patch_num, patch_len]
        batch_size, n_vars, patch_num, patch_len = x_enc.size()

        # Process data in a channel-independent manner
        x_enc = torch.reshape(x_enc, shape=(batch_size * n_vars, patch_num, patch_len))

        # After the large model forward propagation part
        x_dec = self.time_encoder(x_enc)  # [batch_size * n_vars, patch_num, d_model]
        x_dec = torch.reshape(
            x_dec, [batch_size, n_vars, x_dec.shape[-2], x_dec.shape[-1]]
        )

        # Restore the original output dimension of the model
        x_dec = self.flatten_head(x_dec).permute(
            0, 2, 1
        )  # [batch_size, pred_len, num_vars]

        # add the trend part of the decompsition
        if self.use_avg is True:
            x_dec = x_dec + trend_part

        # De-Normalization from Non-stationary Transformer
        x_dec = x_dec * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        x_dec = x_dec + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))

        return x_dec

    def patching(self, ts: torch.Tensor) -> torch.Tensor:
        """Divide the time series into patch"""
        if self.padding_patch is True:
            ts = self.padding_patch_layer(ts)
        ts = ts.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return ts

    def load_pretrained(self) -> None:
        """Loading pre-trained model parameters"""
        print("Now loading pre-trained model params...")
        self.time_encoder.load_state_dict(
            torch.load(self.pretrain_path, weights_only=True)
        )
