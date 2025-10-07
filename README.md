# <img width="50px" src="https://github.com/wwhenxuan/S2Generator/blob/main/images/sum.png?raw=true"> SymTime NeurIPS 2025 <img width="20%" align="right" src="https://github.com/wwhenxuan/S2Generator/blob/main/images/S2Generator_logo.png?raw=true">

This code is the official PyTorch implementation of our NeurIPS'25 paper: **Synthetic Series-Symbol Data Generation for Time Series Foundation Models**.

<div align="center">

[![ICLR](https://img.shields.io/badge/NeurIPS'25-SymTime-orange)]() [![PyPI version](https://badge.fury.io/py/s2generator.svg)](https://pypi.org/project/s2generator/) [![Python](https://img.shields.io/badge/python-3.10+-blue?logo=python)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-blue)](https://pytorch.org/)

[Paper]() | [Quickstart](#Quickstart) | [Poster]() | [Blog]() | [Citation](#Citation)

</div>

## Introduction

**CATCH**, a framework based on frequency patching, flexibly utilizing the channel correlations to reconstruct all the frequency spectrums in a fine-grained way to achieve remarkable detection accuracy. Technically,  we propose a **Channel Fusion Module** (CFM), which features a patch-wise **mask generator** and the masked-attention mechanism. Driven by a bi-level multi-objective optimization algorithm, the CFM is encouraged to iteratively discover appropriate patch-wise channel correlations and **cluster similar channels in the hidden spaces while isolate the adverse effects from irrelevant channels**, which provides both the **capacity and robustness** of the attention mechanism.

<div style="text-align: center;">
    <img src="configs/images/S2Generator_SymTime.png" alt="SymTime" style="zoom:80%;" />
</div>

## Quickstart

### Installation

First create a Python virtual environment (preferably version 3.10.15), then install the required dependencies by running the following command:

```
pip install -r requirements.txt
```

### Data preparation

Prepare Data. You can obtained the well pre-processed datasets from [OneDrive](https://1drv.ms/u/c/801ce36c4ff3f93b/EVTDLHyvegpEn_Oxa6ZiuFIBjTsKk6m9JldUqWDqvrVCnQ?e=P2T3Vc) or [BaiduCloud](https://pan.baidu.com/s/1W7UoAWKZjoukSZ74FTipYA?pwd=2255). (This may take some time, please wait patiently.) Then place the downloaded data under the folder `./dataset`. 

### Train and evaluate model

- To see the model structure of CATCH,  [click here](./ts_benchmark/baselines/catch/CATCH.py).
- We provide the experiment scripts for CATCH and other baselines under the folder `./scripts/multivariate_detection`. For example you can reproduce a experiment result as the following:

```shell
sh ./scripts/multivariate_detection/detect_label/MSL_script/CATCH.sh

sh ./scripts/multivariate_detection/detect_score/MSL_script/CATCH.sh
```



## Results

### Main Results


### Benchmark Results

Extensive experiments on 10 real-world datasets and 12 synthetic datasets demonstrate that CATCH achieves state-of-the-art performance. We show the main results of all the 10 real-world datasets, and report the mean results of the 6 types of synthetic datasets:

<div style="text-align: center;">
    <img src="configs/images/finetune_benchmark_results.png" alt="benchmark" style="zoom:80%;" />
</div>


### 


## Setup for Running Baseline Models
If you want to test all baseline models, please refer to the Time Series Anomaly Detection Benchmark [TAB](https://github.com/decisionintelligence/TAB):


## Citation

If you find this repo useful, please cite our paper.

```


```


## Contact



3.10.15