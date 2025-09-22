# -*- coding: utf-8 -*-
"""
Created on 2024/9/21 20:20
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/SymTime
获取大语言模型的Tokenizer
"""
from transformers import BertTokenizer, GPT2Tokenizer, DistilBertTokenizer
from typing import Any


def get_tokenizer(llm_name: str = "DistilBert") -> Any:
    """获取大模型自然语言处理的Tokenizer配置"""
    if llm_name == "DistilBert":
        try:
            # 先尝试从本地加载
            tokenizer = DistilBertTokenizer.from_pretrained(
                "distilbert-base-uncased", trust_remote_code=True, local_files_only=True
            )
        except EnvironmentError:
            # 如果本地不存在则尝试从网络中请求下载
            tokenizer = DistilBertTokenizer.from_pretrained(
                "distilbert-base-uncased",
                trust_remote_code=True,
                local_files_only=False,
            )
    elif llm_name == "Bert":
        try:
            # 先尝试从本地加载
            tokenizer = BertTokenizer.from_pretrained(
                "google-bert/bert-base-uncased",
                trust_remote_code=True,
                local_files_only=True,
            )
        except EnvironmentError:
            # 如果本地不存在则尝试从网络中请求下载
            tokenizer = BertTokenizer.from_pretrained(
                "google-bert/bert-base-uncased",
                trust_remote_code=True,
                local_files_only=False,
            )
    elif llm_name == "GPT2":
        try:
            # 先尝试从本地加载
            tokenizer = GPT2Tokenizer.from_pretrained(
                "openai-community/gpt2", trust_remote_code=True, local_files_only=True
            )
        except EnvironmentError:
            # 如果本地不存在则尝试从网络中请求下载
            tokenizer = GPT2Tokenizer.from_pretrained(
                "openai-community/gpt2", trust_remote_code=True, local_files_only=False
            )
    else:
        # 使用大模型的名称输入错误
        raise ValueError
    return tokenizer
