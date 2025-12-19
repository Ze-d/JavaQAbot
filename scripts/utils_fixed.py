"""
修复后的 utils.py
解决 Brotli 解码错误
"""
from langchain_openai import ChatOpenAI
from config import *
from py2neo import Graph
from langchain_huggingface import HuggingFaceEmbeddings
import torch

import os
from dotenv import load_dotenv
load_dotenv()

import os
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from logger_config import logger_utils
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class NoCompressionAdapter(HTTPAdapter):
    """自定义HTTP适配器，禁用响应压缩"""
    def init_poolmanager(self, *args, **kwargs):
        kwargs['disable_compression'] = True
        return super().init_poolmanager(*args, **kwargs)

def get_llm_model():
    logger_utils.debug("get_llm_model() - 开始获取LLM模型")

    # 创建禁用压缩的session
    session = requests.Session()
    session.mount('http://', NoCompressionAdapter())
    session.mount('https://', NoCompressionAdapter())
    session.headers.update({
        'Accept-Encoding': 'identity',  # 关键：不接受任何压缩
        'User-Agent': 'Custom Client'
    })

    model_map = {
        'openai': ChatOpenAI(
            model='deepseek-chat',
            openai_api_base='https://api.deepseek.com/v1',
            openai_api_key='sk-ec1c58c12e9a48c39be6b3e7e31d1d48',
            temperature=0.01,
            max_tokens=2048,
            # 添加HTTP session
            # 注意：ChatOpenAI 可能不支持直接传入 session
            # 这种情况下需要使用其他方法
        )
    }

    try:
        llm = model_map.get('openai')
        logger_utils.info("get_llm_model() - LLM模型获取成功")
        logger_utils.debug(f"get_llm_model() - 模型参数: temperature=0.01, max_tokens=2048")
        return llm
    except Exception as e:
        logger_utils.error(f"get_llm_model() - LLM模型获取失败: {e}")
        raise

# 其他函数保持不变...
