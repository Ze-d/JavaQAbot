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


def get_embeddings_model():
    logger_utils.debug("get_embeddings_model() - 开始初始化嵌入模型")

    # 1. 检查 GPU 是否可用
    is_cuda_available = torch.cuda.is_available()
    device = "cuda" if is_cuda_available else "cpu"
    logger_utils.info(f"get_embeddings_model() - 设备选择: {device}")

    # 2. 模型配置（移除 dtype，仅保留 device 和 trust_remote_code）
    model_kwargs = {
        "device": device,  # 指定运行设备（GPU/CPU）
        "trust_remote_code": True
    }
    logger_utils.debug(f"get_embeddings_model() - 模型参数: {model_kwargs}")

    # 3. 编码配置（批量大小根据设备调整）
    encode_kwargs = {
        "normalize_embeddings": True,  # 必须开启，保证检索精度
        "batch_size": 600 if is_cuda_available else 200,  # 8GB GPU 推荐 600
        "device": device  # 关键：在编码时指定设备（解决 dtype 问题的核心）
    }
    logger_utils.debug(f"get_embeddings_model() - 编码参数: {encode_kwargs}")

    # 4. 加载模型（通过 HuggingFaceEmbeddings 封装）
    # model_path = os.getenv('EMBEDDING_MODEL_PATH')
    model_path = 'C:/02-study/model/embeding/bge-base'
    logger_utils.info(f"get_embeddings_model() - 模型路径: {model_path}")

    try:
        emb_model = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        logger_utils.info("get_embeddings_model() - 嵌入模型初始化成功")
        return emb_model
    except Exception as e:
        logger_utils.error(f"get_embeddings_model() - 嵌入模型初始化失败: {e}")
        raise
# todo
def get_llm_model():
    logger_utils.debug("get_llm_model() - 开始获取LLM模型")

    model_map = {
        'openai': ChatOpenAI(
            # model = os.getenv('OPENAI_LLM_MODEL','deepseek-chat'),
            model = 'deepseek-chat',
            openai_api_base='https://api.deepseek.com/v1',
            openai_api_key='sk-ec1c58c12e9a48c39be6b3e7e31d1d48',
            # temperature = os.getenv('TEMPERATURE'),
            temperature = 0.01,
            # max_tokens = os .getenv('MAX_TOKEND')
            max_tokens = 2048,
            # 添加超时设置（可选）
            request_timeout=30
        )
    }
    # return  model_map.get(os.getenv('LLM_MODEL'))

    try:
        llm = model_map.get('openai')
        logger_utils.info("get_llm_model() - LLM模型获取成功")
        logger_utils.debug(f"get_llm_model() - 模型参数: temperature=0.01, max_tokens=2048")
        return llm
    except Exception as e:
        logger_utils.error(f"get_llm_model() - LLM模型获取失败: {e}")

        # 如果出现 Brotli 错误，给出详细提示
        error_msg = str(e).lower()
        if "brotli" in error_msg or "content-encoding" in error_msg:
            logger_utils.error("检测到 Brotli 压缩解码错误！")
            logger_utils.error("解决方案:")
            logger_utils.error("1. 更新 brotli 包: pip install --upgrade brotli")
            logger_utils.error("2. 禁用响应压缩: 在环境中设置 DISABLE_COMPRESSION=true")
            logger_utils.error("3. 检查网络连接和代理设置")
            logger_utils.error("4. 尝试重启服务")

        raise

def structured_output_parser(response_schemas):
    logger_utils.debug("structured_output_parser() - 开始生成结构化输出提示词")
    logger_utils.debug(f"structured_output_parser() - 响应模式数量: {len(response_schemas)}")

    text = '''
    请从以下文本中，抽取出实体信息，并按json格式输出，json包含首尾的 "```json" 和 "```"。
    以下是字段含义和类型，要求输出json中，必须包含下列所有字段：\n
    '''
    for schema in response_schemas:
        text += schema.name + ' 字段，表示：' + schema.description + '，类型为：' + schema.type + '\n'

    logger_utils.debug(f"structured_output_parser() - 提示词长度: {len(text)} 字符")
    return text


def replace_token_in_string(string, slots):
    logger_utils.debug(f"replace_token_in_string() - 原始字符串长度: {len(string)}")
    logger_utils.debug(f"replace_token_in_string() - 替换槽位数量: {len(slots)}")

    for key, value in slots:
        string = string.replace('%'+key+'%', value)

    logger_utils.debug(f"replace_token_in_string() - 替换后字符串长度: {len(string)}")
    return string

def get_neo4j_conn():
    logger_utils.debug("get_neo4j_conn() - 开始连接Neo4j数据库")

    try:
        uri = 'neo4j://127.0.0.1:7687'
        username = 'neo4j'
        password = '123456789'

        logger_utils.debug(f"get_neo4j_conn() - 连接参数: uri={uri}, username={username}")

        conn = Graph(uri, auth=(username, password))
        logger_utils.info("get_neo4j_conn() - Neo4j连接成功")
        return conn
    except Exception as e:
        logger_utils.error(f"get_neo4j_conn() - Neo4j连接失败: {e}")
        raise