"""
工具函数模块
作者：zjy
创建时间：2024年

该模块提供Java文档问答系统的核心工具函数，包括：
1. 嵌入模型初始化和配置
2. LLM模型获取和配置（含Brotli错误修复）
3. 结构化输出解析
4. 字符串模板替换
5. Neo4j图数据库连接

所有工具函数都集成了日志记录功能，便于调试和监控。
"""

import os
import torch
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings as CommunityHuggingFaceEmbeddings
from py2neo import Graph
from logger_config import logger_utils

# 加载环境变量
load_dotenv()


def get_embeddings_model() -> HuggingFaceEmbeddings:
    """
    初始化嵌入模型

    根据GPU可用性自动选择设备，并配置批处理大小等参数。
    使用BGE基础模型进行文本嵌入。

    Returns:
        HuggingFaceEmbeddings: 初始化好的嵌入模型

    Raises:
        Exception: 当模型初始化失败时抛出异常
    """
    logger_utils.debug("开始初始化嵌入模型")

    # 1. 检查GPU是否可用
    is_cuda_available = torch.cuda.is_available()
    device = "cuda" if is_cuda_available else "cpu"
    logger_utils.info(f"设备选择: {device}")

    # 2. 模型配置参数
    model_kwargs = {
        "device": device,
        "trust_remote_code": True
    }
    logger_utils.debug(f"模型参数: {model_kwargs}")

    # 3. 编码配置参数
    encode_kwargs = {
        "normalize_embeddings": True,  # 归一化嵌入，保证检索精度
        "batch_size": 600 if is_cuda_available else 200,  # GPU: 600, CPU: 200
        "device": device
    }
    logger_utils.debug(f"编码参数: {encode_kwargs}")

    # 4. 模型路径
    model_path = 'C:/02-study/model/embeding/bge-base'
    logger_utils.info(f"模型路径: {model_path}")

    try:
        emb_model = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        logger_utils.info("嵌入模型初始化成功")
        return emb_model
    except Exception as e:
        logger_utils.error(f"嵌入模型初始化失败: {e}")
        raise


def get_llm_model() -> ChatOpenAI:
    """
    获取LLM模型实例

    使用DeepSeek Chat模型作为LLM后端，配置温度和最大令牌数等参数。
    集成了禁用响应压缩的HTTP适配器，解决Brotli解码错误。

    Returns:
        ChatOpenAI: 配置好的LLM模型实例

    Raises:
        Exception: 当模型获取失败时抛出异常
    """
    logger_utils.debug("开始获取LLM模型")

    try:
        # 创建禁用压缩的 httpx 客户端（LangChain兼容）
        import httpx

        # 创建禁用压缩的传输器
        class NoCompressionTransport(httpx.HTTPTransport):
            """自定义HTTP传输器，禁用响应压缩"""

            def handle_request(self, request):
                # 设置请求头禁用压缩
                request.headers.update({
                    'Accept-Encoding': 'identity',
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                return super().handle_request(request)

        # 创建 httpx 客户端
        http_client = httpx.Client(
            transport=NoCompressionTransport(),
            timeout=30.0,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )

        # 创建ChatOpenAI实例
        llm = ChatOpenAI(
            model='deepseek-chat',
            openai_api_base='https://api.deepseek.com/v1',
            openai_api_key='sk-ec1c58c12e9a48c39be6b3e7e31d1d48',
            temperature=0.01,
            max_tokens=2048,
            http_client=http_client  # 使用自定义httpx客户端
        )

        logger_utils.info("LLM模型获取成功")
        logger_utils.debug(f"模型参数: temperature=0.01, max_tokens=2048")
        logger_utils.info("已启用Brotli错误修复方案（禁用响应压缩）")
        return llm
    except Exception as e:
        logger_utils.error(f"LLM模型获取失败: {e}")

        # Brotli错误诊断和解决建议
        error_msg = str(e).lower()
        if "brotli" in error_msg or "content-encoding" in error_msg:
            logger_utils.error("检测到 Brotli 压缩解码错误！")
            logger_utils.error("解决方案:")
            logger_utils.error("1. 更新 brotli 包: pip install --upgrade brotli")
            logger_utils.error("2. 禁用响应压缩（已应用此方案）")
            logger_utils.error("3. 检查网络连接和代理设置")
            logger_utils.error("4. 尝试重启服务")

        raise


def structured_output_parser(response_schemas) -> str:
    """
    生成结构化输出提示词

    根据响应模式生成用于实体抽取的提示词模板。

    Args:
        response_schemas: 响应模式列表

    Returns:
        str: 生成的提示词文本
    """
    logger_utils.debug("开始生成结构化输出提示词")
    logger_utils.debug(f"响应模式数量: {len(response_schemas)}")

    text = '''
    请从以下文本中，抽取出实体信息，并按json格式输出，json包含首尾的 "```json" 和 "```"。
    以下是字段含义和类型，要求输出json中，必须包含下列所有字段：\n
    '''
    for schema in response_schemas:
        text += schema.name + ' 字段，表示：' + schema.description + '，类型为：' + schema.type + '\n'

    logger_utils.debug(f"提示词长度: {len(text)} 字符")
    return text


def replace_token_in_string(string: str, slots: list) -> str:
    """
    字符串模板替换

    将字符串中的槽位占位符替换为实际值。

    Args:
        string (str): 原始字符串，包含%slot%格式的占位符
        slots (list): 槽位列表，格式为[[key, value], ...]

    Returns:
        str: 替换后的字符串
    """
    logger_utils.debug(f"原始字符串长度: {len(string)}")
    logger_utils.debug(f"替换槽位数量: {len(slots)}")

    for key, value in slots:
        string = string.replace('%' + key + '%', value)

    logger_utils.debug(f"替换后字符串长度: {len(string)}")
    return string


def get_neo4j_conn() -> Graph:
    """
    连接Neo4j图数据库

    使用默认配置连接到本地Neo4j数据库实例。

    Returns:
        Graph: Neo4j图数据库连接对象

    Raises:
        Exception: 当数据库连接失败时抛出异常
    """
    logger_utils.debug("开始连接Neo4j数据库")

    try:
        uri = 'neo4j://127.0.0.1:7687'
        username = 'neo4j'
        password = '123456789'

        logger_utils.debug(f"连接参数: uri={uri}, username={username}")

        conn = Graph(uri, auth=(username, password))
        logger_utils.info("Neo4j连接成功")
        return conn
    except Exception as e:
        logger_utils.error(f"Neo4j连接失败: {e}")
        raise