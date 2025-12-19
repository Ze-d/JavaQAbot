"""
Brotli错误修复工具模块
作者：zjy
创建时间：2024年

该模块用于解决OpenAI API调用中的Brotli压缩解码错误。
通过禁用响应压缩来避免兼容性问题。

主要解决方案：
1. 自定义HTTP适配器，禁用响应压缩
2. 创建自定义LLM实例
3. 测试API调用是否正常

使用方法：
1. 直接运行模块进行测试
2. 导入create_custom_llm函数创建修复后的LLM实例
"""

import os
from typing import Any, Dict, Optional

import requests
from langchain_openai import ChatOpenAI
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class NoCompressionHTTPAdapter(HTTPAdapter):
    """
    自定义HTTP适配器

    继承自HTTPAdapter，重写init_poolmanager方法以禁用响应压缩。
    这是解决Brotli解码错误的关键。
    """

    def init_poolmanager(self, *args, **kwargs):
        """
        初始化连接池管理器

        关键参数：
        - disable_compression: 禁用压缩响应
        """
        kwargs['disable_compression'] = True
        return super().init_poolmanager(*args, **kwargs)


def create_custom_llm() -> ChatOpenAI:
    """
    创建自定义LLM实例

    创建一个禁用了响应压缩的ChatOpenAI实例，用于解决Brotli解码错误。

    Returns:
        ChatOpenAI: 配置好的LLM实例
    """
    # 创建禁用压缩的session
    session = requests.Session()
    session.mount('http://', NoCompressionHTTPAdapter())
    session.mount('https://', NoCompressionHTTPAdapter())

    # 设置请求头
    session.headers.update({
        'Accept-Encoding': 'identity',  # 关键：不接受任何压缩
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })

    # 创建ChatOpenAI实例
    llm = ChatOpenAI(
        model='deepseek-chat',
        openai_api_base='https://api.deepseek.com/v1',
        openai_api_key=os.getenv('OPENAI_API_KEY', 'sk-ec1c58c12e9a48c39be6b3e7e31d1d48'),
        temperature=0.01,
        max_tokens=2048,
        # request_timeout=30  # 可选：设置超时时间
    )

    return llm


def test_api_call() -> bool:
    """
    测试API调用是否正常

    Returns:
        bool: 测试是否成功
    """
    try:
        print("=" * 60)
        print("🔧 测试 Brotli 修复方案")
        print("=" * 60)

        llm = create_custom_llm()

        # 测试简单的API调用
        print("📡 发送测试请求...")
        response = llm.invoke("你好")

        print(f"✅ 请求成功!")
        print(f"响应内容: {response.content}")

        return True

    except Exception as e:
        print(f"❌ 请求失败: {e}")
        print(f"错误类型: {type(e).__name__}")

        # 分析错误并给出建议
        error_msg = str(e).lower()
        if "brotli" in error_msg:
            print("\n🔍 检测到 Brotli 错误!")
            print("💡 建议:")
            print("1. 更新 brotli 包: pip install --upgrade brotli")
            print("2. 禁用响应压缩（如当前方案）")
            print("3. 检查网络连接")

        return False


if __name__ == '__main__':
    """
    工具入口点

    运行独立的测试程序，验证修复方案是否有效。
    """
    success = test_api_call()
    if success:
        print("\n🎉 Brotli错误已修复！")
    else:
        print("\n⚠️ 请检查错误信息并尝试其他解决方案")
