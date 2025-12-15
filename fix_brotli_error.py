"""
修复 Brotli 解码错误
通过禁用响应压缩解决问题
"""
import os
import logging
from typing import Any, Dict, Optional
from langchain_openai import ChatOpenAI
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class NoCompressionHTTPAdapter(HTTPAdapter):
    """自定义HTTP适配器，禁用响应压缩"""
    def init_poolmanager(self, *args, **kwargs):
        kwargs['disable_compression'] = True  # 关键：禁用压缩
        return super().init_poolmanager(*args, **kwargs)

def create_custom_llm():
    """创建自定义LLM，禁用压缩"""
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
        # 禁用流式响应（可选）
        # request_timeout=30
    )

    return llm

def test_api_call():
    """测试API调用是否正常"""
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

        # 分析错误
        error_msg = str(e).lower()
        if "brotli" in error_msg:
            print("\n🔍 检测到 Brotli 错误!")
            print("💡 建议:")
            print("1. 更新 brotli 包: pip install --upgrade brotli")
            print("2. 禁用响应压缩（如当前方案）")
            print("3. 检查网络连接")

        return False

if __name__ == '__main__':
    test_api_call()
