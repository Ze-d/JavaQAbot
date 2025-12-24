"""
Java文档问答系统启动脚本
作者：zjy
创建时间：2024年

该脚本是Java文档问答系统的主启动入口，用于启动Gradio Web应用。
"""

import sys
import os

# 获取项目根目录（src/main的上级目录的上级目录）
# 当前文件: src/main/main.py
# 项目根目录应该是: C:\02-study\RAG\JavaQAbot
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.main.app import demo
from src.utils.logger_config import logger_app

if __name__ == '__main__':
    """
    应用程序入口点
    启动Gradio Web服务，提供Java文档问答助手功能
    """
    logger_app.info("启动Java文档问答助手应用")
    logger_app.debug("Gradio界面配置: max_width=850px")
    logger_app.debug(f"项目根目录: {project_root}")

    try:
        # 启动Web应用，启用分享功能
        demo.launch(share=True)
        logger_app.info("应用启动成功")
    except Exception as e:
        logger_app.error(f"应用启动失败: {e}")
