"""
Java文档问答助手 Gradio 应用入口
作者：zjy
创建时间：2024年

该模块提供基于Gradio的Web界面，用于Java文档问答助手的用户交互。
支持实时对话、问题历史记录，并集成了日志记录系统。
"""

import gradio as gr
from typing import List
from service import Service
from logger_config import logger_app


def java_doc_bot(message: str, history: List[List[str]]) -> str:
    """
    Java文档问答助手核心处理函数

    Args:
        message (str): 用户输入的问题文本
        history (List[List[str]]): 对话历史记录，格式为 [[用户问题, AI回答], ...]

    Returns:
        str: AI生成的回答内容

    Raises:
        Exception: 当问题处理失败时抛出异常
    """
    logger_app.debug(f"收到用户消息: {message}")
    logger_app.debug(f"对话历史记录数: {len(history)} 条")

    try:
        # 初始化服务层
        logger_app.debug("创建Service实例")
        service = Service()

        # 处理用户问题
        logger_app.info("开始处理用户问题")
        result = service.answer(message, history)

        logger_app.info("问题处理完成")
        logger_app.debug(f"生成回答长度: {len(result)} 字符")
        return result
    except Exception as e:
        logger_app.error(f"问题处理失败: {e}")
        return "抱歉，系统暂时无法回答您的问题，请稍后重试。"

# Gradio界面样式配置
# 设置容器最大宽度为850px，居中显示
css = '''
.gradio-container {
    max-width: 850px !important;
    margin: 20px auto !important;
}
.message {
    padding: 10px !important;
    font-size: 14px !important;
}
'''

# 创建Gradio聊天界面
demo = gr.ChatInterface(
    css=css,
    fn=java_doc_bot,
    title='Java文档问答助手',
    chatbot=gr.Chatbot(height=400, bubble_full_width=False),
    theme=gr.themes.Default(spacing_size='sm', radius_size='sm'),
    textbox=gr.Textbox(
        placeholder="在此输入您的问题",
        container=False,
        scale=7
    ),
    examples=[
        "Java 8有哪些新特性？",
        "Spring Boot如何快速搭建一个Web应用？",
        "MyBatis和Hibernate的区别是什么？",
        "如何处理Java中的NullPointerException？",
        "Java并发编程的最佳实践有哪些？"
    ],
    submit_btn=gr.Button('提交', variant='primary'),
    # clear_btn=gr.Button('清空记录'),
    # retry_btn=None,
    # undo_btn=None,
)


if __name__ == '__main__':
    """
    应用程序入口点
    启动Gradio Web服务，提供Java文档问答助手功能
    """
    logger_app.info("启动Java文档问答助手应用")
    logger_app.debug("Gradio界面配置: max_width=850px")

    try:
        # 启动Web应用，启用分享功能
        demo.launch(share=True)
        logger_app.info("应用启动成功")
    except Exception as e:
        logger_app.error(f"应用启动失败: {e}")