"""
Java文档问答服务层
作者：zjy
创建时间：2024年

该模块提供Java文档问答的核心业务逻辑服务，包括对话摘要生成和问题处理。
作为应用层和智能体之间的中间层，负责协调各个功能模块。
"""

from typing import List
from src.prompts.prompt import *
from src.utils.utils import *
from src.core.agent import *
from langchain_core.prompts import PromptTemplate
from src.utils.logger_config import logger_service


class Service:
    """
    Java文档问答服务类

    负责协调Agent进行问题处理，并在需要时生成对话摘要。
    提供统一的接口给上层应用调用。
    """

    def __init__(self):
        """
        初始化Java服务层

        创建Agent实例，建立服务层的核心组件。
        """
        logger_service.debug("开始初始化Java服务层")

        # 创建Agent实例，用于处理具体问题
        logger_service.debug("创建Agent实例")
        self.agent = Agent()
        logger_service.info("Java服务层初始化完成")

    def get_summary_message(self, message: str, history: List[List[str]]) -> str:
        """
        生成对话摘要

        对历史对话和当前消息进行摘要，提取关键信息形成完整的问题描述。
        主要用于处理多轮对话中的指代消解问题。

        Args:
            message (str): 用户当前输入的消息
            history (List[List[str]]): 历史对话记录，格式为 [[问题, 回答], ...]

        Returns:
            str: 摘要后的问题文本

        Raises:
            Exception: 当摘要生成失败时抛出异常
        """
        logger_service.debug("开始生成对话摘要")
        logger_service.debug(f"当前消息: {message}")
        logger_service.debug(f"历史记录数: {len(history)} 条")

        try:
            # 加载LLM模型
            logger_service.debug("加载LLM模型")
            llm = get_llm_model()

            # 加载摘要提示词模板
            logger_service.debug("加载摘要提示词模板")
            prompt = PromptTemplate.from_template(SUMMARY_PROMPT_TPL)
            llm_chain = prompt | llm

            # 格式化历史对话（仅保留最近2轮）
            chat_history = ''
            for q, a in history[-2:]:
                chat_history += f'问题:{q}, 答案:{a}\n'

            logger_service.debug(f"压缩后历史对话: {chat_history[:100]}...")

            # 生成摘要
            response = llm_chain.invoke({
                'query': message,
                'chat_history': chat_history
            })
            logger_service.info("对话摘要生成完成")

            # 提取响应内容
            result = getattr(response, "content", response)
            logger_service.debug(f"摘要长度: {len(result)} 字符")
            return result
        except Exception as e:
            logger_service.error(f"对话摘要生成失败: {e}")
            raise

    def answer(self, message: str, history: List[List[str]]) -> str:
        """
        处理用户问题

        根据历史对话和当前问题，生成合适的回答。
        如果存在历史对话，会先进行摘要处理，然后委托Agent进行具体回答。

        Args:
            message (str): 用户输入的问题
            history (List[List[str]]): 历史对话记录

        Returns:
            str: 生成的回答文本

        Raises:
            Exception: 当问题处理失败时抛出异常
        """
        logger_service.info(f"开始处理用户问题: {message}")
        logger_service.debug(f"历史对话记录数: {len(history)} 条")

        try:
            # 如果存在历史对话，需要进行摘要处理
            if history:
                logger_service.debug("检测到历史对话，进行摘要处理")
                message = self.get_summary_message(message, history)
                logger_service.debug(f"摘要后的问题: {message}")

            # 委托Agent处理问题
            logger_service.debug("委托Agent进行问题回答")
            result = self.agent.query(message)
            logger_service.info("问题处理完成")
            return result
        except Exception as e:
            logger_service.error(f"问题处理失败: {e}")
            raise


if __name__ == '__main__':
    """
    服务层测试入口
    用于验证Java服务层的核心功能
    """
    service = Service()

    # 测试1：简单问候
    # print(service.answer('你好', []))

    # 测试2：单轮对话
    # print(service.answer('Spring Boot是什么？', [
    #     ['你好', '你好，有什么可以帮到您的吗？']
    # ]))

    # 测试3：多轮对话（验证摘要功能）
    print(service.answer('如何使用？', [
        ['你好', '你好，有什么可以帮到您的吗？'],
        ['Spring Boot是什么？', 'Spring Boot是一个基于Spring框架的快速开发框架，简化了Spring应用的搭建和开发过程。'],
    ]))
