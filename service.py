from prompt import *
from utils import *
from agent import *

from langchain_core.prompts import PromptTemplate
from logger_config import logger_service


class Service():
    def __init__(self):
        logger_service.debug("Service.__init__() - 开始初始化Service")

        logger_service.debug("Service.__init__() - 创建Agent实例")
        self.agent = Agent()
        logger_service.info("Service.__init__() - Service初始化完成")

    def get_summary_message(self, message, history):
        logger_service.debug("get_summary_message() - 开始生成摘要")
        logger_service.debug(f"get_summary_message() - 当前消息: {message}")
        logger_service.debug(f"get_summary_message() - 历史记录: {len(history)} 条")

        try:
            logger_service.debug("get_summary_message() - 加载LLM模型")
            llm = get_llm_model()

            logger_service.debug("get_summary_message() - 加载摘要提示词")
            prompt = PromptTemplate.from_template(SUMMARY_PROMPT_TPL)
            llm_chain = prompt | llm

            chat_history = ''
            for q, a in history[-2:]:
                chat_history += f'问题:{q}, 答案:{a}\n'

            logger_service.debug(f"get_summary_message() - 压缩后历史: {chat_history[:100]}...")

            response = llm_chain.invoke({'query': message, 'chat_history': chat_history})
            logger_service.info("get_summary_message() - 摘要生成完成")

            result = getattr(response, "content", response)
            logger_service.debug(f"get_summary_message() - 摘要长度: {len(result)} 字符")
            return result
        except Exception as e:
            logger_service.error(f"get_summary_message() - 摘要生成失败: {e}")
            raise

    def answer(self, message, history):
        logger_service.info(f"Service.answer() - 开始处理问题: {message}")
        logger_service.debug(f"Service.answer() - 历史记录: {len(history)} 条")

        try:
            if history:
                logger_service.debug("Service.answer() - 需要摘要历史对话")
                message = self.get_summary_message(message, history)
                logger_service.debug(f"Service.answer() - 摘要后消息: {message}")

            logger_service.debug("Service.answer() - 委托Agent处理")
            result = self.agent.query(message)
            logger_service.info("Service.answer() - 问题处理完成")
            return result
        except Exception as e:
            logger_service.error(f"Service.answer() - 问题处理失败: {e}")
            raise


if __name__ == '__main__':
    service = Service()
    # print(service.answer('你好', []))
    # print(service.answer('得了鼻炎怎么办？', [
    #     ['你好', '你好，有什么可以帮到您的吗？']
    # ]))
    print(service.answer('大概多长时间能治好？', [
        ['你好', '你好，有什么可以帮到您的吗？'],
        ['得了鼻炎怎么办？', '以考虑使用丙酸氟替卡松鼻喷雾剂、头孢克洛颗粒等药物进行治疗。'],
    ]))
