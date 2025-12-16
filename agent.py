"""
医疗问诊智能体
作者：zjy
创建时间：2024年

该模块实现医疗问诊的核心智能体，负责协调多个工具（通用对话、向量检索、图数据库查询、网络搜索）
来回答用户的医疗相关问题。支持多轮对话、证据追踪和上下文理解。
"""

import os
import json
import urllib.parse
import requests
from typing import List, Dict, Any

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field

from utils import *
from config import *
from prompt import *
from logger_config import logger_agent


class Agent:
    """
    医疗问诊智能体

    集成多种工具和方法，包括：
    - 通用对话工具：处理日常问候和身份相关问题
    - 向量检索工具：基于知识库的语义检索
    - 图谱工具：基于Neo4j图数据库的医疗知识查询
    - 搜索工具：基于网络的实时搜索

    具备对话历史管理和证据追踪能力。
    """

    def __init__(self):
        """
        初始化医疗智能体

        完成以下初始化任务：
        1. 加载LLM模型
        2. 初始化向量数据库
        3. 构建工具列表
        4. 创建Agent实例
        """
        logger_agent.debug("开始初始化医疗智能体")

        # 1. 初始化LLM模型
        logger_agent.debug("初始化LLM模型")
        self._llm = get_llm_model()
        logger_agent.info("LLM模型加载完成")

        # 2. 加载向量数据库
        logger_agent.debug("加载向量数据库")
        db_path = os.path.join(os.path.dirname(__file__), './data/db')
        self.vdb = Chroma(
            persist_directory=db_path,
            embedding_function=get_embeddings_model()
        )
        logger_agent.info(f"向量数据库加载完成: {db_path}")

        # 3. 初始化证据追踪列表
        self.current_retrieved_contexts = []
        logger_agent.debug("证据追踪列表初始化完成")

        # 4. 构建工具列表
        logger_agent.debug("构建工具列表")
        self.tools = self._build_tools()
        logger_agent.info(f"工具列表构建完成: {len(self.tools)}个工具")

        # 5. 准备系统提示词
        system_prompt = SYSTEM_PROMPT_TPL
        logger_agent.debug(f"系统提示词长度: {len(system_prompt)}字符")

        # 6. 创建Agent实例
        logger_agent.debug("创建Agent实例")
        self.chat_history = []
        self._agent = create_agent(
            model=self._llm,
            tools=self.tools,
            middleware=[
                SummarizationMiddleware(
                    model=self._llm,
                    max_tokens_before_summary=600
                )
            ],
            system_prompt=system_prompt,
        )
        logger_agent.info("医疗智能体初始化完成")


    @staticmethod
    def _extract_content(response: Any) -> str:
        """
        从响应对象中提取文本内容

        统一不同类型响应的内容提取方式，支持多种响应格式。

        Args:
            response: 可能是包含content属性的对象、字典或字符串

        Returns:
            str: 提取的文本内容
        """
        # 如果响应对象有content属性，直接返回
        if hasattr(response, "content"):
            return response.content

        # 如果是字典，尝试从常见键中提取内容
        if isinstance(response, dict):
            for key in ("output", "output_text", "text"):
                if key in response:
                    return response[key]

        # 兜底：转换为字符串
        return str(response)

    def _build_tools(self) -> List:
        """
        构建Agent可用的工具列表

        包含四个核心工具：
        1. generic_tool: 通用对话工具
        2. retrival_tool: 向量检索工具
        3. graph_tool: 图数据库查询工具
        4. search_tool: 网络搜索工具

        Returns:
            List: 工具函数列表
        """
        logger_agent.debug("开始构建工具列表")

        agent_self = self

        # 工具1：通用对话工具
        @tool("generic_tool")
        def generic_tool(query: str) -> str:
            """
            通用对话工具

            处理日常问候、身份介绍等非医疗专业问题。
            确保机器人正确介绍自己的身份和限制。

            Args:
                query (str): 用户查询

            Returns:
                str: 回答内容
            """
            logger_agent.info(f"通用工具调用: {query[:50]}...")
            return agent_self.generic_func(query)

        # 工具2：向量检索工具
        @tool("retrival_tool")
        def retrival_tool(query: str) -> str:
            """
            向量检索工具

            基于向量数据库的语义检索，主要用于药品说明书等结构化文档查询。
            支持相似度搜索和相关性评分。

            Args:
                query (str): 检索查询

            Returns:
                str: 检索结果和生成的回答
            """
            logger_agent.info(f"检索工具调用: {query[:50]}...")
            return agent_self.retrival_func(query)

        # 工具3：图数据库查询工具
        @tool("graph_tool")
        def graph_tool(query: str) -> str:
            """
            图数据库查询工具

            基于Neo4j图数据库的医疗知识查询，包括疾病、症状、药物等实体关系。
            支持实体提取、模板匹配和Cypher查询。

            Args:
                query (str): 查询问题

            Returns:
                str: 基于图数据库的回答
            """
            logger_agent.info(f"图谱工具调用: {query[:50]}...")
            return agent_self.graph_func(query)

        # 工具4：网络搜索工具
        @tool("search_tool")
        def search_tool(query: str) -> str:
            """
            网络搜索工具

            当其他工具无法提供答案时，使用网络搜索补充常识性回答。
            支持搜索结果摘要和相关性过滤。

            Args:
                query (str): 搜索查询

            Returns:
                str: 搜索结果和生成的回答
            """
            logger_agent.info(f"搜索工具调用: {query[:50]}...")
            return agent_self.search_func(query)

        # 组合所有工具
        tools = [generic_tool, retrival_tool, graph_tool, search_tool]
        logger_agent.info(f"工具构建完成: {[t.name for t in tools]}")
        return tools
    def generic_func(self, query: str) -> str:
        """
        处理通用对话问题

        使用通用提示词模板处理非医疗专业问题，如问候、自我介绍等。
        确保机器人身份一致性和安全边界。

        Args:
            query (str): 用户查询

        Returns:
            str: 生成的回答

        Raises:
            Exception: 当处理失败时抛出异常
        """
        logger_agent.debug(f"处理通用问题: {query}")

        try:
            # 加载通用提示词模板
            prompt = PromptTemplate.from_template(GENERIC_PROMPT_TPL)
            logger_agent.debug("通用提示词模板加载完成")

            # 构建LLM链
            llm_chain = prompt | self._llm
            logger_agent.debug("LLM链创建完成")

            # 生成回答
            response = llm_chain.invoke({'query': query})
            logger_agent.info("通用回答生成成功")

            # 提取内容
            result = self._extract_content(response)
            logger_agent.debug(f"答案长度: {len(result)}字符")
            return result
        except Exception as e:
            logger_agent.error(f"通用对话处理失败: {e}")
            raise

    def retrival_func(self, query: str) -> str:
        """
        向量检索功能

        基于向量数据库进行语义检索，主要用于查询药品说明书等文档内容。
        包含相似度搜索、相关性过滤、证据保存和答案生成四个步骤。

        Args:
            query (str): 检索查询

        Returns:
            str: 基于检索结果的生成回答

        Raises:
            Exception: 当检索失败时抛出异常
        """
        logger_agent.info(f"开始向量检索: {query}")

        try:
            # 1. 执行向量相似度搜索
            logger_agent.debug("执行相似度搜索，k=10")
            documents = self.vdb.similarity_search_with_relevance_scores(query, k=10)
            logger_agent.debug(f"检索到 {len(documents)} 个文档")

            # 2. 过滤低相关性结果
            threshold = 0.2
            query_result = [doc[0].page_content for doc in documents if doc[1] > threshold]
            logger_agent.debug(f"相关性阈值 {threshold}，过滤后剩余 {len(query_result)} 个文档")

            # 3. 保存检索证据
            if query_result:
                self.current_retrieved_contexts.extend(query_result)
                logger_agent.debug(f"已保存 {len(query_result)} 条证据到追踪列表")

            # 4. 加载检索提示词模板
            prompt = PromptTemplate.from_template(RETRIVAL_PROMPT_TPL)
            logger_agent.debug("检索提示词模板加载完成")

            # 5. 构建检索链
            retrival_chain = prompt | self._llm
            inputs = {
                'query': query,
                'query_result': '\n\n'.join(query_result) if query_result else '没有查到'
            }
            logger_agent.debug("输入参数准备完成")

            # 6. 生成回答
            response = retrival_chain.invoke(inputs)
            logger_agent.info("检索回答生成成功")

            # 7. 提取内容并返回
            result = self._extract_content(response)
            logger_agent.debug(f"答案长度: {len(result)}字符")
            return result
        except Exception as e:
            logger_agent.error(f"向量检索失败: {e}")
            raise

    def graph_func(self, query: str) -> str:
        """
        图数据库查询功能

        基于Neo4j图数据库进行医疗知识查询，包含以下步骤：
        1. 实体提取（NER）：从用户问题中提取疾病、症状、药物等实体
        2. 模板匹配：根据提取的实体匹配对应的查询模板
        3. 相似度筛选：对匹配的模板进行语义相似度筛选
        4. 图查询：执行Cypher语句查询图数据库
        5. 答案生成：基于查询结果生成最终回答

        Args:
            query (str): 用户查询问题

        Returns:
            str: 基于图数据库查询的答案

        Note:
            性能优化TODO:
            - 需要审计当前瓶颈并设计模板/FAISS缓存方案
            - 实现缓存的FAISS索引和模板重用，避免每请求重建
            - 添加回归测试或轻量级基准测试
        """
        logger_agent.debug(f"开始图数据库查询: {query}")

        # 步骤1：定义医疗实体提取模型
        class Medical(BaseModel):
            """医疗实体提取模型"""
            disease: List[str] = Field(default=[], description="疾病名称实体")
            symptom: List[str] = Field(default=[], description="疾病症状实体")
            drug: List[str] = Field(default=[], description="药物名称实体")

        # 步骤2：配置结构化输出策略
        response_schemas = ToolStrategy(Medical)
        format_instructions = response_schemas

        # 步骤3：初始化输出解析器
        output_parser = StrOutputParser(response_schemas=response_schemas)

        # 步骤4：构建NER提示词
        ner_prompt = PromptTemplate(
            template=NER_PROMPT_TPL,
            partial_variables={'format_instructions': format_instructions},
            input_variables=['query']
        )

        # 步骤5：执行实体提取
        ner_chain = ner_prompt | self._llm
        ner_response = ner_chain.invoke({'query': query})
        parsed_str = output_parser.parse(self._extract_content(ner_response))
        ner_result = json.loads(parsed_str)
        logger_agent.debug(f"实体提取结果: {ner_result}")

        # 步骤6：模板匹配和填充
        graph_templates = []
        for template in GRAPH_TEMPLATE.values():
            slot = template['slots'][0]
            slot_values = ner_result.get(slot, [])

            for value in slot_values:
                graph_templates.append({
                    'question': replace_token_in_string(template['question'], [[slot, value]]),
                    'cypher': replace_token_in_string(template['cypher'], [[slot, value]]),
                    'answer': replace_token_in_string(template['answer'], [[slot, value]]),
                })

        logger_agent.debug(f"匹配到 {len(graph_templates)} 个查询模板")

        # 如果没有匹配的模板，返回空结果
        if not graph_templates:
            logger_agent.debug("未找到匹配的查询模板")
            return '没有查到'

        # 步骤7：将模板转换为文档并建立FAISS索引
        graph_documents = [
            Document(page_content=template['question'], metadata=template)
            for template in graph_templates
        ]
        db = FAISS.from_documents(graph_documents, get_embeddings_model())
        graph_documents_filter = db.similarity_search_with_relevance_scores(query, k=3)
        logger_agent.debug(f"相似度筛选后剩余 {len(graph_documents_filter)} 个模板")

        # 步骤8：执行图数据库查询
        query_result = []
        neo4j_conn = get_neo4j_conn()
        for document, score in graph_documents_filter:
            # 相似度阈值过滤
            if score < 0.5:
                continue

            question = document.page_content
            cypher = document.metadata['cypher']
            answer = document.metadata['answer']

            try:
                # 执行Cypher查询
                result = neo4j_conn.run(cypher).data()

                # 检查查询结果是否有效
                if result and any(value for value in result[0].values()):
                    # 格式化答案
                    answer_str = replace_token_in_string(answer, list(result[0].items()))
                    formatted_text = f'问题{question}\n答案：{answer_str}'
                    query_result.append(formatted_text)

                    # 保存证据
                    self.current_retrieved_contexts.append(formatted_text)
            except Exception as e:
                logger_agent.warning(f"图查询执行失败: {e}")
                continue

        logger_agent.debug(f"成功查询到 {len(query_result)} 条结果")

        # 步骤9：生成最终回答
        prompt = PromptTemplate.from_template(GRAPH_PROMPT_TPL)
        graph_chain = prompt | self._llm
        inputs = {
            'query': query,
            'query_result': '\n\n'.join(query_result) if query_result else '没有查到'
        }
        response = graph_chain.invoke(inputs)

        # 返回提取的内容
        return self._extract_content(response)


    def search_func(self, query: str) -> str:
        """
        网络搜索功能

        当其他工具无法提供答案时，使用网络搜索补充常识性回答。
        包含搜索、结果处理和答案生成三个步骤。

        Args:
            query (str): 搜索查询

        Returns:
            str: 基于搜索结果的生成回答

        Note:
            使用360搜索API进行搜索
        """
        logger_agent.debug(f"开始网络搜索: {query}")

        # 步骤1：构建搜索提示词
        prompt = PromptTemplate.from_template(SEARCH_PROMPT_TPL)
        chain = prompt | self._llm

        # 步骤2：编码查询并构建搜索URL
        encoded_query = urllib.parse.quote(query)
        url = f"https://www.so.com/s?src=lm&ls=sm3020463&lm_extend=ctype:31&q={encoded_query}"
        logger_agent.debug(f"搜索URL: {url}")

        # 步骤3：执行网络请求
        response = requests.get(url)
        response.raise_for_status()
        search_result = response.text
        logger_agent.debug(f"原始搜索结果长度: {len(search_result)}字符")

        # 步骤4：如果结果过长，进行摘要处理
        if len(search_result) > 3000:
            logger_agent.debug("搜索结果过长，进行摘要处理")

            # 定义摘要提示词模板
            summary_prompt_template = """请将以下搜索结果总结为简洁的摘要（不超过300字），
            重点保留与查询 "{query}" 相关的信息：

            {search_result}
            """

            # 创建摘要链
            summary_prompt = PromptTemplate.from_template(summary_prompt_template)
            summary_chain = summary_prompt | self._llm

            # 准备输入数据（截断以避免过长）
            inputs = {
                "query": query,
                "search_result": search_result[:5000]
            }

            # 生成摘要
            summary_response = summary_chain.invoke(inputs)
            search_result = summary_response.content
            logger_agent.debug(f"摘要后长度: {len(search_result)}字符")

        # 步骤5：生成最终回答
        inputs = {
            'query': query,
            'query_result': search_result
        }

        response = chain.invoke(inputs)
        return self._extract_content(response)

    def query(self, query: str, return_context: bool = False):
        """
        处理用户查询

        Agent的核心查询入口，包含以下步骤：
        1. 重置证据追踪列表
        2. 管理对话历史
        3. 调用Agent生成回答
        4. 提取和返回答案

        Args:
            query (str): 用户查询问题
            return_context (bool): 是否返回检索的上下文证据

        Returns:
            str or tuple: 如果return_context为False，返回答案字符串；
                         如果为True，返回(答案, 上下文列表)元组

        Raises:
            Exception: 当查询处理失败时抛出异常
        """
        logger_agent.info(f"收到用户查询: {query}")
        logger_agent.debug(f"return_context参数: {return_context}")

        try:
            # 步骤1：重置证据追踪列表
            self.current_retrieved_contexts = []
            logger_agent.debug("证据追踪列表已重置")

            # 步骤2：添加用户消息到历史
            self.chat_history.append(HumanMessage(content=query))
            logger_agent.debug(f"用户消息已添加，当前历史长度: {len(self.chat_history)}")

            # 步骤3：控制对话历史长度（保留最近4条消息）
            if len(self.chat_history) > 4:
                self.chat_history = self.chat_history[-4:]
                logger_agent.debug("对话历史长度已截断至4条")

            # 步骤4：调用Agent生成回答（支持两种调用模式）
            logger_agent.debug("开始调用Agent")
            try:
                # 模式1：标准消息模式
                result = self._agent.invoke({"messages": self.chat_history})
                logger_agent.debug("Agent调用成功(模式1)")
            except Exception as e:
                logger_agent.warning(f"Agent调用失败，尝试备用模式: {e}")
                # 模式2：备用输入模式
                result = self._agent.invoke({"input": query, "chat-history": self.chat_history})
                logger_agent.debug("Agent调用成功(模式2)")

            # 步骤5：提取最终答案
            final_answer = ""
            extracted_from_messages = False

            logger_agent.debug("开始提取答案")
            if isinstance(result, dict):
                messages = result.get("messages")
                if messages:
                    logger_agent.debug(f"提取到 {len(messages)} 条消息")
                    last_msg = messages[-1]

                    # 从消息中提取内容
                    if isinstance(last_msg, dict):
                        final_answer = last_msg.get("content", "")
                        extracted_from_messages = True
                        logger_agent.debug("从字典消息中提取答案")
                    else:
                        final_answer = getattr(last_msg, "content", str(last_msg))
                        extracted_from_messages = True
                        logger_agent.debug("从对象消息中提取答案")

            # 兜底提取逻辑
            if not extracted_from_messages:
                logger_agent.debug("使用兜底提取逻辑")
                final_answer = self._extract_content(result)

            logger_agent.info(f"答案提取完成，长度: {len(final_answer)} 字符")

            # 步骤6：添加AI回答到历史
            self.chat_history.append(AIMessage(content=final_answer))
            logger_agent.debug(f"AI回答已添加，当前历史长度: {len(self.chat_history)}")

            # 步骤7：根据参数返回结果
            if return_context:
                contexts = self.current_retrieved_contexts if self.current_retrieved_contexts else ["未检索到相关文档"]
                logger_agent.info(f"返回答案和上下文，上下文数量: {len(contexts)}")
                return final_answer, contexts

            logger_agent.info("返回最终答案")
            return final_answer

        except Exception as e:
            logger_agent.error(f"查询处理失败: {e}")
            raise

if __name__ == '__main__':
    """
    智能体测试入口

    提供各种测试用例，用于验证Agent的各项功能。
    可以根据需要取消注释相应的测试代码。
    """
    agent = Agent()

    # ===== 测试1：通用对话功能 =====
    # print(agent.query('你好'))
    # print(agent.query('寻医问药网获得过哪些投资？'))

    # ===== 测试2：Agent.query()方法（集成测试） =====
    # print(agent.query('鼻炎和感冒是并发症吗？'))
    # print(agent.query('鼻炎怎么治疗？'))
    # print(agent.query('烧橙子可以治感冒吗？'))

    # ===== 测试3：通用工具单独测试 =====
    # print(agent.generic_func('你好'))
    # print(agent.generic_func('你叫什么名字？'))

    # ===== 测试4：向量检索工具单独测试 =====
    # print(agent.retrival_func('介绍一下奥沙利铂注射液'))
    # print(agent.retrival_func('寻医问药网的客服电话是多少？'))

    # ===== 测试5：图谱工具单独测试 =====
    # print(agent.graph_func('感冒和鼻炎是并发症吗？'))
    # print(agent.graph_func('感冒一般是由什么引起的？'))
    # print(agent.graph_func('感冒了现在咳嗽的厉害吃什么药好得快？可以吃阿莫西林吗？'))

    # ===== 测试6：搜索工具单独测试 =====
    # print(agent.search_func('微软是什么？'))
