from utils import *
from config import *
from prompt import *

import os
from typing import List
import json


from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.agents.middleware import SummarizationMiddleware

from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field
from langchain_community.vectorstores import  FAISS
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage
from langchain.tools import tool
from langchain_chroma import Chroma
import urllib.parse
import requests
from logger_config import logger_agent

class Agent():
    def __init__(self):
        logger_agent.debug("Agent.__init__() - 开始初始化Agent")

        logger_agent.debug("Agent.__init__() - 初始化LLM模型")
        self._llm = get_llm_model()
        logger_agent.info("Agent.__init__() - LLM模型加载完成")

        logger_agent.debug("Agent.__init__() - 加载向量数据库")
        self.vdb = Chroma(
            persist_directory=os.path.join(os.path.dirname(__file__), './data/db'),
            embedding_function=get_embeddings_model()
        )
        logger_agent.info("Agent.__init__() - 向量数据库加载完成: ./data/db")

        logger_agent.debug("Agent.__init__() - 初始化证据追踪列表")
        self.current_retrieved_contexts = []

        logger_agent.debug("Agent.__init__() - 构建工具列表")
        self.tools = self._build_tools()
        logger_agent.info(f"Agent.__init__() - 工具列表构建完成: {len(self.tools)}个工具")

        system_prompt = (SYSTEM_PROMPT_TPL)
        logger_agent.debug(f"Agent.__init__() - 系统提示长度: {len(system_prompt)}字符")

        logger_agent.debug("Agent.__init__() - 创建Agent实例")
        self.chat_history = []
        self._agent = create_agent(
            model=self._llm,
            tools=self.tools,
            middleware = [
                SummarizationMiddleware(
                    model=self._llm,
                    max_tokens_before_summary=600
                )
        ],
            system_prompt=system_prompt,


        )
        logger_agent.info("Agent.__init__() - Agent初始化完成")


    @staticmethod
    def _extract_content(response) -> str:
        if hasattr(response, "content"):
            return response.content
        if isinstance(response, dict):
            for key in ("output", "output_text", "text"):
                if key in response:
                    return response[key]
        return str(response)

    def _build_tools(self) -> List:
        logger_agent.debug("Agent._build_tools() - 开始构建工具列表")

        agent_self = self

        @tool("generic_tool")
        def generic_tool(query: str) -> str:
            """可以解答通用领域的知识，例如打招呼，问你是谁等问题"""
            logger_agent.info(f"generic_tool - 调用通用工具: {query[:50]}...")
            return agent_self.generic_func(query)

        @tool("retrival_tool")
        def retrival_tool(query: str) -> str:
            """用于回答药品说明相关问题"""
            logger_agent.info(f"retrival_tool - 调用检索工具: {query[:50]}...")
            return agent_self.retrival_func(query)

        @tool("graph_tool")
        def graph_tool(query: str) -> str:
            """用于回答疾病、症状等医疗相关问题"""
            logger_agent.info(f"graph_tool - 调用图谱工具: {query[:50]}...")
            return agent_self.graph_func(query)

        @tool("search_tool", description="当其他工具无法给出答案时，通过搜索补充常识性回答")
        def search_tool(query: str) -> str:
            logger_agent.info(f"search_tool - 调用搜索工具: {query[:50]}...")
            return agent_self.search_func(query)

        tools = [generic_tool, retrival_tool, graph_tool, search_tool]
        logger_agent.info(f"Agent._build_tools() - 工具构建完成: {[t.name for t in tools]}")
        return tools
        # return [retrival_tool]
    def generic_func(self, query: str) -> str:
        logger_agent.debug(f"generic_func() - 处理通用问题: {query}")

        try:
            prompt = PromptTemplate.from_template(GENERIC_PROMPT_TPL)
            logger_agent.debug("generic_func() - 通用提示词模板加载完成")

            llm_chain = prompt | self._llm
            logger_agent.debug("generic_func() - LLM链创建完成")

            response = llm_chain.invoke({'query': query})
            logger_agent.info("generic_func() - 通用回答生成成功")

            result = self._extract_content(response)
            logger_agent.debug(f"generic_func() - 答案长度: {len(result)}字符")
            return result
        except Exception as e:
            logger_agent.error(f"generic_func() - 通用对话失败: {e}")
            raise

    def retrival_func(self, query: str) -> str:
        logger_agent.info(f"retrival_func() - 开始向量检索: {query}")

        try:
            # 1. 向量相似度搜索
            logger_agent.debug("retrival_func() - 执行相似度搜索，k=10")
            documents = self.vdb.similarity_search_with_relevance_scores(query, k=10)
            logger_agent.debug(f"retrival_func() - 检索到 {len(documents)} 个文档")

            # 2. 过滤低分结果
            threshold = 0.2
            query_result = [doc[0].page_content for doc in documents if doc[1] > threshold]
            logger_agent.debug(f"retrival_func() - 相关性阈值 {threshold}，过滤后剩余 {len(query_result)} 个")

            # 3. 保存证据
            if query_result:
                self.current_retrieved_contexts.extend(query_result)
                logger_agent.debug(f"retrival_func() - 已保存 {len(query_result)} 条证据")

            # 4. 生成答案
            prompt = PromptTemplate.from_template(RETRIVAL_PROMPT_TPL)
            logger_agent.debug("retrival_func() - 检索提示词加载完成")

            retrival_chain = prompt | self._llm
            inputs = {
                'query': query,
                'query_result': '\n\n'.join(query_result) if query_result else '没有查到'
            }
            logger_agent.debug("retrival_func() - 输入参数准备完成")

            response = retrival_chain.invoke(inputs)
            logger_agent.info("retrival_func() - 检索回答生成成功")

            result = self._extract_content(response)
            logger_agent.debug(f"retrival_func() - 答案长度: {len(result)}字符")
            return result
        except Exception as e:
            logger_agent.error(f"retrival_func() - 向量检索失败: {e}")
            raise

    def graph_func(self, query: str) -> str:
        # TODO(graph_func-perf):
        # 1. Audit current bottlenecks and design caching plan for templates / FAISS.
        # 2. Implement cached FAISS index + template reuse to avoid per-request rebuild.
        # 3. Add regression tests or lightweight benchmarks after optimization.

        class Medical(BaseModel):
            disease: List[str] = Field(default=[], description="疾病名称实体")
            symptom: List[str] = Field(default=[], description='疾病症状实体')
            drug: List[str] = Field(default=[], description='药物名称实体')

        response_schemas = ToolStrategy(Medical)
        #LangChain v1 的 ToolStrategy + Pydantic 模型原生支持多实体提取，无需手动编写循环填充逻辑。只需定义好列表类型的字段，框架会自动处理多实体的识别和存储。
        format_instructions = response_schemas


        output_parser = StrOutputParser(response_schemas = response_schemas)

        ner_prompt = PromptTemplate(
            template = NER_PROMPT_TPL,
            partial_variables={'format_instructions': format_instructions},
            input_variables=['query']
        )
      #  ner_prompt = PromptTemplate.from_template(NER_PROMPT_TPL)
      #  result = ner_prompt.format_prompt(format_instructions = format_instructions , query = query)


        ner_chain = ner_prompt | self._llm
        ner_response = ner_chain.invoke({'query': query})
        parsed_str = output_parser.parse(self._extract_content(ner_response))
        ner_result = json.loads(parsed_str)
        #print(ner_result)
        graph_templates = []
        for template in GRAPH_TEMPLATE.values():

            slot = template['slots'][0]

            slot_values = ner_result.get(slot,[])

            for value in slot_values:
                graph_templates.append({
                    'question': replace_token_in_string(template['question'], [[slot, value]]),
                    'cypher': replace_token_in_string(template['cypher'], [[slot, value]]),
                    'answer': replace_token_in_string(template['answer'], [[slot, value]]),
                })
        if not graph_templates:
            return '没有查到'

        graph_documents = [
            Document(page_content=template['question'], metadata=template)
            for template in graph_templates
        ]
        db = FAISS.from_documents(graph_documents, get_embeddings_model())
        graph_documents_filter = db.similarity_search_with_relevance_scores(query, k=3)
        #print(graph_documents_filter)

        query_result = []
        neo4j_conn = get_neo4j_conn()
        for document,score in graph_documents_filter:
            if score < 0.5:
                continue
            question = document.page_content
            cypher = document.metadata['cypher']
            answer = document.metadata['answer']
            try:
                result = neo4j_conn.run(cypher).data()
                if result and any(value for value in result[0].values()):
                    answer_str = replace_token_in_string(answer,list(result[0].items()))
                    formatted_text = f'问题{question}\n答案：{answer_str}'
                    query_result.append(formatted_text)
                    self.current_retrieved_contexts.append(formatted_text)
            except Exception:
                continue

        prompt = PromptTemplate.from_template(GRAPH_PROMPT_TPL)
        graph_chain = prompt | self._llm
        inputs = {
            'query':query,
            'query_result':'\n\n'.join(query_result) if query_result else'没有查到'
        }
        response = graph_chain.invoke(inputs)
        #print(response.content)
        return self._extract_content(response)


    def search_func(self,query: str) -> str:
        prompt = PromptTemplate.from_template(SEARCH_PROMPT_TPL)
        chain = prompt | self._llm
        encoded_query = urllib.parse.quote(query)
        url = f"https://www.so.com/s?src=lm&ls=sm3020463&lm_extend=ctype:31&q={encoded_query}"
        response = requests.get(url)
        response.raise_for_status()
        search_result = response.text
        if len(search_result) > 3000:
            # 1. 定义一个带有占位符的模板字符串
            summary_prompt_template = """请将以下搜索结果总结为简洁的摘要（不超过300字），
            重点保留与查询 "{query}" 相关的信息：

            {search_result}
            """
            # 2. 使用模板创建 PromptTemplate 对象
            prompt = PromptTemplate.from_template(summary_prompt_template)

            # 3. 创建链
            summary_chain = prompt | self._llm

            # 4. 准备一个字典，包含要填充到模板中的数据
            # 在这里进行截断，确保传给大模型的内容不会超长
            inputs = {
                "query": query,
                "search_result": search_result[:5000]
            }

            # 5. 调用链，并传入这个字典
            summary_response = summary_chain.invoke(inputs)

            # 6. 获取最终的文本内容
            search_result = summary_response.content
        inputs = {
            'query': query,
            'query_result':search_result
        }


        response = chain.invoke(inputs)
        return self._extract_content(response)

    def query(self, query: str, return_context: bool = False):
        logger_agent.info(f"query() - 收到用户查询: {query}")
        logger_agent.debug(f"query() - return_context参数: {return_context}")

        try:
            # 1. 重置上下文
            self.current_retrieved_contexts = []
            logger_agent.debug("query() - 证据列表已重置")

            # 2. 添加用户消息
            self.chat_history.append(HumanMessage(content=query))
            logger_agent.debug(f"query() - 用户消息已添加，当前历史长度: {len(self.chat_history)}")

            # 3. 控制历史长度
            if len(self.chat_history) > 4:
                self.chat_history = self.chat_history[-4:]
                logger_agent.debug("query() - 历史长度已截断至4条")

            # 4. Agent调用
            logger_agent.debug("query() - 开始调用Agent")
            try:
                result = self._agent.invoke({"messages": self.chat_history})
                logger_agent.debug("query() - Agent调用成功(模式1)")
            except Exception as e:
                logger_agent.warning(f"query() - Agent调用失败，尝试备用模式: {e}")
                result = self._agent.invoke({"input": query, "chat-history": self.chat_history})
                logger_agent.debug("query() - Agent调用成功(模式2)")

            # 5. 提取答案
            final_answer = ""
            extracted_from_messages = False

            logger_agent.debug("query() - 开始提取答案")
            if isinstance(result, dict):
                messages = result.get("messages")
                if messages:
                    logger_agent.debug(f"query() - 提取到 {len(messages)} 条消息")
                    last_msg = messages[-1]
                    if isinstance(last_msg, dict):
                        final_answer = last_msg.get("content", "")
                        extracted_from_messages = True
                        logger_agent.debug("query() - 从字典消息中提取答案")
                    else:
                        final_answer = getattr(last_msg, "content", str(last_msg))
                        extracted_from_messages = True
                        logger_agent.debug("query() - 从对象消息中提取答案")

            if not extracted_from_messages:
                logger_agent.debug("query() - 使用兜底提取逻辑")
                final_answer = self._extract_content(result)

            logger_agent.info(f"query() - 答案提取完成，长度: {len(final_answer)} 字符")

            # 6. 添加AI回答
            self.chat_history.append(AIMessage(content=final_answer))
            logger_agent.debug(f"query() - AI回答已添加，当前历史长度: {len(self.chat_history)}")

            # 7. 返回结果
            if return_context:
                contexts = self.current_retrieved_contexts if self.current_retrieved_contexts else ["未检索到相关文档"]
                logger_agent.info(f"query() - 返回答案和上下文，上下文数量: {len(contexts)}")
                return final_answer, contexts

            logger_agent.info("query() - 返回最终答案")
            return final_answer

        except Exception as e:
            logger_agent.error(f"query() - 查询处理失败: {e}")
            raise

if __name__ == '__main__':
    agent = Agent()

    #print(agent.query('你好'))
    #print(agent.query('寻医问药网获得过哪些投资？'))
    # print(agent.query('鼻炎和感冒是并发症吗？'))
    #print(agent.query('鼻炎怎么治疗？'))
    # print(agent.query('烧橙子可以治感冒吗？'))


    #print(agent.generic_func('你好'))
    # print(agent.generic_func('你叫什么名字？'))

    #print(agent.retrival_func('介绍一下奥沙利铂注射液'))
    # print(agent.retrival_func('寻医问药网的客服电话是多少？'))

    # print(agent.graph_func('感冒和鼻炎是并发症吗？'))
    # print(agent.graph_func('感冒一般是由什么引起的？'))
    #print(agent.graph_func('感冒了现在咳嗽的厉害吃什么药好得快？可以吃阿莫西林吗？'))

    #print(agent.search_func('微软是什么？'))
