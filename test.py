from utils import *
from config import *
from prompt import *

import os
from typing import List, Tuple
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

class Agent():
    def __init__(self):
        self._llm = get_llm_model()
        self.vdb = Chroma(
            persist_directory=os.path.join(os.path.dirname(__file__), './data/db'),
            embedding_function=get_embeddings_model()
        )
        self.current_retrieved_contexts = []
        self.tools = self._build_tools()
        system_prompt = (SYSTEM_PROMPT_TPL)
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
        agent_self = self

        @tool("generic_tool")
        def generic_tool(query: str) -> str:
            """可以解答通用领域的知识，例如打招呼，问你是谁等问题"""
            return agent_self.generic_func(query)

        @tool("retrival_tool")
        def retrival_tool(query: str) -> str:
            """用于回答药品说明相关问题"""
            return agent_self.retrival_func(query)

        @tool("graph_tool")
        def graph_tool(query: str) -> str:
            """用于回答疾病、症状等医疗相关问题"""
            return agent_self.graph_func(query)

        @tool("search_tool", description="当其他工具无法给出答案时，通过搜索补充常识性回答")
        def search_tool(query: str) -> str:
            return agent_self.search_func(query)

        return [generic_tool, retrival_tool, graph_tool, search_tool]

    def generic_func(self, query: str) -> str:
        prompt = PromptTemplate.from_template(GENERIC_PROMPT_TPL)
        llm_chain = prompt | self._llm
        response = llm_chain.invoke({'query': query})
        return self._extract_content(response)

    def _normalize_scores(self, items: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        if not items:
            return []
        scores = [score for _, score in items]
        min_score = min(scores)
        max_score = max(scores)
        denom = max(max_score - min_score, 1e-8)
        normalized = [
            (text, (score - min_score) / denom)
            for text, score in items
        ]
        return normalized

    def _graph_context_search(self, query: str, track_context: bool = True) -> List[Tuple[str, float]]:

        class Medical(BaseModel):
            disease: List[str] = Field(default=[], description="疾病名称实体")
            symptom: List[str] = Field(default=[], description='疾病症状实体')
            drug: List[str] = Field(default=[], description='药物名称实体')

        response_schemas = ToolStrategy(Medical)
        format_instructions = response_schemas

        output_parser = StrOutputParser(response_schemas = response_schemas)

        ner_prompt = PromptTemplate(
            template = NER_PROMPT_TPL,
            partial_variables={'format_instructions': format_instructions},
            input_variables=['query']
        )

        ner_chain = ner_prompt | self._llm
        ner_response = ner_chain.invoke({'query': query})
        parsed_str = output_parser.parse(self._extract_content(ner_response))
        ner_result = json.loads(parsed_str)

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
            return []

        graph_documents = [
            Document(page_content=template['question'], metadata=template)
            for template in graph_templates
        ]
        db = FAISS.from_documents(graph_documents, get_embeddings_model())
        graph_documents_filter = db.similarity_search_with_relevance_scores(query, k=3)

        query_result = []
        neo4j_conn = get_neo4j_conn()
        for document,score in graph_documents_filter:
            if score < 0.5:
                break
            question = document.page_content
            cypher = document.metadata['cypher']
            answer = document.metadata['answer']
            try:
                result = neo4j_conn.run(cypher).data()
                if result and any(value for value in result[0].values()):
                    answer_str = replace_token_in_string(answer,list(result[0].items()))
                    formatted_text = f'问题{question}\n答案：{answer_str}'
                    if track_context:
                        self.current_retrieved_contexts.append(formatted_text)
                    query_result.append((formatted_text, score))
            except Exception:
                continue
        return query_result

    def _refine_answer(self, query: str, contexts: List[str]) -> str:
        contexts = [context for context in contexts if context]
        if not contexts:
            return '没有查到'
        initial_prompt = PromptTemplate.from_template(
            """你是一名专业的中文医疗助手，只能根据提供的参考信息回答问题，如果参考信息中没有答案，请回复“没有查到”。

                用户问题：{query}
                参考信息：
                {context}

                请给出答案："""
        )
        refine_prompt = PromptTemplate.from_template(
            """你是一名专业的中文医疗助手，已有的回答如下：
{existing_answer}

现在阅读新的参考信息，如有新的重要信息或修正，请更新回答；否则保持原回答。
不得编造参考信息之外的内容。

用户问题：{query}
新的参考信息：
{context}

请输出更新后的答案："""
        )

        answer = self._extract_content(
            (initial_prompt | self._llm).invoke({'query': query, 'context': contexts[0]})
        )

        for context in contexts[1:]:
            answer = self._extract_content(
                (refine_prompt | self._llm).invoke({
                    'query': query,
                    'context': context,
                    'existing_answer': answer
                })
            )

        return answer.strip() if answer else '没有查到'

    def _multi_channel_retrieve(self, query: str, vector_k: int = 10, top_n: int = 6) -> List[str]:
        vector_docs = self.vdb.similarity_search_with_relevance_scores(query, k=vector_k)
        vector_results = [
            (doc.page_content, score)
            for doc, score in vector_docs
            if doc.page_content
        ]
        graph_results = self._graph_context_search(query, track_context=False)

        vector_norm = self._normalize_scores(vector_results)
        graph_norm = self._normalize_scores(graph_results)

        combined = []
        for text, score in vector_norm:
            combined.append((text, score * 0.6))
        for text, score in graph_norm:
            combined.append((text, score * 0.4))

        combined.sort(key=lambda item: item[1], reverse=True)

        seen = set()
        merged = []
        for text, score in combined:
            if not text or text in seen:
                continue
            seen.add(text)
            merged.append(text)
            if len(merged) >= top_n:
                break

        return merged

    def retrival_func(self, query: str) -> str:
        query_result = self._multi_channel_retrieve(query)
        if query_result:
            self.current_retrieved_contexts.extend(query_result)
        return self._refine_answer(query, query_result)

    def graph_func(self, query: str) -> str:
        query_result_with_scores = self._graph_context_search(query)
        query_result = [text for text, _ in query_result_with_scores]
        if query_result:
            self.current_retrieved_contexts.extend(query_result)
        return self._refine_answer(query, query_result)


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
        self.current_retrieved_contexts = []
        self.chat_history.append(HumanMessage(content=query))
        if len(self.chat_history) > 4:
            self.chat_history = self.chat_history[-4:]

        try:
            result = self._agent.invoke({
                "messages": self.chat_history
            })
        except Exception:
            result = self._agent.invoke({"input": query,"chat-history": self.chat_history})
        final_answer = ""
        extracted_from_messages = False
        if isinstance(result, dict):
            messages = result.get("messages")
            if messages:
                last_msg = messages[-1]
                if isinstance(last_msg, dict):
                    final_answer = last_msg.get("content", "")
                    extracted_from_messages = True
                else:
                    final_answer = getattr(last_msg, "content", str(last_msg))
                    extracted_from_messages = True

        # 如果上面的逻辑没有提取到（比如不是 dict 或者没有 messages），则走原来的兜底逻辑
        if not extracted_from_messages:
            final_answer = self._extract_content(result)
        self.chat_history.append(AIMessage(content=final_answer))
        if return_context:
            # 如果没有查到任何东西，给一个默认值，防止 Ragas 计算 faithfulness 时报错
            contexts = self.current_retrieved_contexts if self.current_retrieved_contexts else ["未检索到相关文档"]
            return final_answer, contexts

        return final_answer

if __name__ == '__main__':
    agent = Agent()

    #print(agent.query('你好'))
    #print(agent.query('寻医问药网获得过哪些投资？'))
    # print(agent.query('鼻炎和感冒是并发症吗？'))
    #print(agent.query('鼻炎怎么治疗？'))
    # print(agent.query('烧橙子可以治感冒吗？'))


    #print(agent.generic_func('你好'))
    # print(agent.generic_func('你叫什么名字？'))

    #print(agent.retrival_func('介绍一下奥沙利铂注射液这个东西可以同来治疗什么疾病'))
    
    # print(agent.retrival_func('寻医问药网的客服电话是多少？'))

    # print(agent.graph_func('感冒和鼻炎是并发症吗？'))
    # print(agent.graph_func('感冒一般是由什么引起的？'))
    #print(agent.graph_func('感冒了现在咳嗽的厉害吃什么药好得快？可以吃阿莫西林吗？'))

    #print(agent.search_func('微软是什么？'))
