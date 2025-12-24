"""
图数据库查询调试脚本
详细查看每个步骤的执行情况，找出匹配不到模板的原因

作者：zjy
"""

import os
import sys
import json
from typing import List

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents.structured_output import ToolStrategy
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field

from src.utils.utils import get_llm_model, get_embeddings_model, replace_token_in_string
from src.core.config import GRAPH_TEMPLATE
from src.prompts.prompt import NER_PROMPT_TPL
from src.utils.logger_config import setup_logger

# 设置日志
logger_debug = setup_logger('GraphDebug', 'DEBUG')


def debug_graph_query(query: str):
    """详细调试图数据库查询流程"""

    print("=" * 80)
    print(f"🔍 图数据库查询调试 - 问题: {query}")
    print("=" * 80)

    # 步骤1：定义Java技术实体提取模型
    print("\n📋 步骤1: 定义NER模型")
    class JavaTech(BaseModel):
        class_or_interface: List[str] = Field(default=[], description="Java类或接口实体")
        framework: List[str] = Field(default=[], description="Java框架实体")
        method_name: List[str] = Field(default=[], description="Java方法实体")
        technology: List[str] = Field(default=[], description="Java技术实体")

    print("   ✅ 模型定义完成")
    print(f"   字段: {list(JavaTech.model_fields.keys())}")

    # 步骤2-3：配置结构化输出
    print("\n📋 步骤2-3: 配置输出解析器")
    response_schemas = ToolStrategy(JavaTech)
    format_instructions = response_schemas
    output_parser = StrOutputParser(response_schemas=response_schemas)
    print("   ✅ 输出解析器配置完成")

    # 步骤4-5：执行实体提取
    print("\n📋 步骤4-5: 执行NER实体提取")
    ner_prompt = PromptTemplate(
        template=NER_PROMPT_TPL,
        partial_variables={'format_instructions': format_instructions},
        input_variables=['query']
    )

    llm = get_llm_model()
    ner_chain = ner_prompt | llm

    try:
        ner_response = ner_chain.invoke({'query': query})
        print(f"   🔍 原始LLM响应: {ner_response.content[:200]}...")

        parsed_str = output_parser.parse(ner_response.content)
        print(f"   🔍 解析后的字符串: {parsed_str[:200]}...")

        ner_result = json.loads(parsed_str)
        print(f"   ✅ NER提取结果:")
        for key, value in ner_result.items():
            print(f"      {key}: {value}")

        # 检查是否有提取到实体
        has_entities = any([
            ner_result.get('class_or_interface', []),
            ner_result.get('framework', []),
            ner_result.get('method_name', []),
            ner_result.get('technology', [])
        ])

        if not has_entities:
            print("   ❌ 警告: 未提取到任何实体！")
            return None
        else:
            print("   ✅ 成功提取到实体")

    except Exception as e:
        print(f"   ❌ NER提取失败: {e}")
        logger_debug.error(f"NER提取失败: {e}")
        return None

    # 步骤6：模板匹配和填充
    print("\n📋 步骤6: 模板匹配和填充")
    graph_templates = []

    print(f"   📊 可用模板数量: {len(GRAPH_TEMPLATE)}")
    print(f"   📊 模板类型: {list(GRAPH_TEMPLATE.keys())}")

    for template_name, template in GRAPH_TEMPLATE.items():
        slot = template['slots'][0]  # 获取模板需要的槽位
        slot_values = ner_result.get(slot, [])  # 从NER结果中获取对应的实体

        print(f"\n   🔍 模板: {template_name}")
        print(f"      需要槽位: {slot}")
        print(f"      提取到的值: {slot_values}")

        if slot_values:
            print(f"      ✅ 槽位匹配成功")
            for value in slot_values:
                filled_template = {
                    'question': replace_token_in_string(template['question'], [[slot, value]]),
                    'cypher': replace_token_in_string(template['cypher'], [[slot, value]]),
                    'answer': replace_token_in_string(template['answer'], [[slot, value]]),
                }
                graph_templates.append(filled_template)
                print(f"         填充结果: {filled_template['question'][:50]}...")
        else:
            print(f"      ❌ 槽位匹配失败")

    print(f"\n   ✅ 匹配到 {len(graph_templates)} 个查询模板")

    if not graph_templates:
        print("   ❌ 错误: 没有匹配到任何模板！")
        return None

    # 步骤7：相似度筛选
    print("\n📋 步骤7: 相似度筛选")
    try:
        graph_documents = [
            Document(page_content=template['question'], metadata=template)
            for template in graph_templates
        ]
        print(f"   📊 构建了 {len(graph_documents)} 个文档")

        db = FAISS.from_documents(graph_documents, get_embeddings_model())
        graph_documents_filter = db.similarity_search_with_relevance_scores(query, k=3)
        print(f"   📊 相似度筛选后剩余 {len(graph_documents_filter)} 个模板")

        for i, (doc, score) in enumerate(graph_documents_filter):
            print(f"      {i+1}. 相似度: {score:.3f}, 问题: {doc.page_content[:50]}...")

    except Exception as e:
        print(f"   ❌ 相似度筛选失败: {e}")
        logger_debug.error(f"相似度筛选失败: {e}")
        return None

    # 步骤8：返回结果供后续测试
    print("\n" + "=" * 80)
    print("调试完成，可以进行图数据库查询测试")
    print("=" * 80)

    return {
        'ner_result': ner_result,
        'graph_templates': graph_templates,
        'filtered_templates': graph_documents_filter
    }


def test_neo4j_connection():
    """测试Neo4j连接"""
    print("\n" + "=" * 80)
    print("🔗 测试Neo4j连接")
    print("=" * 80)

    try:
        from src.utils.utils import get_neo4j_conn
        conn = get_neo4j_conn()
        result = conn.run("RETURN 1 AS test").data()
        print(f"   ✅ Neo4j连接成功: {result}")
        return True
    except Exception as e:
        print(f"   ❌ Neo4j连接失败: {e}")
        return False


def test_with_sample_data():
    """使用示例数据测试完整流程"""
    print("\n" + "=" * 80)
    print("🧪 使用示例数据测试")
    print("=" * 80)

    # 模拟NER提取结果
    sample_ner_result = {
        'class_or_interface': ['Spring Boot'],
        'framework': [],
        'method_name': [],
        'technology': []
    }

    print(f"   📊 模拟NER结果: {sample_ner_result}")

    # 使用模拟数据进行模板匹配
    graph_templates = []
    for template_name, template in GRAPH_TEMPLATE.items():
        slot = template['slots'][0]
        slot_values = sample_ner_result.get(slot, [])

        if slot_values:
            for value in slot_values:
                filled_template = {
                    'question': replace_token_in_string(template['question'], [[slot, value]]),
                    'cypher': replace_token_in_string(template['cypher'], [[slot, value]]),
                    'answer': replace_token_in_string(template['answer'], [[slot, value]]),
                }
                graph_templates.append(filled_template)

    print(f"   ✅ 使用模拟数据匹配到 {len(graph_templates)} 个模板")

    if graph_templates:
        print("   📋 前3个匹配的模板:")
        for i, template in enumerate(graph_templates[:3]):
            print(f"      {i+1}. {template['question']}")
            print(f"         Cypher: {template['cypher'][:100]}...")
    else:
        print("   ❌ 使用模拟数据也没有匹配到模板")


if __name__ == '__main__':
    # 1. 测试示例数据
    test_with_sample_data()

    # 2. 测试NER连接
    test_neo4j_connection()

    # 3. 调试实际查询
    test_query = "Spring Boot是什么？"
    debug_graph_query(test_query)