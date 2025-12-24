"""
简单的NER测试脚本
"""

import os
import sys
import json

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    from src.utils.utils import get_llm_model
    from src.prompts.prompt import NER_PROMPT_TPL
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain.agents.structured_output import ToolStrategy
    from pydantic import BaseModel, Field
    from typing import List

    print("导入成功")

    # 定义模型
    class JavaTech(BaseModel):
        class_or_interface: List[str] = Field(default=[], description="Java类或接口实体")
        framework: List[str] = Field(default=[], description="Java框架实体")
        method_name: List[str] = Field(default=[], description="Java方法实体")
        technology: List[str] = Field(default=[], description="Java技术实体")

    # 配置输出
    response_schemas = ToolStrategy(JavaTech)
    format_instructions = response_schemas
    output_parser = StrOutputParser(response_schemas=response_schemas)

    # 构建提示词
    ner_prompt = PromptTemplate(
        template=NER_PROMPT_TPL,
        partial_variables={'format_instructions': format_instructions},
        input_variables=['query']
    )

    # 获取模型
    llm = get_llm_model()
    ner_chain = ner_prompt | llm

    # 测试查询
    query = "Spring Boot是什么？"
    print(f"测试查询: {query}")

    # 执行NER
    ner_response = ner_chain.invoke({'query': query})
    print(f"LLM响应: {ner_response.content}")

    # 解析结果
    parsed_str = output_parser.parse(ner_response.content)
    print(f"解析字符串: {parsed_str}")

    # 解析JSON
    ner_result = json.loads(parsed_str)
    print(f"NER结果: {ner_result}")

except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()