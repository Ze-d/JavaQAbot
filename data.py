"""
测试数据生成模块
作者：zjy
创建时间：2024年

该模块用于从向量数据库生成测试数据集，包含以下功能：
1. 从Chroma向量数据库加载文档数据
2. 使用K-Means聚类筛选代表性样本
3. 基于LLM生成问答对
4. 输出为CSV格式的测试集

主要配置参数：
- CHROMA_DB_DIR: 向量数据库路径
- TESTSET_FILE: 输出测试集文件名
- N_CLUSTERS: K-Means聚类数量
- QUESTIONS_PER_DOC: 每个文档生成的问题数量
"""

import os
import warnings
from typing import List

import numpy as np
import pandas as pd
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from utils import get_llm_model, get_embeddings_model

# 忽略警告信息
warnings.filterwarnings('ignore')


# ================= 配置区域 =================
CHROMA_DB_DIR = './data/db'
TESTSET_FILE = "auto_generated_testset.csv"
N_CLUSTERS = 20
QUESTIONS_PER_DOC = 1
# =========================================


class QAData(BaseModel):
    """
    问答数据结构模型

    用于接收和验证LLM生成的问答对数据。
    """
    question: str = Field(description="生成的测试问题")
    answer: str = Field(description="问题的标准答案")
    type: str = Field(description="问题类型: simple 或 reasoning")


def main():
    """
    主函数：生成测试数据集

    流程：
    1. 初始化LLM和嵌入模型
    2. 从Chroma数据库加载文档
    3. 使用K-Means筛选代表性样本
    4. 生成问答对
    5. 保存为CSV文件
    """
    print("🚀 启动兜底方案：直接使用 LLM 生成测试集 (含格式自适应修复)...")

    # 初始化模型
    llm = get_llm_model()
    embedding_model = get_embeddings_model()

    print("📖 从 Chroma 加载数据...")
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_model)
    db_data = vectorstore.get()
    texts = db_data['documents']

    if not texts:
        raise ValueError("❌ 数据库为空！")

    # K-Means 筛选逻辑
    target_indices = []
    if len(texts) > N_CLUSTERS:
        print(f"⚡ 正在筛选 {N_CLUSTERS} 个代表性片段...")
        embeddings = vectorstore.get(include=['embeddings'])['embeddings']

        # 判空修复
        if embeddings is None or len(embeddings) == 0:
            print("   计算 Embeddings...")
            embeddings = embedding_model.embed_documents(texts)

        embeddings_np = np.array(embeddings)
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
        kmeans.fit(embeddings_np)
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings_np)
        target_indices = closest
    else:
        target_indices = range(len(texts))

    # 初始化解析器
    parser = JsonOutputParser(pydantic_object=QAData)

    # 强化 Prompt，明确要求单一对象
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个 QA 数据集生成专家。请根据用户提供的文本，生成一个 JSON 对象。\n"
                   "JSON 必须包含 keys: 'question', 'answer', 'type'。\n"
                   "type 只能是 'simple' 或 'reasoning'。\n"
                   "{format_instructions}"),
        ("human", "原文内容：\n{context}\n\n请生成 JSON：")
    ])

    chain = prompt | llm | parser

    results = []
    print(f"🧠 开始生成问题...")

    for i, idx in enumerate(target_indices):
        context_text = texts[idx]
        print(f"   Processing {i + 1}/{len(target_indices)}...", end="\r")

        try:
            response = chain.invoke({
                "context": context_text,
                "format_instructions": parser.get_format_instructions()
            })

            # 兼容列表和字典
            data_item = response

            # 如果返回的是列表 [{}], 取第一个元素
            if isinstance(response, list):
                if len(response) > 0:
                    data_item = response[0]
                else:
                    continue

            # 如果此时 data_item 不是字典，跳过
            if not isinstance(data_item, dict):
                print(f"\n   ⚠️ 第 {i + 1} 条格式异常 (类型: {type(data_item)}), 跳过...")
                continue

            # 容错取值 (防止大小写差异)
            question = data_item.get('question') or data_item.get('Question')
            answer = data_item.get('answer') or data_item.get('Answer')
            q_type = data_item.get('type') or 'simple'

            if not question or not answer:
                print(f"\n   ⚠️ 第 {i + 1} 条缺少必要字段, 跳过...")
                continue

            row = {
                'user_input': question,
                'reference': answer,
                'reference_contexts': [context_text],
                'type': q_type
            }
            results.append(row)

        except Exception as e:
            print(f"\n   ⚠️ 第 {i + 1} 条生成失败: {str(e)}")
            continue

    if not results:
        print("\n❌ 生成失败，未得到结果。")
        return

    df = pd.DataFrame(results)
    final_cols = ['user_input', 'reference', 'reference_contexts', 'type']
    save_cols = [c for c in final_cols if c in df.columns]

    df[save_cols].to_csv(TESTSET_FILE, index=False, encoding='utf-8-sig')

    print(f"\n\n🎉 成功生成 {len(df)} 条数据！")
    print(f"📂 已保存至: {TESTSET_FILE}")


if __name__ == "__main__":
    main()
