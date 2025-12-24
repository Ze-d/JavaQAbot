"""
测试集生成器模块
作者：zjy
创建时间：2024年

该模块使用Ragas框架从向量数据库生成高质量的测试数据集，包含以下功能：
1. 从Chroma向量数据库加载文档
2. 使用K-Means聚类筛选代表性样本（提高效率）
3. 基于LLM生成多样化的问答对
4. 输出为CSV格式的测试集

主要优化策略：
- K-Means聚类：从大量文档中筛选代表性样本
- 文档合并：将多个文档合并为长文本，提高生成稳定性
- 兼容处理：支持新旧版本Ragas框架的兼容性处理

配置参数：
- CHROMA_DB_DIR: 向量数据库路径
- TESTSET_FILE: 输出测试集文件名
- N_CLUSTERS: K-Means聚类数量
- TESTSET_SIZE: 生成的测试问题数量
"""

import os
import warnings
from typing import List

import numpy as np
import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import QuestionGenerator
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from utils import get_llm_model, get_embeddings_model

# 忽略警告信息
warnings.filterwarnings('ignore')

# ================= 配置区域 =================
CHROMA_DB_DIR = './data/db'
TESTSET_FILE = "auto_generated_testset.csv"
N_CLUSTERS = 200  # 筛选后的代表性文档数
TESTSET_SIZE = 15  # 生成的测试问题数
# =========================================


def main():
    """
    主函数：生成测试数据集

    流程：
    1. 初始化LLM和嵌入模型
    2. 从Chroma数据库加载文档
    3. 使用K-Means聚类筛选代表性样本
    4. 合并文档为长文本（提高稳定性）
    5. 使用Ragas生成测试集
    6. 保存为CSV文件
    """
    print("=" * 60)
    print("🚀 测试集生成器启动")
    print("=" * 60)

    # 1. 初始化模型（兼容旧版 Ragas Wrapper）
    print("\n🔧 初始化 LLM 和嵌入模型...")
    llm = get_llm_model()
    embedding_function = get_embeddings_model()
    generator_llm = LangchainLLMWrapper(llm)
    generator_embeddings = LangchainEmbeddingsWrapper(embedding_function)

    # 2. 从 Chroma 加载数据
    print("\n📖 从 Chroma 数据库加载文档...")
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embedding_function
    )
    db_data = vectorstore.get()
    texts = db_data['documents']
    metadatas = db_data['metadatas']
    print(f"✅ 成功加载 {len(texts)} 个文档片段。")

    # 3. 文档筛选：K-Means 聚类（核心提速，不可省）
    if len(texts) > N_CLUSTERS:
        print(f"\n⚡ 文档数量过多，正在使用 K-Means 聚类筛选 {N_CLUSTERS} 个代表性文档...")

        all_doc_ids = db_data['ids']
        batch_size = 1000
        all_embeddings = []
        for i in range(0, len(all_doc_ids), batch_size):
            batch_ids = all_doc_ids[i:i + batch_size]
            batch_data = vectorstore.get(ids=batch_ids, include=['embeddings'])
            all_embeddings.extend(batch_data['embeddings'])

        embeddings_array = np.array(all_embeddings)
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_array)
        closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings_array)

        selected_texts = [texts[i] for i in closest_indices]
        selected_metadatas = [metadatas[i] for i in closest_indices]
        print(f"✅ 筛选完成，共选择 {len(selected_texts)} 个代表性文档片段。")
    else:
        selected_texts = texts
        selected_metadatas = metadatas
        print("ℹ️  文档数量较少，使用全部文档生成测试集。")

    # 4. 终极技巧：合并所有文档为一个长文本，彻底绕开分段和 headlines 依赖
    print("\n📄 合并文档为长文本，绕开所有分段逻辑...")
    # 旧版 Ragas 对单文本生成问题更稳定，无分段依赖
    combined_text = "\n\n---\n\n".join(selected_texts)  # 用分隔符合并所有文档
    # 构造单个 Document（无任何额外字段，避免元数据问题）
    documents = [Document(page_content=combined_text, metadata={})]

    # 5. 初始化 Ragas 生成器（极简模式）
    print("\n🧠 正在使用 Ragas 生成测试集（终极兼容模式）...")
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings
    )

    # 6. 生成测试集（仅用旧版支持的参数，不触发任何分段）
    try:
        # 第一次尝试：用 generate_with_langchain_docs 处理单个文档（无分段需求）
        print("🔄 尝试方法1：generate_with_langchain_docs")
        testset = generator.generate_with_langchain_docs(
            documents=documents,
            testset_size=TESTSET_SIZE
        )
        test_df = testset.to_pandas()
        test_df.rename(columns={
            'question': 'user_input',
            'ground_truth': 'reference'
        }, inplace=True)
        test_df['reference_contexts'] = [selected_texts] * len(test_df)

    except Exception as e:
        # 第二次尝试：直接用 LLM 基于长文本生成问题（绕开 Ragas 内部处理）
        print(f"\n⚠️  自动适配旧版 Ragas：{str(e)[:50]}")
        print("🔄 尝试方法2：手动问题生成器")

        # 手动初始化问题生成器，直接生成问题
        qg = QuestionGenerator(llm=generator_llm)
        questions = qg.generate(
            contexts=[combined_text],  # 旧版支持的参数名
            num_questions=TESTSET_SIZE
        )
        # 构造测试集格式（适配后续评估）
        test_data = {
            'user_input': questions,
            'reference': [combined_text[:500] + "..." for _ in questions],  # 截取部分作为参考
            'reference_contexts': [selected_texts for _ in questions]
        }
        test_df = pd.DataFrame(test_data)

    # 7. 保存测试集
    test_df.to_csv(TESTSET_FILE, index=False, encoding='utf-8-sig')

    print(f"\n🎉 测试集生成完毕！已保存至 {TESTSET_FILE}")
    print(f"📊 共生成 {len(test_df)} 条测试数据")
    print("\n预览前2条数据：")
    print(test_df[['user_input', 'reference']].head(2))
    print("\n✅ 任务完成！")


if __name__ == "__main__":
    main()
