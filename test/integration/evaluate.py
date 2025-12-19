"""
模型评估模块
作者：zjy
创建时间：2024年

该模块使用Ragas框架对医疗问诊Agent进行全面评估，包括：
1. 加载测试数据集
2. 运行Agent生成回答
3. 使用多个评估指标（准确率、相关性、忠实度）
4. 生成详细的评估报告

评估指标：
- answer_correctness: 答案准确率（与标准答案对比）
- answer_relevancy: 答案相关性（与问题的匹配度）
- faithfulness: 忠实度（答案是否基于检索上下文，无幻觉）

输出：
- 控制台显示评估结果
- Excel文件保存详细报告
"""

import ast
import os
from typing import List

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    faithfulness
)

from agent import Agent
from utils import get_llm_model, get_embeddings_model


# 配置参数
TESTSET_FILE = "auto_generated_testset.csv"
EVALUATOR_MODEL = "deepseek-chat"


def parse_context(context_str: str) -> List[str]:
    """
    解析上下文字符串为列表

    将CSV中存储的字符串格式上下文转换为Python列表。

    Args:
        context_str (str): 上下文字符串，可能包含列表格式

    Returns:
        List[str]: 解析后的上下文列表
    """
    # 处理空值或空字符串
    if pd.isna(context_str) or context_str == "":
        return []

    try:
        # 清理换行符和多余空格，避免解析错误
        clean_str = context_str.replace("\n", "").strip()
        return ast.literal_eval(clean_str)
    except (SyntaxError, ValueError):
        # 若解析失败，返回单元素列表（包含原始字符串）
        return [str(context_str)]


def main():
    """
    主评估流程

    执行以下步骤：
    1. 加载测试数据集
    2. 初始化Agent和评估模型
    3. 对每个问题生成回答和检索上下文
    4. 使用Ragas框架进行全面评估
    5. 生成并保存评估报告
    """
    print("=" * 60)
    print("🏥 医疗问诊Agent评估程序启动")
    print("=" * 60)

    # 1. 加载测试数据集
    print("\n📊 加载测试数据集...")
    test_df = pd.read_csv(TESTSET_FILE, encoding='utf-8')
    required_columns = ["user_input", "reference", "reference_contexts"]
    test_df = test_df.dropna(subset=required_columns)
    print(f"✅ 成功加载 {len(test_df)} 条测试数据")

    # 2. 初始化Agent
    print("\n🤖 初始化Agent实例...")
    agent = Agent()

    # 3. 提取关键数据
    questions = test_df["user_input"].tolist()
    ground_truths = [str(gt) for gt in test_df["reference"].tolist()]
    # 解析 reference_contexts 为实际列表（用于后续对比检索效果）
    reference_contexts = [parse_context(ctx) for ctx in test_df["reference_contexts"].tolist()]

    generated_answers = []
    retrieved_contexts = []  # 存储 Agent 实际检索到的上下文

    print(f"\n💬 开始生成回答 (共 {len(questions)} 题)...")

    # 4. 遍历所有问题，生成回答
    for i, q in enumerate(questions):
        print(f"\n📝 [题 {i + 1}] {q}")

        # 调用 Agent 并获取答案和检索上下文
        ans, ctx = agent.query(q, return_context=True)

        print(f"   🗣️ 答: {ans[:50]}...")
        print(f"   📚 检索到 {len(ctx)} 条证据")

        generated_answers.append(str(ans))
        retrieved_contexts.append(ctx)

    # 5. 构建 Ragas 评估数据集
    print("\n🔧 构建Ragas评估数据集...")
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": generated_answers,
        "contexts": retrieved_contexts,  # Agent 实际检索的上下文
        "ground_truth": ground_truths,
        "reference_contexts": reference_contexts  # 原始参考上下文（可选，用于分析）
    })

    # 6. 配置评估模型
    print("\n⚙️ 配置评估模型...")
    llm = get_llm_model()
    embedding_function = get_embeddings_model()

    evaluator_llm = LangchainLLMWrapper(llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embedding_function)

    # 7. 选择评估指标
    metrics = [
        answer_correctness,  # 答案准确率（与 ground_truth 对比）
        answer_relevancy,  # 答案相关性（与问题的匹配度）
        faithfulness  # 忠实度（答案是否基于检索上下文，无幻觉）
    ]

    # 8. 执行评估
    print(f"\n⚖️ 正在进行全维度打分 (含幻觉检测)...")
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    # 9. 显示评估结果
    print("\n" + "=" * 60)
    print("🏆 完整评估报告")
    print("=" * 60)
    print(results)

    # 10. 保存详细报告
    print("\n💾 保存详细评估报告...")
    result_df = results.to_pandas()
    # 合并原始测试集数据，方便分析
    original_df = test_df.reset_index(drop=True)
    final_df = pd.concat([original_df, result_df], axis=1)
    final_df.to_excel("full_evaluation_report.xlsx", index=False)

    print("✅ 报告已保存至 full_evaluation_report.xlsx")
    print("\n🎉 评估完成！")


if __name__ == "__main__":
    main()
