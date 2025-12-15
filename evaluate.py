import os
import ast
import pandas as pd
from datasets import Dataset
from agent import Agent  # 你的 Agent 类
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    faithfulness
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


TESTSET_FILE = "auto_generated_testset.csv"
EVALUATOR_MODEL = "deepseek-chat"



def parse_context(context_str):

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

    test_df = pd.read_csv(TESTSET_FILE,encoding='utf-8')
    required_columns = ["user_input", "reference", "reference_contexts"]
    test_df = test_df.dropna(subset=required_columns)

    agent = Agent()


    # 提取关键数据
    questions = test_df["user_input"].tolist()
    ground_truths = [str(gt) for gt in test_df["reference"].tolist()]
    # 解析 reference_contexts 为实际列表（用于后续对比检索效果）
    reference_contexts = [parse_context(ctx) for ctx in test_df["reference_contexts"].tolist()]

    generated_answers = []
    retrieved_contexts = []  # 存储 Agent 实际检索到的上下文

    print(f" ({len(questions)} 题)...")

    for i, q in enumerate(questions):
        print(f"\n📝 [题 {i + 1}] {q}")

        # 调用 Agent 并获取答案和检索上下文
        ans, ctx = agent.query(q, return_context=True)

        print(f"   🗣️ 答: {ans[:50]}...")
        print(f"   📚 检索到 {len(ctx)} 条证据")

        generated_answers.append(str(ans))
        retrieved_contexts.append(ctx)


    # 构建 Ragas 评估数据集
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": generated_answers,
        "contexts": retrieved_contexts,  # Agent 实际检索的上下文
        "ground_truth": ground_truths,
        "reference_contexts": reference_contexts  # 原始参考上下文（可选，用于分析）
    })

    # 配置评估模型（使用你的 utils 函数获取 LLM 和 Embeddings）
    from utils import get_llm_model, get_embeddings_model  # 确保导入正确
    llm = get_llm_model()
    embedding_function = get_embeddings_model()

    evaluator_llm = LangchainLLMWrapper(llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embedding_function)

    # 选择评估指标
    metrics = [
        answer_correctness,  # 答案准确率（与 ground_truth 对比）
        answer_relevancy,  # 答案相关性（与问题的匹配度）
        faithfulness  # 忠实度（答案是否基于检索上下文，无幻觉）
    ]

    print(f"\n⚖️ 正在进行全维度打分 (含幻觉检测)...")
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    print("\n====== 🏆 完整评估报告 ======")
    print(results)

    # 保存结果（包含原始数据和评估分数）
    result_df = results.to_pandas()
    # 合并原始测试集数据，方便分析
    original_df = test_df.reset_index(drop=True)
    final_df = pd.concat([original_df, result_df], axis=1)
    final_df.to_excel("full_evaluation_report.xlsx", index=False)

    print("✅ 报告已保存至 full_evaluation_report.xlsx")


if __name__ == "__main__":
    main()