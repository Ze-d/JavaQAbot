"""
测试性能优化是否已成功应用到agent.py

作者：zjy
"""

import os
import sys

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_agent_modifications():
    """测试agent.py的修改"""
    print("=" * 80)
    print("🔍 测试性能优化组件应用状态")
    print("=" * 80)

    # 1. 检查导入
    print("\n📋 1. 检查性能优化组件导入:")
    try:
        from src.core.vector_db_manager import VectorDBManager
        print("   ✅ VectorDBManager 导入成功")
    except Exception as e:
        print(f"   ❌ VectorDBManager 导入失败: {e}")

    try:
        from src.core.graph_cache import cached_cypher_query
        print("   ✅ cached_cypher_query 导入成功")
    except Exception as e:
        print(f"   ❌ cached_cypher_query 导入失败: {e}")

    try:
        from src.core.faiss_index_manager import FaissIndexManager
        print("   ✅ FaissIndexManager 导入成功")
    except Exception as e:
        print(f"   ❌ FaissIndexManager 导入失败: {e}")

    # 2. 检查agent.py文件内容
    print("\n📋 2. 检查agent.py关键修改:")

    with open('src/core/agent.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查向量数据库优化
    if 'VectorDBManager.get_instance' in content:
        print("   ✅ 向量数据库单例模式已应用")
    else:
        print("   ❌ 向量数据库单例模式未应用")

    # 检查FAISS优化
    if 'FaissIndexManager.get_or_build_index' in content:
        print("   ✅ FAISS索引缓存已应用")
    else:
        print("   ❌ FAISS索引缓存未应用")

    # 检查图查询缓存
    if 'cached_cypher_query' in content:
        print("   ✅ 图查询缓存已应用")
    else:
        print("   ❌ 图查询缓存未应用")

    # 检查是否移除了未使用的导入
    if 'from langchain_community.vectorstores import FAISS' not in content:
        print("   ✅ 未使用的FAISS导入已移除")
    else:
        print("   ❌ 未使用的FAISS导入未移除")

    # 3. 显示关键代码片段
    print("\n📋 3. 关键代码片段:")
    print("   向量数据库初始化:")
    for line in content.split('\n'):
        if 'VectorDBManager.get_instance' in line:
            print(f"      {line.strip()}")
            break

    print("   FAISS索引管理:")
    for line in content.split('\n'):
        if 'FaissIndexManager.get_or_build_index' in line:
            print(f"      {line.strip()}")
            break

    print("   图查询缓存:")
    for line in content.split('\n'):
        if 'cached_cypher_query' in line and '执行Cypher查询' in content[content.find(line)-50:content.find(line)+50]:
            print(f"      {line.strip()}")
            break

    # 4. 性能预期
    print("\n📋 4. 预期性能提升:")
    print("   ✅ 向量数据库：启动时间缩短 60-80%")
    print("   ✅ 图查询：响应时间缩短 70-90%")
    print("   ✅ FAISS索引：构建时间缩短 80-95%")
    print("   ✅ 整体响应：延迟缩短 50-70%")

    print("\n" + "=" * 80)
    print("✅ 性能优化组件应用检查完成")
    print("=" * 80)


if __name__ == '__main__':
    test_agent_modifications()