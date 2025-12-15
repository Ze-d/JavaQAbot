"""
Neo4j连接测试脚本
用于诊断连接问题和配置错误
"""
import os
import sys
from dotenv import load_dotenv
from py2neo import Graph
from logger_config import logger_utils

def test_neo4j_connection():
    """测试Neo4j连接"""
    print("=" * 60)
    print("🔍 Neo4j 连接诊断工具")
    print("=" * 60)

    # 1. 检查环境变量
    print("\n📋 1. 检查环境变量:")
    uri = os.getenv('NEO4J_URI')
    username = os.getenv('NEO4J_USERNAME')
    password = os.getenv('NEO4J_PASSWORD')

    print(f"   NEO4J_URI: {uri if uri else '❌ 未设置'}")
    print(f"   NEO4J_USERNAME: {username if username else '❌ 未设置'}")
    print(f"   NEO4J_PASSWORD: {'*' * len(password) if password else '❌ 未设置'}")

    # 2. 检查 .env 文件
    print("\n📁 2. 检查 .env 文件:")
    if os.path.exists('.env'):
        print("   ✅ 找到 .env 文件")
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    print(f"   {key} = {value[:20]}{'...' if len(value) > 20 else ''}")
    else:
        print("   ❌ 未找到 .env 文件")

    # 3. 测试连接
    print("\n🔗 3. 测试连接:")
    try:
        # 使用环境变量
        if uri and username and password:
            print("   使用环境变量连接...")
            graph = Graph(uri, auth=(username, password))
        else:
            # 使用默认配置
            print("   使用默认配置连接...")
            uri = 'bolt://localhost:7687'
            username = 'neo4j'
            password = '123456'
            graph = Graph(uri, auth=(username, password))

        # 测试查询
        result = graph.run("RETURN 1 AS test, 'Neo4j连接成功' AS message").data()
        print(f"   ✅ 连接成功！")
        print(f"   测试结果: {result}")

        # 查看数据库信息
        print("\n📊 4. 数据库信息:")
        try:
            # 节点数量
            node_count = graph.run("MATCH (n) RETURN count(n) AS count").data()[0]['count']
            print(f"   节点总数: {node_count}")

            # 关系数量
            rel_count = graph.run("MATCH ()-[r]->() RETURN count(r) AS count").data()[0]['count']
            print(f"   关系总数: {rel_count}")

            # 标签列表
            labels = graph.run("CALL db.labels()").data()
            print(f"   节点标签: {[label['label'] for label in labels]}")
        except Exception as e:
            print(f"   ⚠️ 获取数据库信息失败: {e}")

    except Exception as e:
        print(f"   ❌ 连接失败: {e}")
        print(f"\n   错误类型: {type(e).__name__}")

        # 分析错误原因
        print("\n🔧 5. 错误分析:")
        error_msg = str(e).lower()

        if "connection refused" in error_msg:
            print("   🔍 可能原因:")
            print("      - Neo4j 服务未启动")
            print("      - 端口 7687 被占用")
            print("      - 防火墙阻止连接")
            print("   💡 解决方案:")
            print("      - 启动 Neo4j: neo4j start")
            print("      - 检查端口: netstat -an | grep 7687")
        elif "authentication failed" in error_msg:
            print("   🔍 可能原因:")
            print("      - 用户名或密码错误")
            print("      - 未重置初始密码")
            print("   💡 解决方案:")
            print("      - 重置密码: neo4j-admin set-initial-password 你的密码")
            print("      - 确保密码与代码中一致")
        elif "service unavailable" in error_msg:
            print("   🔍 可能原因:")
            print("      - Neo4j 服务异常")
            print("      - 数据库文件损坏")
            print("   💡 解决方案:")
            print("      - 重启服务: neo4j restart")
            print("      - 查看日志: neo4j logs")
        else:
            print(f"   🔍 未知错误: {error_msg}")

    # 6. 修复建议
    print("\n" + "=" * 60)
    print("💡 修复建议:")
    print("=" * 60)
    print("1. 确保 Neo4j 服务正在运行")
    print("2. 设置正确的环境变量:")
    print("   NEO4J_URI=bolt://localhost:7687")
    print("   NEO4J_USERNAME=neo4j")
    print("   NEO4J_PASSWORD=your_password")
    print("3. 或者创建 .env 文件并设置上述变量")
    print("4. 重置密码（如需要）:")
    print("   neo4j-admin set-initial-password 123456")
    print("5. 检查防火墙设置")
    print("6. 验证连接: http://localhost:7474")
    print("=" * 60)

if __name__ == '__main__':
    # 加载 .env 文件
    load_dotenv()

    # 运行测试
    test_neo4j_connection()
