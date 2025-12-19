"""
测试调试日志系统
用于验证日志配置是否正常工作
"""
import os
import sys

# 确保能找到模块
sys.path.append(os.path.dirname(__file__))

from logger_config import (
    logger_agent,
    logger_service,
    logger_utils,
    logger_app,
    logger_data
)

def test_loggers():
    """测试所有日志记录器是否正常工作"""
    print("=" * 60)
    print("🔍 医疗QA聊天机器人 - 日志系统测试")
    print("=" * 60)

    # 测试Agent日志
    print("\n📍 测试 Agent 模块日志:")
    logger_agent.debug("这是一条DEBUG日志 - Agent模块")
    logger_agent.info("这是一条INFO日志 - Agent模块")
    logger_agent.warning("这是一条WARNING日志 - Agent模块")
    logger_agent.error("这是一条ERROR日志 - Agent模块")

    # 测试Service日志
    print("\n📍 测试 Service 模块日志:")
    logger_service.debug("这是一条DEBUG日志 - Service模块")
    logger_service.info("这是一条INFO日志 - Service模块")
    logger_service.warning("这是一条WARNING日志 - Service模块")
    logger_service.error("这是一条ERROR日志 - Service模块")

    # 测试Utils日志
    print("\n📍 测试 Utils 模块日志:")
    logger_utils.debug("这是一条DEBUG日志 - Utils模块")
    logger_utils.info("这是一条INFO日志 - Utils模块")
    logger_utils.warning("这是一条WARNING日志 - Utils模块")
    logger_utils.error("这是一条ERROR日志 - Utils模块")

    # 测试App日志
    print("\n📍 测试 App 模块日志:")
    logger_app.debug("这是一条DEBUG日志 - App模块")
    logger_app.info("这是一条INFO日志 - App模块")
    logger_app.warning("这是一条WARNING日志 - App模块")
    logger_app.error("这是一条ERROR日志 - App模块")

    # 测试Data日志
    print("\n📍 测试 DataProcess 模块日志:")
    logger_data.debug("这是一条DEBUG日志 - DataProcess模块")
    logger_data.info("这是一条INFO日志 - DataProcess模块")
    logger_data.warning("这是一条WARNING日志 - DataProcess模块")
    logger_data.error("这是一条ERROR日志 - DataProcess模块")

    # 测试日志文件
    print("\n📁 检查日志文件:")
    log_files = [f for f in os.listdir('logs') if f.endswith('.log')]
    if log_files:
        print(f"✅ 找到 {len(log_files)} 个日志文件:")
        for f in log_files:
            size = os.path.getsize(f'logs/{f}')
            print(f"   - {f} ({size} bytes)")
    else:
        print("❌ 未找到日志文件")

    print("\n" + "=" * 60)
    print("✅ 日志系统测试完成！")
    print("=" * 60)

if __name__ == '__main__':
    # 确保logs目录存在
    os.makedirs('logs', exist_ok=True)

    # 运行测试
    test_loggers()

    print("\n💡 提示:")
    print("1. 查看控制台输出确认日志显示正常")
    print(f"2. 检查 logs/ 目录中的日志文件")
    print("3. 运行 python app.py 测试完整流程")
