"""
日志配置文件
为医疗QA聊天机器人项目提供统一的日志系统
"""
import logging
import sys
from datetime import datetime

def setup_logger(name: str, level: str = 'INFO') -> logging.Logger:
    """
    设置日志配置

    Args:
        name: 日志记录器名称
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR)

    Returns:
        配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - [%(name)s] - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)  # ✅ 允许DEBUG级别日志输出
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(f'logs/medical_qa_{timestamp}.log', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# 创建主要模块的日志记录器
logger_agent = setup_logger('Agent', 'DEBUG')
logger_service = setup_logger('Service', 'INFO')
logger_utils = setup_logger('Utils', 'INFO')
logger_app = setup_logger('App', 'INFO')
logger_data = setup_logger('DataProcess', 'INFO')
