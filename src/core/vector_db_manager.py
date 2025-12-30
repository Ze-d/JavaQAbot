"""
向量数据库管理器
使用单例模式实现延迟初始化和连接池管理

作者：zjy
"""

import os
from langchain_chroma import Chroma
from src.utils.utils import get_embeddings_model
from src.utils.logger_config import setup_logger

logger_vdb = setup_logger('VectorDBManager', 'INFO')

class VectorDBManager:
    """向量数据库管理器 - 单例模式"""

    _instance = None
    _db = None

    @classmethod
    def get_instance(cls, db_path: str = None):
        """
        获取单例实例

        Args:
            db_path: 数据库路径，如果为None则使用环境变量

        Returns:
            VectorDBManager: 单例实例
        """
        if cls._instance is None:
            if db_path is None:
                db_path = os.getenv("VECTOR_DB_PATH", "resources/data/db")
            cls._instance = cls(db_path)
        return cls._instance

    def __init__(self, db_path: str):
        """
        初始化向量数据库管理器

        Args:
            db_path: 数据库路径
        """
        self.db_path = db_path
        logger_vdb.info(f"向量数据库管理器初始化，路径: {db_path}")

    def get_db(self) -> Chroma:
        """
        获取向量数据库实例（延迟初始化）

        Returns:
            Chroma: 向量数据库实例
        """
        if self._db is None:
            logger_vdb.info("首次加载向量数据库...")
            self._db = Chroma(
                persist_directory=self.db_path,
                embedding_function=get_embeddings_model()
            )
            logger_vdb.info("向量数据库加载完成")
        return self._db

    def reload_db(self):
        """重新加载向量数据库（用于测试或重置）"""
        logger_vdb.info("重新加载向量数据库...")
        self._db = None
        return self.get_db()

    def close(self):
        """关闭数据库连接"""
        if self._db:
            logger_vdb.info("关闭向量数据库连接")
            self._db = None

    @classmethod
    def reset_instance(cls):
        """重置单例实例（用于测试）"""
        logger_vdb.info("重置向量数据库管理器实例")
        cls._instance = None
        cls._db = None