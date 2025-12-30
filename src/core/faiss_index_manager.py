"""
FAISS索引管理器
用于缓存和复用FAISS索引，避免频繁重建

作者：zjy
"""

import hashlib
import pickle
import os
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.utils.utils import get_embeddings_model
from src.utils.logger_config import setup_logger

logger_faiss = setup_logger('FaissIndexManager', 'INFO')

# 索引缓存目录
FAISS_CACHE_DIR = "resources/data/faiss_cache"
os.makedirs(FAISS_CACHE_DIR, exist_ok=True)

class FaissIndexManager:
    """FAISS索引管理器"""

    _index_cache = {}  # 内存缓存

    @classmethod
    def _compute_documents_hash(cls, documents: List[Document]) -> str:
        """
        计算文档列表的哈希值

        Args:
            documents: 文档列表

        Returns:
            str: 哈希值
        """
        content = "".join([doc.page_content for doc in documents])
        return hashlib.md5(content.encode()).hexdigest()

    @classmethod
    def _get_cache_path(cls, doc_hash: str) -> str:
        """
        获取缓存文件路径

        Args:
            doc_hash: 文档哈希值

        Returns:
            str: 缓存文件路径
        """
        return os.path.join(FAISS_CACHE_DIR, f"{doc_hash}.pkl")

    @classmethod
    def get_or_build_index(cls, graph_documents: List[Document]) -> FAISS:
        """
        获取或构建FAISS索引

        Args:
            graph_documents: 图查询文档列表

        Returns:
            FAISS: FAISS索引实例
        """
        doc_hash = cls._compute_documents_hash(graph_documents)

        # 1. 尝试从内存缓存获取
        if doc_hash in cls._index_cache:
            logger_faiss.debug(f"内存缓存命中: {len(graph_documents)}个文档")
            return cls._index_cache[doc_hash]

        # 2. 尝试从磁盘缓存加载
        cache_path = cls._get_cache_path(doc_hash)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    index = pickle.load(f)
                logger_faiss.info(f"磁盘缓存加载成功: {len(graph_documents)}个文档")
                cls._index_cache[doc_hash] = index
                return index
            except Exception as e:
                logger_faiss.warning(f"加载磁盘缓存失败: {e}")

        # 3. 构建新索引
        logger_faiss.info(f"构建新索引: {len(graph_documents)}个文档")
        try:
            # 使用嵌入模型构建索引
            index = FAISS.from_documents(graph_documents, get_embeddings_model())

            # 保存到磁盘缓存
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(index, f)
                logger_faiss.info(f"索引已保存到磁盘: {cache_path}")
            except Exception as e:
                logger_faiss.warning(f"保存磁盘缓存失败: {e}")

            # 缓存到内存
            cls._index_cache[doc_hash] = index

            # 限制内存缓存大小
            if len(cls._index_cache) > 10:
                oldest_key = next(iter(cls._index_cache))
                del cls._index_cache[oldest_key]
                logger_faiss.debug("内存缓存已满，删除最旧条目")

            return index

        except Exception as e:
            logger_faiss.error(f"构建FAISS索引失败: {e}")
            raise

    @classmethod
    def clear_cache(cls):
        """清空所有缓存"""
        logger_faiss.info("清空FAISS索引缓存")

        # 清空内存缓存
        cls._index_cache.clear()

        # 清空磁盘缓存
        try:
            for filename in os.listdir(FAISS_CACHE_DIR):
                if filename.endswith('.pkl'):
                    os.remove(os.path.join(FAISS_CACHE_DIR, filename))
            logger_faiss.info("磁盘缓存已清空")
        except Exception as e:
            logger_faiss.error(f"清空磁盘缓存失败: {e}")

    @classmethod
    def get_cache_stats(cls) -> Dict[str, any]:
        """
        获取缓存统计信息

        Returns:
            Dict[str, any]: 统计信息
        """
        try:
            disk_files = [f for f in os.listdir(FAISS_CACHE_DIR) if f.endswith('.pkl')]
            disk_size = sum(
                os.path.getsize(os.path.join(FAISS_CACHE_DIR, f))
                for f in disk_files
            )
        except Exception:
            disk_files = []
            disk_size = 0

        return {
            'memory_cache_count': len(cls._index_cache),
            'disk_cache_count': len(disk_files),
            'disk_cache_size_mb': round(disk_size / 1024 / 1024, 2),
            'cache_dir': FAISS_CACHE_DIR
        }

    @classmethod
    def preload_common_indices(cls):
        """
        预加载常用索引（可选，用于优化启动性能）
        """
        logger_faiss.info("开始预加载常用索引...")

        # 这里可以预定义一些常用的文档集合
        # 例如：常见的查询模板等
        # 实际实现需要根据业务需求调整

        logger_faiss.info("常用索引预加载完成")