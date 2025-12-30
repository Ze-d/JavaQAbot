"""
图数据库查询缓存管理器
用于缓存Cypher查询结果，减少Neo4j压力

作者：zjy
"""

import hashlib
import time
from typing import List, Dict, Any, Optional
from src.utils.logger_config import setup_logger

logger_cache = setup_logger('GraphCache', 'INFO')

class GraphQueryCache:
    """图查询缓存管理器"""

    _cache = {}
    _max_size = 1000  # 最大缓存条目数
    _ttl = 3600  # 缓存生存时间（秒）

    @classmethod
    def _generate_key(cls, cypher: str, params: Dict[str, Any] = None) -> str:
        """
        生成缓存键

        Args:
            cypher: Cypher查询语句
            params: 查询参数

        Returns:
            str: 缓存键
        """
        content = cypher + str(params or {})
        return hashlib.md5(content.encode()).hexdigest()

    @classmethod
    def _is_expired(cls, timestamp: float) -> bool:
        """
        检查缓存是否过期

        Args:
            timestamp: 缓存时间戳

        Returns:
            bool: 是否过期
        """
        return time.time() - timestamp > cls._ttl

    @classmethod
    def get(cls, cypher: str, params: Dict[str, Any] = None) -> Optional[List[Dict]]:
        """
        获取缓存的查询结果

        Args:
            cypher: Cypher查询语句
            params: 查询参数

        Returns:
            Optional[List[Dict]]: 缓存的结果，如果不存在或过期则返回None
        """
        key = cls._generate_key(cypher, params)

        if key in cls._cache:
            result, timestamp = cls._cache[key]
            if not cls._is_expired(timestamp):
                logger_cache.debug(f"缓存命中: {cypher[:50]}...")
                return result
            else:
                # 缓存过期，删除
                logger_cache.debug(f"缓存过期: {cypher[:50]}...")
                del cls._cache[key]

        return None

    @classmethod
    def set(cls, cypher: str, result: List[Dict], params: Dict[str, Any] = None):
        """
        设置缓存

        Args:
            cypher: Cypher查询语句
            result: 查询结果
            params: 查询参数
        """
        key = cls._generate_key(cypher, params)

        # 如果缓存已满，删除最旧的条目
        if len(cls._cache) >= cls._max_size:
            oldest_key = min(cls._cache.keys(), key=lambda k: cls._cache[k][1])
            del cls._cache[oldest_key]
            logger_cache.debug("缓存已满，删除最旧条目")

        cls._cache[key] = (result, time.time())
        logger_cache.debug(f"缓存设置: {cypher[:50]}... (共{len(result)}条结果)")

    @classmethod
    def clear(cls):
        """清空缓存"""
        logger_cache.info("清空图查询缓存")
        cls._cache.clear()

    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        total_entries = len(cls._cache)
        expired_entries = sum(1 for _, timestamp in cls._cache.values() if cls._is_expired(timestamp))

        return {
            'total_entries': total_entries,
            'expired_entries': expired_entries,
            'valid_entries': total_entries - expired_entries,
            'max_size': cls._max_size,
            'ttl': cls._ttl,
            'hit_rate': 'N/A'  # 简化实现，实际应记录命中次数
        }


def cached_cypher_query(neo4j_conn, cypher: str, params: Dict[str, Any] = None) -> List[Dict]:
    """
    带缓存的Cypher查询

    Args:
        neo4j_conn: Neo4j连接
        cypher: Cypher查询语句
        params: 查询参数

    Returns:
        List[Dict]: 查询结果
    """
    # 先尝试从缓存获取
    cached_result = GraphQueryCache.get(cypher, params)
    if cached_result is not None:
        return cached_result

    # 缓存未命中，执行实际查询
    try:
        if params:
            result = neo4j_conn.run(cypher, parameters=params).data()
        else:
            result = neo4j_conn.run(cypher).data()

        # 设置缓存
        GraphQueryCache.set(cypher, result, params)

        return result
    except Exception as e:
        logger_cache.error(f"Cypher查询失败: {e}")
        raise