"""
数据处理模块
作者：zjy
创建时间：2024年

该模块用于将原始文档数据转换为向量数据库格式，包含以下功能：
1. 读取多种格式的文档（CSV、PDF、TXT）
2. 文本分割和预处理
3. 向量化并存储到Chroma数据库

主要处理流程：
1. 遍历输入目录中的所有文档
2. 根据文件类型选择合适的加载器
3. 使用RecursiveCharacterTextSplitter进行文本分割
4. 使用嵌入模型进行向量化
5. 持久化到Chroma向量数据库
"""

import os
from glob import glob

from langchain_community.document_loaders import CSVLoader, PyMuPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .utils import get_embeddings_model
from .logger_config import logger_data


def doc2vec():
    """
    文档向量化主函数

    读取输入目录中的所有文档，进行分割和向量化，然后存储到向量数据库。

    处理的文件类型：
    - CSV文件：使用CSVLoader，编码为gb18030
    - PDF文件：使用PyMuPDFLoader
    - TXT文件：使用TextLoader，编码为gbk

    Raises:
        Exception: 当向量化过程失败时抛出异常
    """
    logger_data.info("开始文档向量化")

    try:
        # 1. 定义文本分割器
        # chunk_size: 每个文本块的最大字符数
        # chunk_overlap: 相邻文本块之间的重叠字符数
        # separators: 分割符优先级
        logger_data.debug("创建文本分割器")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80,
            separators=["\n\n", "。\n", "；\n", "，", " "]
        )
        logger_data.debug("分割参数: chunk_size=500, overlap=80")

        # 2. 读取并分割文件
        dir_path = os.path.join(os.path.dirname(__file__), './data/inputs/')
        logger_data.info(f"输入目录: {dir_path}")

        documents = []
        file_count = 0

        # 遍历输入目录中的所有文件
        for file_path in glob(dir_path + '*.*'):
            file_count += 1
            logger_data.debug(f"处理文件 {file_count}: {file_path}")

            loader = None
            file_ext = os.path.splitext(file_path)[1]
            logger_data.debug(f"文件扩展名: {file_ext}")

            # 根据文件类型选择加载器
            if '.csv' in file_path:
                loader = CSVLoader(file_path, encoding='gb18030')
                logger_data.debug("使用CSV加载器")
            if '.pdf' in file_path:
                loader = PyMuPDFLoader(file_path)
                logger_data.debug("使用PDF加载器")
            if '.txt' in file_path:
                loader = TextLoader(file_path, encoding='gbk')
                logger_data.debug("使用TXT加载器")

            # 如果找到了合适的加载器，进行文档加载和分割
            if loader:
                docs = loader.load_and_split(text_splitter)
                documents += docs
                logger_data.debug(f"文件 {file_path} 分割为 {len(docs)} 个文档片段")

        logger_data.info(f"文件处理完成，共 {file_count} 个文件，{len(documents)} 个文档片段")

        # 3. 向量化并存储
        if documents:
            logger_data.debug("开始向量化存储")
            vdb = Chroma.from_documents(
                documents=documents,
                embedding=get_embeddings_model(),
                persist_directory=os.path.join(os.path.dirname(__file__), './data/db/')
            )
            logger_data.info("向量化存储完成")
        else:
            logger_data.warning("没有找到可处理的文档")

    except Exception as e:
        logger_data.error(f"向量化失败: {e}")
        raise


if __name__ == '__main__':
    """
    数据处理入口点

    启动文档向量化流程，将原始文档转换为向量数据库格式。
    """
    logger_data.info("启动文档向量化程序")

    try:
        doc2vec()
        logger_data.info("文档向量化完成")
    except Exception as e:
        logger_data.error(f"文档向量化失败: {e}")
