from utils import *

import os
from glob import glob
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader, PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logger_config import logger_data


def doc2vec():
    logger_data.info("doc2vec() - 开始文档向量化")

    try:
        # 1. 定义文本分割器
        logger_data.debug("doc2vec() - 创建文本分割器")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80,
            separators=["\n\n", "。\n", "；\n", "，", " "]
        )
        logger_data.debug("doc2vec() - 分割参数: chunk_size=500, overlap=80")

        # 2. 读取并分割文件
        dir_path = os.path.join(os.path.dirname(__file__), './data/inputs/')
        logger_data.info(f"doc2vec() - 输入目录: {dir_path}")

        documents = []
        file_count = 0

        for file_path in glob(dir_path + '*.*'):
            file_count += 1
            logger_data.debug(f"doc2vec() - 处理文件 {file_count}: {file_path}")

            loader = None
            file_ext = os.path.splitext(file_path)[1]
            logger_data.debug(f"doc2vec() - 文件扩展名: {file_ext}")

            if '.csv' in file_path:
                loader = CSVLoader(file_path, encoding='gb18030')
                logger_data.debug("doc2vec() - 使用CSV加载器")
            if '.pdf' in file_path:
                loader = PyMuPDFLoader(file_path)
                logger_data.debug("doc2vec() - 使用PDF加载器")
            if '.txt' in file_path:
                loader = TextLoader(file_path, encoding='gbk')
                logger_data.debug("doc2vec() - 使用TXT加载器")

            if loader:
                docs = loader.load_and_split(text_splitter)
                documents += docs
                logger_data.debug(f"doc2vec() - 文件 {file_path} 分割为 {len(docs)} 个文档片段")

        logger_data.info(f"doc2vec() - 文件处理完成，共 {file_count} 个文件，{len(documents)} 个文档片段")

        # 3. 向量化并存储
        if documents:
            logger_data.debug("doc2vec() - 开始向量化存储")
            vdb = Chroma.from_documents(
                documents=documents,
                embedding=get_embeddings_model(),
                persist_directory=os.path.join(os.path.dirname(__file__), './data/db/')
            )
            logger_data.info("doc2vec() - 向量化存储完成")
        else:
            logger_data.warning("doc2vec() - 没有找到可处理的文档")

    except Exception as e:
        logger_data.error(f"doc2vec() - 向量化失败: {e}")
        raise



if __name__ == '__main__':
    logger_data.info("data_process.py - 启动文档向量化程序")

    try:
        doc2vec()
        logger_data.info("data_process.py - 文档向量化完成")
    except Exception as e:
        logger_data.error(f"data_process.py - 文档向量化失败: {e}")
