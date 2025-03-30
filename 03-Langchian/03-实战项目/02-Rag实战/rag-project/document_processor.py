"""
文档处理模块，负责加载、处理和分块文档
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union

from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from utils import get_file_extension, measure_time, generate_document_id

logger = logging.getLogger("document_processor")

class DocumentProcessor:
    """文档处理类，处理不同格式的文档并分块"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        supported_extensions: Optional[List[str]] = None
    ):
        """
        初始化文档处理器
        
        Args:
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
            supported_extensions: 支持的文件扩展名列表
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 设置默认支持的文件类型
        self.supported_extensions = supported_extensions or [
            "pdf", "txt", "csv", "docx", "html"
        ]
        
        # 创建文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        
        # 加载器映射
        self.loader_map = {
            "pdf": PyPDFLoader,
            "txt": TextLoader,
            "csv": CSVLoader,
            "docx": Docx2txtLoader,
            "html": UnstructuredHTMLLoader
        }
    
    @measure_time
    def load_single_document(self, file_path: str) -> List[Document]:
        """
        加载单个文档
        
        Args:
            file_path: 文档路径
            
        Returns:
            文档对象列表
        
        Raises:
            ValueError: 如果文件类型不支持
        """
        ext = get_file_extension(file_path)
        
        if ext not in self.supported_extensions:
            raise ValueError(f"不支持的文件类型: {ext}")
        
        if ext not in self.loader_map:
            raise ValueError(f"没有为扩展名 {ext} 配置加载器")
        
        loader_class = self.loader_map[ext]
        try:
            loader = loader_class(file_path)
            return loader.load()
        except Exception as e:
            logger.error(f"加载文件 {file_path} 时出错: {str(e)}")
            raise
    
    @measure_time
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        处理目录中的所有文档
        
        Args:
            directory_path: 目录路径
            
        Returns:
            处理后的文档块列表，包含文本和元数据
        """
        all_chunks = []
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                ext = get_file_extension(file_path)
                
                if ext not in self.supported_extensions:
                    continue
                
                try:
                    # 加载文档
                    docs = self.load_single_document(file_path)
                    
                    # 分块
                    chunks = self.text_splitter.split_documents(docs)
                    
                    # 添加额外元数据
                    for i, chunk in enumerate(chunks):
                        # 确保元数据是可修改的字典
                        if chunk.metadata is None:
                            chunk.metadata = {}
                        
                        # 添加文件路径、文件类型和块索引
                        chunk.metadata["source"] = file_path
                        chunk.metadata["file_type"] = ext
                        chunk.metadata["chunk_index"] = i
                        
                        # 为每个块生成唯一ID
                        chunk_id = generate_document_id(
                            chunk.page_content, 
                            f"{file_path}_{i}"
                        )
                        
                        # 添加到结果列表
                        all_chunks.append({
                            "id": chunk_id,
                            "text": chunk.page_content,
                            "metadata": chunk.metadata
                        })
                        
                except Exception as e:
                    logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
                    continue
        
        logger.info(f"成功处理 {len(all_chunks)} 个文档块")
        return all_chunks
    
    def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        处理文本字符串
        
        Args:
            text: 要处理的文本
            metadata: 可选的元数据
            
        Returns:
            处理后的文档块列表
        """
        # 创建Document对象
        doc = Document(page_content=text, metadata=metadata or {})
        
        # 分块
        chunks = self.text_splitter.split_documents([doc])
        
        # 转换为标准格式
        result = []
        for i, chunk in enumerate(chunks):
            # 确保元数据是可修改的字典
            if chunk.metadata is None:
                chunk.metadata = {}
            
            # 添加块索引
            chunk.metadata["chunk_index"] = i
            
            # 为每个块生成唯一ID
            source_info = metadata.get("source", "text_input") if metadata else "text_input"
            chunk_id = generate_document_id(
                chunk.page_content, 
                f"{source_info}_{i}"
            )
            
            # 添加到结果列表
            result.append({
                "id": chunk_id,
                "text": chunk.page_content,
                "metadata": chunk.metadata
            })
        
        return result 