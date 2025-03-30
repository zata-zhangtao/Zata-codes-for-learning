"""
嵌入管理模块，负责生成和管理文本嵌入
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union

from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
import numpy as np

from utils import chunk_list, measure_time

logger = logging.getLogger("embedding_manager")

class EmbeddingManager:
    """嵌入管理类，处理文本嵌入的生成和管理"""
    
    def __init__(
        self,
        model_name: str = "openai",
        model_kwargs: Optional[Dict[str, Any]] = None,
        batch_size: int = 32
    ):
        """
        初始化嵌入管理器
        
        Args:
            model_name: 嵌入模型名称，可以是'openai'或HuggingFace模型名称
            model_kwargs: 传递给嵌入模型的参数
            batch_size: 批处理大小
        """
        self.model_name = model_name
        self.model_kwargs = model_kwargs or {}
        self.batch_size = batch_size
        
        # 初始化嵌入模型
        self._init_embedding_model()
    
    def _init_embedding_model(self):
        """初始化嵌入模型"""
        if self.model_name.lower() == "openai":
            self.embedding_model = OpenAIEmbeddings(**self.model_kwargs)
            self.embedding_dim = 1536  # OpenAI的ada-002嵌入维度
        else:
            # 使用HuggingFace模型
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs=self.model_kwargs
            )
            # 对于HuggingFace模型，我们需要查询一下嵌入维度
            self.embedding_dim = len(self.embedding_model.embed_query("test"))
        
        logger.info(f"初始化嵌入模型: {self.model_name}, 嵌入维度: {self.embedding_dim}")
    
    @measure_time
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        为多个文档生成嵌入
        
        Args:
            texts: 要嵌入的文本列表
            
        Returns:
            嵌入向量列表
        """
        logger.info(f"为 {len(texts)} 个文档生成嵌入")
        
        # 检查输入
        if not texts:
            return []
        
        # 按批次处理，避免可能的API限制
        all_embeddings = []
        batches = chunk_list(texts, self.batch_size)
        
        for i, batch in enumerate(batches):
            logger.debug(f"处理批次 {i+1}/{len(batches)}")
            try:
                batch_embeddings = self.embedding_model.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"嵌入批次 {i+1} 时出错: {str(e)}")
                # 为失败的批次填充零向量
                zero_vectors = [[0.0] * self.embedding_dim] * len(batch)
                all_embeddings.extend(zero_vectors)
        
        return all_embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """
        为查询文本生成嵌入
        
        Args:
            query: 查询文本
            
        Returns:
            嵌入向量
        """
        try:
            return self.embedding_model.embed_query(query)
        except Exception as e:
            logger.error(f"嵌入查询时出错: {str(e)}")
            return [0.0] * self.embedding_dim
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        计算两个嵌入向量之间的余弦相似度
        
        Args:
            embedding1: 第一个嵌入向量
            embedding2: 第二个嵌入向量
            
        Returns:
            余弦相似度 (0-1之间的值)
        """
        # 转换为numpy数组
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # 计算余弦相似度
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def batch_process_documents(
        self, 
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        批量处理文档，添加嵌入
        
        Args:
            documents: 文档列表，每个文档是一个字典，包含'text'字段
            
        Returns:
            添加了嵌入的文档列表
        """
        # 提取文本
        texts = [doc["text"] for doc in documents]
        
        # 生成嵌入
        embeddings = self.embed_documents(texts)
        
        # 将嵌入添加到文档中
        result = []
        for doc, embedding in zip(documents, embeddings):
            doc_copy = doc.copy()
            doc_copy["embedding"] = embedding
            result.append(doc_copy)
        
        return result 