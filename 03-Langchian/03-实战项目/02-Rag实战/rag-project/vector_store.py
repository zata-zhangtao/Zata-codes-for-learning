"""
向量存储模块，负责管理文档嵌入的存储和检索
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional, Union, Tuple

import chromadb
from chromadb.config import Settings
import numpy as np

from utils import measure_time, safe_get

logger = logging.getLogger("vector_store")

class VectorStore:
    """向量存储类，管理文档嵌入的存储和检索"""
    
    def __init__(
        self,
        persist_directory: str = "db",
        collection_name: str = "documents",
        distance_func: str = "cosine"
    ):
        """
        初始化向量存储
        
        Args:
            persist_directory: 持久化目录
            collection_name: 集合名称
            distance_func: 距离函数，可以是'cosine', 'l2', 'ip'
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.distance_func = distance_func
        
        # 确保持久化目录存在
        os.makedirs(persist_directory, exist_ok=True)
        
        # 初始化ChromaDB客户端
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 获取或创建集合
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=None  # 我们将手动处理嵌入
            )
            logger.info(f"加载现有集合: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=None,  # 我们将手动处理嵌入
                metadata={"hnsw:space": distance_func}
            )
            logger.info(f"创建新集合: {collection_name}")
    
    @measure_time
    def add_documents(
        self, 
        documents: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> None:
        """
        添加文档到向量存储
        
        Args:
            documents: 文档列表，每个文档包含'id', 'text', 'embedding'和'metadata'
            batch_size: 批处理大小
        """
        if not documents:
            logger.warning("没有文档可添加")
            return
        
        total_docs = len(documents)
        logger.info(f"添加 {total_docs} 个文档到向量存储")
        
        # 按批次处理
        for i in range(0, total_docs, batch_size):
            end_idx = min(i + batch_size, total_docs)
            batch = documents[i:end_idx]
            
            # 提取数据
            ids = [doc["id"] for doc in batch]
            embeddings = [doc["embedding"] for doc in batch]
            metadatas = [doc.get("metadata", {}) for doc in batch]
            texts = [doc["text"] for doc in batch]
            
            try:
                # 将文档添加到集合
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=texts
                )
                logger.debug(f"成功添加批次 {i//batch_size + 1}/{(total_docs-1)//batch_size + 1}")
            except Exception as e:
                logger.error(f"添加批次 {i//batch_size + 1} 时出错: {str(e)}")
    
    @measure_time
    def similarity_search(
        self, 
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        执行相似度搜索
        
        Args:
            query_embedding: 查询的嵌入向量
            top_k: 返回的最相似文档数量
            filter_dict: 过滤条件
            
        Returns:
            最相似的文档列表
        """
        try:
            # 执行查询
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_dict
            )
            
            # 处理结果
            documents = []
            for i in range(len(results["ids"][0])):
                doc_id = results["ids"][0][i]
                text = results["documents"][0][i]
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i] if "distances" in results else None
                
                # 计算相似度分数
                if self.distance_func == "cosine" and distance is not None:
                    # 余弦距离转换为相似度
                    similarity = 1 - distance
                elif self.distance_func == "ip" and distance is not None:
                    # 内积已经是相似度
                    similarity = distance
                else:
                    # 如果是L2距离或没有距离，设为None
                    similarity = None
                
                documents.append({
                    "id": doc_id,
                    "text": text,
                    "metadata": metadata,
                    "similarity": similarity
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"执行相似度搜索时出错: {str(e)}")
            return []
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        通过ID获取文档
        
        Args:
            doc_id: 文档ID
            
        Returns:
            文档数据或None
        """
        try:
            result = self.collection.get(ids=[doc_id])
            
            if result["ids"] and len(result["ids"]) > 0:
                return {
                    "id": result["ids"][0],
                    "text": result["documents"][0],
                    "metadata": result["metadatas"][0]
                }
            return None
        except Exception as e:
            logger.error(f"获取文档 {doc_id} 时出错: {str(e)}")
            return None
    
    @measure_time
    def delete_documents(self, doc_ids: List[str]) -> None:
        """
        删除文档
        
        Args:
            doc_ids: 要删除的文档ID列表
        """
        try:
            self.collection.delete(ids=doc_ids)
            logger.info(f"成功删除 {len(doc_ids)} 个文档")
        except Exception as e:
            logger.error(f"删除文档时出错: {str(e)}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Returns:
            集合统计信息
        """
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"获取集合统计信息时出错: {str(e)}")
            return {
                "collection_name": self.collection_name,
                "error": str(e)
            }
    
    def clear_collection(self) -> None:
        """清空集合"""
        try:
            self.collection.delete(where={})
            logger.info(f"已清空集合 {self.collection_name}")
        except Exception as e:
            logger.error(f"清空集合时出错: {str(e)}")
            
    def save_to_disk(self) -> None:
        """将数据保存到磁盘"""
        # ChromaDB已经自动保存，这里只是一个确认方法
        logger.info(f"集合 {self.collection_name} 已保存到 {self.persist_directory}")
        
    def load_from_disk(self) -> None:
        """从磁盘加载数据"""
        # 重新连接客户端和集合
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"已从 {self.persist_directory} 加载集合 {self.collection_name}")
        except Exception as e:
            logger.error(f"从磁盘加载向量存储时出错: {str(e)}")
            raise 