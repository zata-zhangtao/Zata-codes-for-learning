"""
主RAG系统模块，集成所有组件提供完整功能
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional, Union

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

from document_processor import DocumentProcessor
from embedding_manager import EmbeddingManager
from vector_store import VectorStore
from utils import measure_time, truncate_text

# 加载环境变量
load_dotenv()

# 配置日志
logger = logging.getLogger("rag_system")

class RAGSystem:
    """主RAG系统类，整合所有组件"""
    
    def __init__(
        self,
        docs_dir: str = "documents",
        db_dir: str = "db",
        embedding_model: str = "openai",
        llm_model: str = "gpt-3.5-turbo",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        temperature: float = 0.1
    ):
        """
        初始化RAG系统
        
        Args:
            docs_dir: 文档目录
            db_dir: 数据库目录
            embedding_model: 嵌入模型名称
            llm_model: 语言模型名称
            chunk_size: 分块大小
            chunk_overlap: 分块重叠
            temperature: 模型温度参数
        """
        self.docs_dir = docs_dir
        self.db_dir = db_dir
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.temperature = temperature
        
        # 初始化组件
        self._init_components()
        
        logger.info("RAG系统初始化完成")
    
    def _init_components(self):
        """初始化系统组件"""
        # 初始化文档处理器
        self.document_processor = DocumentProcessor(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # 初始化嵌入管理器
        self.embedding_manager = EmbeddingManager(
            model_name=self.embedding_model_name
        )
        
        # 初始化向量存储
        self.vector_store = VectorStore(
            persist_directory=self.db_dir
        )
        
        # 初始化语言模型
        self.llm = ChatOpenAI(
            model_name=self.llm_model_name,
            temperature=self.temperature
        )
        
        # 创建标准提示模板
        self._create_prompt_templates()
    
    def _create_prompt_templates(self):
        """创建标准提示模板"""
        # 基本QA提示模板
        self.qa_prompt_template = ChatPromptTemplate.from_template("""
        你是一个知识丰富的助手。请根据提供的上下文信息回答用户的问题。
        只使用给定的上下文信息回答问题。如果上下文中没有相关信息，请直接说"我无法从提供的信息中找到答案"，不要编造信息。
        给出清晰、简洁且准确的回答。

        上下文信息:
        {context}

        用户问题:
        {query}

        你的回答:
        """)
    
    @measure_time
    def index_documents(self) -> Dict[str, Any]:
        """
        处理并索引文档目录中的所有文档
        
        Returns:
            索引结果统计
        """
        logger.info(f"开始处理并索引文档目录: {self.docs_dir}")
        
        # 检查目录是否存在
        if not os.path.exists(self.docs_dir):
            logger.error(f"文档目录不存在: {self.docs_dir}")
            return {"status": "error", "message": f"文档目录不存在: {self.docs_dir}"}
        
        # 1. 处理文档
        try:
            documents = self.document_processor.process_directory(self.docs_dir)
            logger.info(f"成功处理 {len(documents)} 个文档块")
        except Exception as e:
            logger.error(f"处理文档时出错: {str(e)}")
            return {"status": "error", "message": f"处理文档失败: {str(e)}"}
        
        if not documents:
            logger.warning("没有找到可处理的文档")
            return {"status": "warning", "message": "没有找到可处理的文档"}
        
        # 2. 生成嵌入
        try:
            documents_with_embeddings = self.embedding_manager.batch_process_documents(documents)
            logger.info(f"成功为 {len(documents_with_embeddings)} 个文档生成嵌入")
        except Exception as e:
            logger.error(f"生成嵌入时出错: {str(e)}")
            return {"status": "error", "message": f"生成嵌入失败: {str(e)}"}
        
        # 3. 添加到向量存储
        try:
            self.vector_store.add_documents(documents_with_embeddings)
            logger.info("成功将文档添加到向量存储")
        except Exception as e:
            logger.error(f"添加文档到向量存储时出错: {str(e)}")
            return {"status": "error", "message": f"添加到向量存储失败: {str(e)}"}
        
        # 4. 获取统计信息
        stats = self.vector_store.get_collection_stats()
        
        return {
            "status": "success",
            "message": f"成功索引 {len(documents)} 个文档块",
            "document_count": len(documents),
            "stats": stats
        }
    
    @measure_time
    def process_query(
        self, 
        query: str, 
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        处理用户查询
        
        Args:
            query: 用户查询
            top_k: 检索的文档数量
            filter_dict: 过滤条件
            return_sources: 是否返回来源信息
            
        Returns:
            包含回答和可选的来源的字典
        """
        start_time = time.time()
        
        logger.info(f"处理查询: {truncate_text(query)}")
        
        # 1. 生成查询嵌入
        query_embedding = self.embedding_manager.embed_query(query)
        
        # 2. 执行向量搜索
        search_results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_dict=filter_dict
        )
        
        if not search_results:
            return {
                "answer": "无法找到相关信息，请尝试重新表述您的问题。",
                "sources": [],
                "processing_time": time.time() - start_time
            }
        
        # 3. 准备上下文
        context = "\n\n".join([f"文档片段 {i+1}:\n{doc['text']}" for i, doc in enumerate(search_results)])
        
        # 4. 使用LLM生成回答
        prompt = self.qa_prompt_template.format(context=context, query=query)
        response = self.llm.invoke(prompt)
        
        # 5. 提取来源信息
        sources = []
        if return_sources:
            for doc in search_results:
                metadata = doc["metadata"]
                source_info = {
                    "text": truncate_text(doc["text"], 200),
                    "similarity": doc.get("similarity")
                }
                
                # 添加元数据中可能有用的字段
                source_path = metadata.get("source")
                if source_path:
                    source_info["source"] = source_path
                    # 提取文件名
                    source_info["filename"] = os.path.basename(source_path)
                
                if "page" in metadata:
                    source_info["page"] = metadata["page"]
                
                if "chunk_index" in metadata:
                    source_info["chunk_index"] = metadata["chunk_index"]
                
                sources.append(source_info)
        
        processing_time = time.time() - start_time
        
        return {
            "answer": response.content,
            "sources": sources,
            "processing_time": processing_time
        }
    
    def add_document(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        添加单个文本文档
        
        Args:
            text: 文档文本
            metadata: 文档元数据
            
        Returns:
            结果状态
        """
        if not text:
            return {"status": "error", "message": "文档内容为空"}
        
        # 1. 处理文本
        processed_chunks = self.document_processor.process_text(text, metadata)
        if not processed_chunks:
            return {"status": "error", "message": "文本处理失败"}
        
        # 2. 生成嵌入
        chunks_with_embeddings = self.embedding_manager.batch_process_documents(processed_chunks)
        
        # 3. 添加到向量存储
        self.vector_store.add_documents(chunks_with_embeddings)
        
        return {
            "status": "success", 
            "message": f"成功添加文档，分为 {len(processed_chunks)} 个块",
            "chunk_ids": [chunk["id"] for chunk in processed_chunks]
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取系统统计信息
        
        Returns:
            系统统计信息
        """
        # 获取向量存储统计
        vector_stats = self.vector_store.get_collection_stats()
        
        return {
            "system_info": {
                "docs_dir": self.docs_dir,
                "db_dir": self.db_dir,
                "embedding_model": self.embedding_model_name,
                "llm_model": self.llm_model_name,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap
            },
            "vector_store": vector_stats
        }
    
    def clear_data(self) -> Dict[str, Any]:
        """
        清空系统数据
        
        Returns:
            操作结果
        """
        try:
            self.vector_store.clear_collection()
            return {"status": "success", "message": "成功清空系统数据"}
        except Exception as e:
            logger.error(f"清空数据时出错: {str(e)}")
            return {"status": "error", "message": f"清空数据失败: {str(e)}"}
            
    def reload(self) -> Dict[str, Any]:
        """
        重新加载系统
        
        Returns:
            操作结果
        """
        try:
            # 重新初始化组件
            self._init_components()
            return {"status": "success", "message": "成功重新加载系统"}
        except Exception as e:
            logger.error(f"重新加载系统时出错: {str(e)}")
            return {"status": "error", "message": f"重新加载系统失败: {str(e)}"}
            
    def customize_prompt(self, template: str) -> None:
        """
        自定义提示模板
        
        Args:
            template: 提示模板文本，应包含{context}和{query}占位符
        """
        try:
            self.qa_prompt_template = ChatPromptTemplate.from_template(template)
            logger.info("成功自定义提示模板")
        except Exception as e:
            logger.error(f"自定义提示模板时出错: {str(e)}")
            raise 