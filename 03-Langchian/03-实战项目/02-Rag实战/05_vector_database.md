# 向量数据库

向量数据库是RAG系统的核心存储组件，专门用于高效存储和检索向量数据。本章将介绍向量数据库的基本概念、常用选项以及如何在RAG系统中实现向量存储和检索功能。

## 什么是向量数据库

向量数据库是专为存储和检索高维向量而设计的数据库，它使用特殊的索引结构和算法，能够快速执行相似度搜索（如K最近邻搜索）。

### 传统数据库与向量数据库的区别

| 特性 | 传统数据库 | 向量数据库 |
|------|------------|------------|
| 主要数据类型 | 结构化数据（表格、键值） | 高维向量 |
| 查询方式 | 精确匹配（=, <, >, LIKE等） | 相似度搜索（余弦相似度、欧氏距离等） |
| 索引结构 | B树、哈希表等 | 树形结构（如KD树）、图结构（如HNSW）、量化方法（如PQ） |
| 优化目标 | 精确检索速度 | 近似最近邻搜索效率 |

## 常用向量数据库

### 开源选项

1. **Chroma**
   - 特点：轻量级，易于集成，Python原生支持
   - 适用场景：开发阶段、小型项目
   - 局限性：大规模数据的性能挑战

2. **FAISS (Facebook AI Similarity Search)**
   - 特点：高性能，支持GPU加速，丰富的索引类型
   - 适用场景：大规模向量检索，注重性能
   - 局限性：使用相对复杂，元数据管理有限

3. **Milvus**
   - 特点：分布式架构，高可扩展性，企业级
   - 适用场景：生产环境，大规模应用
   - 局限性：部署和运维相对复杂

4. **Qdrant**
   - 特点：强大的过滤能力，支持实时更新
   - 适用场景：需要复杂查询条件的应用
   - 局限性：相比其他选项较新

### 商业选项

1. **Pinecone**
   - 特点：完全托管，易于使用，高可扩展性
   - 适用场景：快速部署，无需维护
   - 局限性：成本，数据完全托管在第三方

2. **Weaviate**
   - 特点：结合向量搜索和图数据库概念
   - 适用场景：需要复杂知识图谱的应用
   - 局限性：学习曲线，特定用例的优化

## 选择向量数据库的考量因素

1. **规模需求**：预期的向量数量和维度
2. **查询性能**：检索速度与结果准确性的平衡
3. **更新频率**：是否需要频繁添加或修改向量
4. **元数据管理**：是否需要与向量一起存储和检索元数据
5. **部署模式**：本地部署、云服务或混合模式
6. **预算**：开源免费方案与付费商业选项
7. **集成难度**：与现有系统的兼容性

## 向量索引算法

向量数据库使用不同的索引算法来加速相似度搜索：

1. **精确搜索**：线性扫描所有向量（暴力搜索）
2. **树形索引**：如KD树、球树等
3. **基于图的索引**：如HNSW（分层可导航小世界图）
4. **量化索引**：如Product Quantization (PQ)，通过压缩向量减少内存使用
5. **混合索引**：如IVF（倒排文件）+ HNSW等

## 实现向量存储

下面我们将实现与不同向量数据库交互的接口：

```python
# src/vector_store/store.py

from typing import List, Dict, Any, Optional, Union, Tuple
import os
import numpy as np
from langchain.schema import Document
from langchain.vectorstores import Chroma, FAISS
from langchain.vectorstores.base import VectorStore
from langchain.embeddings.base import Embeddings

class VectorStoreFactory:
    """向量存储工厂类，用于创建不同的向量数据库实例"""
    
    @staticmethod
    def create_vector_store(
        store_type: str,
        embeddings: Embeddings,
        documents: Optional[List[Document]] = None,
        texts: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        persist_directory: Optional[str] = None,
        **kwargs
    ) -> VectorStore:
        """
        创建向量存储实例
        
        Args:
            store_type: 向量存储类型，可选值包括"chroma"、"faiss"
            embeddings: 嵌入模型
            documents: 文档列表（可选）
            texts: 文本列表（可选，与documents二选一）
            metadatas: 元数据列表（可选，与texts配合使用）
            persist_directory: 持久化目录（可选）
            **kwargs: 其他参数
            
        Returns:
            向量存储实例
        """
        # 确保指定了documents或texts中的一个
        if documents is None and texts is None:
            raise ValueError("必须指定documents或texts")
            
        # 如果指定了persist_directory，确保目录存在
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            
        # Chroma向量数据库
        if store_type.lower() == "chroma":
            if documents:
                return Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    persist_directory=persist_directory,
                    **kwargs
                )
            else:
                return Chroma.from_texts(
                    texts=texts,
                    embedding=embeddings,
                    metadatas=metadatas,
                    persist_directory=persist_directory,
                    **kwargs
                )
                
        # FAISS向量数据库
        elif store_type.lower() == "faiss":
            if documents:
                return FAISS.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    **kwargs
                )
            else:
                return FAISS.from_texts(
                    texts=texts,
                    embedding=embeddings,
                    metadatas=metadatas,
                    **kwargs
                )
                
        else:
            raise ValueError(f"不支持的向量存储类型: {store_type}")
            
    @staticmethod
    def load_vector_store(
        store_type: str,
        embeddings: Embeddings,
        persist_directory: Optional[str] = None,
        index_path: Optional[str] = None,
        **kwargs
    ) -> VectorStore:
        """
        加载已有的向量存储
        
        Args:
            store_type: 向量存储类型
            embeddings: 嵌入模型
            persist_directory: 持久化目录（Chroma使用）
            index_path: 索引文件路径（FAISS使用）
            **kwargs: 其他参数
            
        Returns:
            向量存储实例
        """
        # 加载Chroma向量数据库
        if store_type.lower() == "chroma":
            if not persist_directory:
                raise ValueError("加载Chroma需要指定persist_directory")
                
            return Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
                **kwargs
            )
            
        # 加载FAISS向量数据库
        elif store_type.lower() == "faiss":
            if not index_path:
                raise ValueError("加载FAISS需要指定index_path")
                
            return FAISS.load_local(
                folder_path=os.path.dirname(index_path),
                embeddings=embeddings,
                index_name=os.path.basename(index_path),
                **kwargs
            )
            
        else:
            raise ValueError(f"不支持的向量存储类型: {store_type}")


class VectorStoreManager:
    """向量存储管理器，提供更高级的向量存储操作"""
    
    def __init__(self, vector_store: VectorStore):
        """
        初始化向量存储管理器
        
        Args:
            vector_store: 向量存储实例
        """
        self.vector_store = vector_store
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        添加文档到向量存储
        
        Args:
            documents: 要添加的文档列表
        """
        self.vector_store.add_documents(documents)
        
        # 如果向量存储支持持久化，则保存
        if hasattr(self.vector_store, "persist"):
            self.vector_store.persist()
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        添加文本到向量存储
        
        Args:
            texts: 要添加的文本列表
            metadatas: 对应的元数据列表
        """
        self.vector_store.add_texts(texts, metadatas=metadatas)
        
        # 如果向量存储支持持久化，则保存
        if hasattr(self.vector_store, "persist"):
            self.vector_store.persist()
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        执行相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filter: 过滤条件
            
        Returns:
            相似文档列表
        """
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        执行相似度搜索并返回得分
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filter: 过滤条件
            
        Returns:
            (文档, 得分)元组列表
        """
        return self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )
    
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        执行最大边际相关性搜索（增加结果多样性）
        
        Args:
            query: 查询文本
            k: 返回结果数量
            fetch_k: 初始获取的结果数量
            lambda_mult: 多样性权重
            filter: 过滤条件
            
        Returns:
            文档列表
        """
        return self.vector_store.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter
        )
```

## 使用向量数据库

### 创建和存储示例

```python
# 创建向量数据库示例

from src.vector_store.store import VectorStoreFactory, VectorStoreManager
from src.embeddings.embedder import EmbeddingFactory, TextEmbedder
from src.data.loader import DocumentLoader
from src.data.processor import TextProcessor
from src.data.splitter import TextSplitter

# 1. 准备文档
loader = DocumentLoader()
processor = TextProcessor()
splitter = TextSplitter(chunk_size=500, chunk_overlap=50)

# 加载和处理文档
raw_docs = loader.load_documents_from_directory("data/raw")
processed_docs = processor.process_documents(raw_docs)
split_docs = splitter.split_documents(processed_docs)

print(f"总共准备了 {len(split_docs)} 个文档块")

# 2. 创建嵌入模型
embedding_model = EmbeddingFactory.create_embeddings(
    embeddings_type="sentence_transformer",
    model_name="all-MiniLM-L6-v2"
)

# 3. 创建向量数据库
vector_store = VectorStoreFactory.create_vector_store(
    store_type="chroma",
    embeddings=embedding_model,
    documents=split_docs,
    persist_directory="data/embeddings/chroma_db"
)

# 4. 创建向量存储管理器
store_manager = VectorStoreManager(vector_store)

print("向量数据库已创建并持久化")
```

### 查询示例

```python
# 向量数据库查询示例

from src.vector_store.store import VectorStoreFactory, VectorStoreManager
from src.embeddings.embedder import EmbeddingFactory, TextEmbedder

# 1. 创建嵌入模型
embedding_model = EmbeddingFactory.create_embeddings(
    embeddings_type="sentence_transformer",
    model_name="all-MiniLM-L6-v2"
)

# 2. 加载已有向量数据库
vector_store = VectorStoreFactory.load_vector_store(
    store_type="chroma",
    embeddings=embedding_model,
    persist_directory="data/embeddings/chroma_db"
)

# 3. 创建向量存储管理器
store_manager = VectorStoreManager(vector_store)

# 4. 执行相似度搜索
query = "RAG系统如何减少大语言模型的幻觉问题?"
results = store_manager.similarity_search(query, k=3)

print(f"查询: {query}")
print("\n相似文档:")
for i, doc in enumerate(results):
    print(f"\n文档 {i+1}:")
    print(f"内容: {doc.page_content[:150]}...")
    print(f"元数据: {doc.metadata}")

# 5. 执行带分数的相似度搜索
scored_results = store_manager.similarity_search_with_score(query, k=3)

print("\n\n带分数的相似文档:")
for i, (doc, score) in enumerate(scored_results):
    print(f"\n文档 {i+1} (得分: {score:.4f}):")
    print(f"内容: {doc.page_content[:150]}...")
```

### 使用过滤条件

向量数据库支持基于元数据进行过滤，这在RAG应用中非常有用：

```python
# 使用过滤条件的示例

# 按文档来源过滤
filter_by_source = {"source": "company_handbook.pdf"}
filtered_results = store_manager.similarity_search(
    query="公司政策", 
    k=5,
    filter=filter_by_source
)

# 按日期范围过滤
filter_by_date = {
    "date": {"$gte": "2023-01-01", "$lte": "2023-12-31"}
}
recent_results = store_manager.similarity_search(
    query="最新财务报告", 
    k=5,
    filter=filter_by_date
)

# 组合过滤条件
combined_filter = {
    "department": "HR",
    "confidentiality": {"$in": ["public", "internal"]}
}
hr_results = store_manager.similarity_search(
    query="员工手册", 
    k=5,
    filter=combined_filter
)
```

## FAISS向量数据库示例

FAISS是一种高性能向量数据库，特别适合大规模向量集合：

```python
# FAISS向量数据库示例

from src.vector_store.store import VectorStoreFactory, VectorStoreManager
from src.embeddings.embedder import EmbeddingFactory, TextEmbedder

# 创建嵌入模型
embedding_model = EmbeddingFactory.create_embeddings(
    embeddings_type="sentence_transformer",
    model_name="all-MiniLM-L6-v2"
)

# 准备示例文本和元数据
texts = [
    "RAG技术可以有效减少大语言模型的幻觉问题",
    "向量数据库是RAG系统的核心组件之一",
    "文本分割对检索效果有重要影响",
    "嵌入模型的选择会影响语义检索的准确性",
    "LLM与检索系统的结合使AI回答更加准确"
]

metadatas = [
    {"source": "article1.pdf", "page": 1, "topic": "RAG"},
    {"source": "article2.pdf", "page": 3, "topic": "vector_db"},
    {"source": "article3.pdf", "page": 2, "topic": "preprocessing"},
    {"source": "article4.pdf", "page": 5, "topic": "embeddings"},
    {"source": "article5.pdf", "page": 7, "topic": "LLM"}
]

# 创建FAISS向量数据库
faiss_store = VectorStoreFactory.create_vector_store(
    store_type="faiss",
    embeddings=embedding_model,
    texts=texts,
    metadatas=metadatas
)

# 保存FAISS索引
import os
save_dir = "data/embeddings/faiss_db"
os.makedirs(save_dir, exist_ok=True)
faiss_store.save_local(save_dir)

# 创建管理器并搜索
faiss_manager = VectorStoreManager(faiss_store)
query = "如何提高检索准确性?"
results = faiss_manager.similarity_search(query, k=2)

for i, doc in enumerate(results):
    print(f"\n结果 {i+1}:")
    print(f"文本: {doc.page_content}")
    print(f"元数据: {doc.metadata}")
```

## 实现混合检索策略

有时单纯的向量相似度搜索可能不足以满足复杂检索需求，混合检索策略可以提高整体检索质量：

```python
# src/vector_store/hybrid_retriever.py

from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore
from langchain.retrievers.base import BaseRetriever

class HybridRetriever(BaseRetriever):
    """混合检索器，结合关键词搜索和向量搜索"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        keyword_index: Dict[str, List[int]],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        k: int = 4
    ):
        """
        初始化混合检索器
        
        Args:
            vector_store: 向量存储
            keyword_index: 关键词索引 {词: [文档索引列表]}
            texts: 原始文本列表
            metadatas: 元数据列表
            k: 返回结果数量
        """
        self.vector_store = vector_store
        self.keyword_index = keyword_index
        self.texts = texts
        self.metadatas = metadatas
        self.k = k
        super().__init__()
    
    def _get_relevant_documents(
        self,
        query: str,
        keyword_weight: float = 0.3,
        run_manager=None,
    ) -> List[Document]:
        """
        获取相关文档
        
        Args:
            query: 查询文本
            keyword_weight: 关键词搜索的权重
            
        Returns:
            相关文档列表
        """
        # 1. 向量搜索
        vector_results = self.vector_store.similarity_search_with_score(query, k=self.k*2)
        
        # 2. 关键词搜索
        keywords = [word.lower() for word in query.split() if len(word) > 3]
        keyword_doc_indices = set()
        for keyword in keywords:
            if keyword in self.keyword_index:
                keyword_doc_indices.update(self.keyword_index[keyword])
        
        keyword_docs = []
        for idx in keyword_doc_indices:
            if idx < len(self.texts):
                doc = Document(
                    page_content=self.texts[idx],
                    metadata=self.metadatas[idx]
                )
                keyword_docs.append((doc, 1.0))  # 分配一个统一的分数
        
        # 3. 合并结果
        # 创建文档ID到(文档,分数)的映射
        results_map = {}
        
        # 添加向量搜索结果
        for doc, score in vector_results:
            doc_id = doc.metadata.get("chunk_id", hash(doc.page_content))
            results_map[doc_id] = (doc, (1 - keyword_weight) * float(score))
        
        # 添加关键词搜索结果
        for doc, score in keyword_docs:
            doc_id = doc.metadata.get("chunk_id", hash(doc.page_content))
            if doc_id in results_map:
                # 如果文档已存在，增加得分
                existing_doc, existing_score = results_map[doc_id]
                results_map[doc_id] = (existing_doc, existing_score + keyword_weight * score)
            else:
                # 如果文档不存在，添加新文档
                results_map[doc_id] = (doc, keyword_weight * score)
        
        # 4. 排序并返回前k个结果
        sorted_results = sorted(
            results_map.values(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [doc for doc, _ in sorted_results[:self.k]]
```

## 向量数据库性能优化

当向量数据库规模增长时，性能优化变得至关重要：

### 索引参数调整

```python
# Chroma使用HNSW索引的优化示例
optimized_chroma = VectorStoreFactory.create_vector_store(
    store_type="chroma",
    embeddings=embedding_model,
    documents=documents,
    persist_directory="data/embeddings/optimized_chroma",
    collection_metadata={"hnsw:space": "cosine", "hnsw:M": 16, "hnsw:efConstruction": 200}
)

# FAISS索引优化
import faiss
dimension = 384  # 与嵌入模型匹配

# 创建IVF索引（适合大规模数据集）
nlist = 100  # 聚类数量
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

# 需要训练
index.train(vectors)  # vectors是numpy数组形式的嵌入向量
index.add(vectors)

# 控制搜索时间和精度的平衡
index.nprobe = 10  # 搜索时检查的聚类数量
```

### 批处理和分片

对于大型数据集，批处理和分片是必要的：

```python
# 批量处理大型文档集合
batch_size = 1000
for i in range(0, len(all_documents), batch_size):
    batch = all_documents[i:i+batch_size]
    store_manager.add_documents(batch)
    print(f"已处理 {i+len(batch)}/{len(all_documents)} 个文档")
```

## 下一步

在本章中，我们学习了向量数据库的基本概念，并实现了创建、存储和查询向量数据的功能。这些组件使我们能够高效检索与用户查询语义相关的文档。在下一章中，我们将学习如何构建一个基础RAG系统，将向量检索与大语言模型集成起来，形成一个完整的检索增强生成系统。 