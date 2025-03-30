# 嵌入模型

嵌入模型是RAG系统的核心组件之一，它将文本转换为密集的向量表示，使我们能够通过计算向量相似度来实现语义搜索。本章将深入探讨嵌入模型的工作原理、选择标准，以及在RAG系统中的实现方法。

## 什么是文本嵌入

文本嵌入(Text Embedding)是将文本转换为高维向量空间中的点的过程。这些向量能够捕捉文本的语义信息，使得语义相似的文本在向量空间中彼此接近。

![文本嵌入示意图](images/text_embedding.png)

### 嵌入的关键特性

1. **维度**：通常从几百到几千维不等，如OpenAI的text-embedding-3-small模型生成1536维向量
2. **语义捕捉**：能够捕捉文本的含义，而不仅仅是字面匹配
3. **相似度计算**：可以通过计算向量间的距离（如余弦相似度）来衡量文本相似性
4. **语言不可知**：许多现代嵌入模型支持多语言输入

## 常用嵌入模型

### 开源模型

1. **Sentence Transformers**
   - 流行模型：`all-MiniLM-L6-v2`、`all-mpnet-base-v2`、`multilingual-e5-large`
   - 优势：本地部署，无需API调用，多语言支持
   - 挑战：计算资源需求，质量可能不如商业模型

2. **HuggingFace Embeddings**
   - 流行模型：`BAAI/bge-large-zh`、`BAAI/bge-base-en`、`intfloat/multilingual-e5-large`
   - 优势：丰富的开源选择，专业化模型（如多语言、领域特定）
   - 挑战：模型大小和性能各异

### 商业模型

1. **OpenAI Embeddings**
   - 模型：`text-embedding-3-small`、`text-embedding-3-large`、`text-embedding-ada-002`(旧版)
   - 优势：高质量，容易使用，可扩展性好
   - 挑战：API成本，需要网络连接，依赖第三方服务

2. **Cohere Embeddings**
   - 模型：`embed-english-v3.0`、`embed-multilingual-v3.0`
   - 优势：多语言支持强，针对检索进行了优化
   - 挑战：与OpenAI类似的API限制

## 选择嵌入模型的考量因素

选择合适的嵌入模型需要考虑以下因素：

1. **性能**：检索准确度和相关性
2. **速度**：生成嵌入的速度和延迟
3. **成本**：API调用成本或计算资源需求
4. **维度**：向量维度影响存储需求和检索速度
5. **语言支持**：是否支持您需要的语言
6. **部署要求**：是否需要在线API或支持离线部署
7. **特定领域适应性**：在您的领域表现如何

## 实现嵌入功能

现在，我们将实现一个嵌入模块，支持多种嵌入模型：

```python
# src/embeddings/embedder.py

from typing import List, Dict, Any, Optional
import os
import numpy as np
from langchain.embeddings.base import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.embeddings import CohereEmbeddings

class EmbeddingFactory:
    """嵌入模型工厂类，用于创建不同的嵌入模型实例"""
    
    @staticmethod
    def create_embeddings(
        embeddings_type: str,
        model_name: Optional[str] = None,
        **kwargs
    ) -> Embeddings:
        """
        创建嵌入模型实例
        
        Args:
            embeddings_type: 嵌入模型类型，可选值包括"openai"、"huggingface"、"sentence_transformer"、"cohere"
            model_name: 模型名称，根据embeddings_type的不同而不同
            **kwargs: 传递给特定嵌入模型的其他参数
            
        Returns:
            嵌入模型实例
        """
        # OpenAI嵌入模型
        if embeddings_type.lower() == "openai":
            # 如果未指定模型名称，使用默认值
            if model_name is None:
                model_name = "text-embedding-3-small"
                
            # 获取API密钥
            api_key = kwargs.get("api_key", os.getenv("OPENAI_API_KEY"))
            if not api_key:
                raise ValueError("未提供OpenAI API密钥")
                
            return OpenAIEmbeddings(
                model=model_name,
                openai_api_key=api_key,
                **{k: v for k, v in kwargs.items() if k != "api_key"}
            )
        
        # Hugging Face嵌入模型
        elif embeddings_type.lower() == "huggingface":
            # 如果未指定模型名称，使用默认值
            if model_name is None:
                model_name = "BAAI/bge-large-zh"  # 中文默认模型
                
            return HuggingFaceEmbeddings(
                model_name=model_name,
                **kwargs
            )
        
        # Sentence Transformer嵌入模型
        elif embeddings_type.lower() == "sentence_transformer":
            # 如果未指定模型名称，使用默认值
            if model_name is None:
                model_name = "all-MiniLM-L6-v2"
                
            return SentenceTransformerEmbeddings(
                model_name=model_name,
                **kwargs
            )
        
        # Cohere嵌入模型
        elif embeddings_type.lower() == "cohere":
            # 如果未指定模型名称，使用默认值
            if model_name is None:
                model_name = "embed-multilingual-v3.0"
                
            # 获取API密钥
            api_key = kwargs.get("api_key", os.getenv("COHERE_API_KEY"))
            if not api_key:
                raise ValueError("未提供Cohere API密钥")
                
            return CohereEmbeddings(
                model=model_name,
                cohere_api_key=api_key,
                **{k: v for k, v in kwargs.items() if k != "api_key"}
            )
        
        else:
            raise ValueError(f"不支持的嵌入模型类型: {embeddings_type}")


class TextEmbedder:
    """文本嵌入器，用于将文本转换为向量表示"""
    
    def __init__(self, embedding_model: Embeddings):
        """
        初始化文本嵌入器
        
        Args:
            embedding_model: 嵌入模型实例
        """
        self.embedding_model = embedding_model
    
    def embed_text(self, text: str) -> List[float]:
        """
        将单个文本转换为向量
        
        Args:
            text: 输入文本
            
        Returns:
            向量表示
        """
        return self.embedding_model.embed_query(text)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        将多个文本转换为向量
        
        Args:
            texts: 输入文本列表
            
        Returns:
            向量表示列表
        """
        return self.embedding_model.embed_documents(texts)
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本之间的相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            相似度分数，范围为[0, 1]
        """
        # 获取两个文本的向量表示
        vec1 = self.embed_text(text1)
        vec2 = self.embed_text(text2)
        
        # 计算余弦相似度
        similarity = self._cosine_similarity(vec1, vec2)
        
        return similarity
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        计算两个向量之间的余弦相似度
        
        Args:
            vec1: 第一个向量
            vec2: 第二个向量
            
        Returns:
            余弦相似度，范围为[0, 1]
        """
        # 将列表转换为numpy数组
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        # 计算点积
        dot_product = np.dot(vec1_np, vec2_np)
        
        # 计算范数
        norm_vec1 = np.linalg.norm(vec1_np)
        norm_vec2 = np.linalg.norm(vec2_np)
        
        # 计算余弦相似度
        similarity = dot_product / (norm_vec1 * norm_vec2)
        
        return float(similarity)
```

## 使用嵌入模型

### 基本用法示例

```python
# 嵌入模型的基本用法示例

from src.embeddings.embedder import EmbeddingFactory, TextEmbedder

# 创建OpenAI嵌入模型（需要API密钥）
embedding_model = EmbeddingFactory.create_embeddings(
    embeddings_type="openai",
    model_name="text-embedding-3-small"
)

# 创建文本嵌入器
embedder = TextEmbedder(embedding_model)

# 嵌入单个文本
text = "检索增强生成是一种将大语言模型与外部知识库结合的技术"
embedding = embedder.embed_text(text)
print(f"向量维度: {len(embedding)}")
print(f"向量前5个元素: {embedding[:5]}")

# 嵌入多个文本
texts = [
    "RAG技术可以减少大语言模型的幻觉问题",
    "向量数据库用于存储和检索文本的向量表示",
    "文本分割是RAG系统中的重要步骤"
]
embeddings = embedder.embed_texts(texts)
print(f"生成了 {len(embeddings)} 个向量")

# 计算文本相似度
text1 = "RAG系统可以提高大语言模型的准确性"
text2 = "检索增强生成技术能够改善AI回答的精确度"
similarity = embedder.compute_similarity(text1, text2)
print(f"相似度: {similarity:.4f}")
```

### 使用开源模型

如果您希望使用本地部署的开源模型，可以使用以下代码：

```python
# 使用Sentence Transformers的示例

from src.embeddings.embedder import EmbeddingFactory, TextEmbedder

# 创建Sentence Transformer嵌入模型
embedding_model = EmbeddingFactory.create_embeddings(
    embeddings_type="sentence_transformer",
    model_name="all-mpnet-base-v2"
)

# 创建文本嵌入器
embedder = TextEmbedder(embedding_model)

# 嵌入文本并计算相似度
query = "如何实现有效的知识管理系统?"
documents = [
    "知识管理系统帮助企业组织和检索重要信息",
    "数据库设计是软件开发的关键步骤",
    "机器学习算法可以从数据中学习模式"
]

# 计算查询与每个文档的相似度
for doc in documents:
    similarity = embedder.compute_similarity(query, doc)
    print(f"查询: '{query}'")
    print(f"文档: '{doc}'")
    print(f"相似度: {similarity:.4f}")
    print("-" * 50)
```

## 批量处理文档嵌入

当需要为大量文档生成嵌入时，批量处理可以提高效率：

```python
# src/embeddings/batch_embedder.py

import os
import json
import pickle
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import numpy as np
from langchain.schema import Document

from src.embeddings.embedder import EmbeddingFactory, TextEmbedder

class BatchEmbedder:
    """批量文档嵌入处理器"""
    
    def __init__(
        self, 
        embedder: TextEmbedder,
        batch_size: int = 32,
        save_path: Optional[str] = None
    ):
        """
        初始化批量嵌入处理器
        
        Args:
            embedder: 文本嵌入器实例
            batch_size: 批处理大小
            save_path: 嵌入向量保存路径
        """
        self.embedder = embedder
        self.batch_size = batch_size
        self.save_path = save_path
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    def embed_documents(
        self, 
        documents: List[Document],
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        为文档列表生成嵌入向量
        
        Args:
            documents: 文档列表
            show_progress: 是否显示进度条
            
        Returns:
            包含嵌入结果的字典
        """
        # 提取文档内容
        texts = [doc.page_content for doc in documents]
        
        # 记录元数据
        metadatas = [doc.metadata for doc in documents]
        
        # 分批处理
        embeddings = []
        
        # 创建进度条
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="生成嵌入向量", unit="batch")
            
        # 批量处理
        for i in iterator:
            # 获取当前批次
            batch_texts = texts[i:i + self.batch_size]
            
            # 生成嵌入向量
            batch_embeddings = self.embedder.embed_texts(batch_texts)
            
            # 添加到结果列表
            embeddings.extend(batch_embeddings)
        
        # 创建结果字典
        result = {
            "texts": texts,
            "embeddings": embeddings,
            "metadatas": metadatas
        }
        
        # 保存结果（如果指定了保存路径）
        if self.save_path:
            self._save_embeddings(result)
            
        return result
    
    def _save_embeddings(self, result: Dict[str, Any]) -> None:
        """
        保存嵌入结果
        
        Args:
            result: 嵌入结果字典
        """
        # 确定文件扩展名
        _, ext = os.path.splitext(self.save_path)
        
        # 保存为pickle文件
        if ext.lower() == '.pkl':
            with open(self.save_path, 'wb') as f:
                pickle.dump(result, f)
                
        # 保存为JSON文件（将numpy数组转换为列表）
        elif ext.lower() == '.json':
            # 复制结果字典并转换numpy数组
            json_result = {
                "texts": result["texts"],
                "embeddings": [emb.tolist() if isinstance(emb, np.ndarray) else emb 
                               for emb in result["embeddings"]],
                "metadatas": result["metadatas"]
            }
            
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(json_result, f, ensure_ascii=False, indent=2)
                
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
        
        print(f"嵌入向量已保存到 {self.save_path}")
    
    @staticmethod
    def load_embeddings(file_path: str) -> Dict[str, Any]:
        """
        加载保存的嵌入向量
        
        Args:
            file_path: 文件路径
            
        Returns:
            嵌入结果字典
        """
        # 确定文件扩展名
        _, ext = os.path.splitext(file_path)
        
        # 加载pickle文件
        if ext.lower() == '.pkl':
            with open(file_path, 'rb') as f:
                return pickle.load(f)
                
        # 加载JSON文件
        elif ext.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
```

### 批量处理示例

```python
# 批量处理嵌入的示例

from src.embeddings.embedder import EmbeddingFactory, TextEmbedder
from src.embeddings.batch_embedder import BatchEmbedder
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

print(f"准备为 {len(split_docs)} 个文档块生成嵌入向量")

# 2. 创建嵌入模型
embedding_model = EmbeddingFactory.create_embeddings(
    embeddings_type="sentence_transformer",
    model_name="all-MiniLM-L6-v2"
)
embedder = TextEmbedder(embedding_model)

# 3. 创建批处理器并生成嵌入
batch_embedder = BatchEmbedder(
    embedder=embedder,
    batch_size=16,
    save_path="data/embeddings/document_embeddings.json"
)

# 4. 批量处理文档
embedding_results = batch_embedder.embed_documents(split_docs)

print(f"成功生成 {len(embedding_results['embeddings'])} 个嵌入向量")
print(f"每个向量的维度: {len(embedding_results['embeddings'][0])}")
```

## 嵌入模型评估

选择合适的嵌入模型对RAG系统的性能至关重要。以下是评估嵌入模型的一些方法：

```python
# 嵌入模型评估示例

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from src.embeddings.embedder import EmbeddingFactory, TextEmbedder

def evaluate_embedding_models(
    models_config: list,
    test_queries: list,
    test_docs: list,
    relevance_judgments: list
):
    """
    评估嵌入模型
    
    Args:
        models_config: 模型配置列表，每个配置包含'type'和'name'
        test_queries: 测试查询列表
        test_docs: 测试文档列表
        relevance_judgments: 相关性判断，格式为[(query_idx, doc_idx, relevance_score), ...]
    """
    results = {}
    
    for config in models_config:
        model_name = f"{config['type']}/{config['name']}"
        print(f"评估模型: {model_name}")
        
        # 创建嵌入模型
        embedding_model = EmbeddingFactory.create_embeddings(
            embeddings_type=config['type'],
            model_name=config['name']
        )
        embedder = TextEmbedder(embedding_model)
        
        # 测量嵌入速度
        start_time = time.time()
        query_embeddings = embedder.embed_texts(test_queries)
        doc_embeddings = embedder.embed_texts(test_docs)
        embed_time = time.time() - start_time
        
        # 计算查询-文档相似度矩阵
        similarities = []
        for q_emb in query_embeddings:
            q_similarities = []
            for d_emb in doc_embeddings:
                sim = np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb))
                q_similarities.append(float(sim))
            similarities.append(q_similarities)
        
        # 计算评估指标
        precision_scores = []
        recall_scores = []
        
        for query_idx in range(len(test_queries)):
            # 获取相关文档的真实索引
            relevant_docs = [doc_idx for q_idx, doc_idx, _ in relevance_judgments 
                             if q_idx == query_idx]
            
            # 获取模型预测的相关文档
            query_similarities = similarities[query_idx]
            top_doc_indices = np.argsort(query_similarities)[::-1][:len(relevant_docs)]
            
            # 计算准确率和召回率
            tp = len(set(top_doc_indices) & set(relevant_docs))
            precision = tp / len(top_doc_indices) if top_doc_indices else 0
            recall = tp / len(relevant_docs) if relevant_docs else 0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
        
        # 记录结果
        results[model_name] = {
            'avg_precision': np.mean(precision_scores),
            'avg_recall': np.mean(recall_scores),
            'embedding_time': embed_time,
            'vector_dim': len(query_embeddings[0])
        }
        
        print(f"  平均准确率: {results[model_name]['avg_precision']:.4f}")
        print(f"  平均召回率: {results[model_name]['avg_recall']:.4f}")
        print(f"  嵌入时间: {results[model_name]['embedding_time']:.2f}秒")
        print(f"  向量维度: {results[model_name]['vector_dim']}")
        print("-" * 50)
    
    return results
```

## 常见问题与解决方案

### 1. 嵌入计算成本高

**问题**：为大量文档生成嵌入向量需要大量计算资源或API调用。

**解决方案**：
- 使用批处理减少API调用次数
- 先过滤掉不相关文档，再生成嵌入
- 考虑使用更轻量级的本地嵌入模型
- 实现嵌入缓存，避免重复计算

### 2. 维度灾难

**问题**：高维向量可能会面临"维度灾难"，导致搜索效率降低。

**解决方案**：
- 使用降维技术如PCA或t-SNE
- 采用专为高维空间设计的索引结构（如HNSW、IVF）
- 选择合适的向量维度平衡精度和效率

### 3. 领域特定语言

**问题**：通用嵌入模型可能不能很好地处理专业领域术语。

**解决方案**：
- 使用领域特定的预训练嵌入模型
- 考虑微调嵌入模型以适应特定领域
- 为专业术语创建同义词扩展

## 下一步

在本章中，我们探讨了嵌入模型的工作原理，并实现了用于生成文本向量表示的组件。这些向量将用于构建向量数据库，使我们能够高效地检索相关信息。在下一章中，我们将学习如何使用这些嵌入向量创建和查询向量数据库。 