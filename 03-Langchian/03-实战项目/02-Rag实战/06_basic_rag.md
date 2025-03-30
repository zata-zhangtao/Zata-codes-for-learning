# 基础RAG实现

在前几章中，我们探讨了RAG系统的各个组件，包括数据处理、嵌入模型和向量数据库。现在，我们将把这些组件整合起来，构建一个完整的基础RAG系统。

## RAG架构概述

一个基本的RAG系统包含以下关键步骤:

1. **查询处理**：处理用户输入的查询
2. **检索**：从向量数据库中检索相关文档
3. **上下文合成**：将检索到的文档与原始查询组合
4. **生成**：使用大语言模型生成最终回答

![基础RAG架构](images/basic_rag_architecture.png)

## 实现基础RAG系统

让我们从一个简单但功能完整的RAG实现开始：

```python
# src/rag/basic_rag.py

from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.base import BaseRetriever
from langchain.llms.base import BaseLLM

class BasicRAG:
    """基础RAG系统实现"""
    
    def __init__(
        self, 
        retriever: BaseRetriever,
        llm: BaseLLM,
        prompt_template: Optional[str] = None
    ):
        """
        初始化基础RAG系统
        
        Args:
            retriever: 检索器实例
            llm: 大语言模型实例
            prompt_template: 提示模板字符串（可选）
        """
        self.retriever = retriever
        self.llm = llm
        
        # 如果未指定提示模板，使用默认模板
        if prompt_template is None:
            prompt_template = """使用以下上下文来回答问题。如果你不知道答案，就说你不知道，不要试图编造答案。
            
上下文:
{context}

问题: {question}

回答:"""
        
        self.prompt = PromptTemplate.from_template(prompt_template)
        
        # 构建RAG链
        self.rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | self.prompt
            | llm
            | StrOutputParser()
        )
    
    def query(self, question: str) -> str:
        """
        处理用户查询
        
        Args:
            question: 用户问题
            
        Returns:
            回答文本
        """
        return self.rag_chain.invoke(question)
    
    def get_retrieved_documents(self, question: str) -> List[Document]:
        """
        获取检索到的文档（用于调试和解释）
        
        Args:
            question: 用户问题
            
        Returns:
            检索到的文档列表
        """
        return self.retriever.get_relevant_documents(question)
    
    def query_with_sources(self, question: str) -> Dict[str, Any]:
        """
        处理用户查询并返回来源信息
        
        Args:
            question: 用户问题
            
        Returns:
            包含回答和来源的字典
        """
        # 获取检索到的文档
        docs = self.get_retrieved_documents(question)
        
        # 生成回答
        answer = self.query(question)
        
        # 提取来源信息
        sources = []
        seen_sources = set()
        
        for doc in docs:
            source = doc.metadata.get("source", "未知来源")
            if source not in seen_sources:
                sources.append({
                    "source": source,
                    "page": doc.metadata.get("page", "未知页码"),
                    "content_preview": doc.page_content[:100] + "..."
                })
                seen_sources.add(source)
        
        return {
            "answer": answer,
            "sources": sources
        }
```

## 创建检索器

检索器是RAG系统的关键组件，负责从向量数据库中检索相关文档：

```python
# src/retriever/retriever.py

from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.retrievers.base import BaseRetriever
from langchain.vectorstores.base import VectorStore

class VectorStoreRetriever(BaseRetriever):
    """向量存储检索器"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        初始化向量存储检索器
        
        Args:
            vector_store: 向量存储实例
            search_type: 搜索类型，可选值包括"similarity"、"mmr"
            search_kwargs: 搜索参数
        """
        self.vector_store = vector_store
        self.search_type = search_type
        self.search_kwargs = search_kwargs or {"k": 4}
        super().__init__()
    
    def _get_relevant_documents(
        self,
        query: str,
        run_manager=None,
    ) -> List[Document]:
        """
        获取相关文档
        
        Args:
            query: 查询文本
            
        Returns:
            相关文档列表
        """
        if self.search_type == "similarity":
            return self.vector_store.similarity_search(
                query=query,
                **self.search_kwargs
            )
        elif self.search_type == "mmr":
            return self.vector_store.max_marginal_relevance_search(
                query=query,
                **self.search_kwargs
            )
        else:
            raise ValueError(f"不支持的搜索类型: {self.search_type}")
    
    def get_top_k_docs(self, query: str, k: int) -> List[Document]:
        """
        获取前k个相关文档
        
        Args:
            query: 查询文本
            k: 文档数量
            
        Returns:
            相关文档列表
        """
        # 保存原始k值
        original_k = self.search_kwargs.get("k")
        
        # 更新k值
        self.search_kwargs["k"] = k
        
        # 获取文档
        docs = self._get_relevant_documents(query)
        
        # 恢复原始k值
        if original_k is not None:
            self.search_kwargs["k"] = original_k
        
        return docs
```

## 配置LLM模型接口

要完成RAG系统，我们需要与大语言模型集成：

```python
# src/llm/model.py

from typing import Any, List, Dict, Optional
import os
from langchain.llms.base import BaseLLM
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline

class LLMFactory:
    """大语言模型工厂类，用于创建不同的LLM实例"""
    
    @staticmethod
    def create_llm(
        llm_type: str,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs
    ) -> BaseLLM:
        """
        创建LLM实例
        
        Args:
            llm_type: LLM类型，可选值包括"openai"、"huggingface"
            model_name: 模型名称
            temperature: 温度参数
            **kwargs: 其他参数
            
        Returns:
            LLM实例
        """
        # OpenAI模型
        if llm_type.lower() == "openai":
            # 如果未指定模型名称，使用默认值
            if model_name is None:
                model_name = "gpt-3.5-turbo"
                
            # 获取API密钥
            api_key = kwargs.get("api_key", os.getenv("OPENAI_API_KEY"))
            if not api_key:
                raise ValueError("未提供OpenAI API密钥")
                
            return ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                openai_api_key=api_key,
                **{k: v for k, v in kwargs.items() if k != "api_key"}
            )
        
        # HuggingFace模型
        elif llm_type.lower() == "huggingface":
            # 需要安装transformers和torch
            try:
                import torch
                from transformers import pipeline
            except ImportError:
                raise ImportError("使用HuggingFace需要安装transformers和torch库")
            
            # 如果未指定模型名称，使用默认值
            if model_name is None:
                model_name = "THUDM/chatglm3-6b"  # 中文模型示例
            
            # 创建HuggingFace管道
            pipe = pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                **{k: v for k, v in kwargs.items() if k not in ["api_key", "max_new_tokens", "max_length"]}
            )
            
            # 设置生成参数
            model_kwargs = {
                "temperature": temperature,
                "max_new_tokens": kwargs.get("max_new_tokens", 512),
                "max_length": kwargs.get("max_length", 1024)
            }
            
            return HuggingFacePipeline(pipeline=pipe, model_kwargs=model_kwargs)
        
        else:
            raise ValueError(f"不支持的LLM类型: {llm_type}")
```

## 构建完整RAG系统

现在，让我们把所有组件组合起来，构建一个完整的RAG系统：

```python
# 完整RAG系统示例

from src.embeddings.embedder import EmbeddingFactory
from src.vector_store.store import VectorStoreFactory, VectorStoreManager
from src.retriever.retriever import VectorStoreRetriever
from src.llm.model import LLMFactory
from src.rag.basic_rag import BasicRAG

# 1. 创建嵌入模型
embedding_model = EmbeddingFactory.create_embeddings(
    embeddings_type="openai",
    model_name="text-embedding-3-small"
)

# 2. 加载向量数据库
vector_store = VectorStoreFactory.load_vector_store(
    store_type="chroma",
    embeddings=embedding_model,
    persist_directory="data/embeddings/chroma_db"
)

# 3. 创建检索器
retriever = VectorStoreRetriever(
    vector_store=vector_store,
    search_type="similarity",
    search_kwargs={"k": 4}
)

# 4. 创建LLM
llm = LLMFactory.create_llm(
    llm_type="openai",
    model_name="gpt-3.5-turbo",
    temperature=0.0
)

# 5. 创建RAG系统
rag_system = BasicRAG(
    retriever=retriever,
    llm=llm
)

# 6. 处理查询
question = "RAG系统的主要优势是什么？"
result = rag_system.query_with_sources(question)

print(f"问题: {question}")
print(f"\n回答: {result['answer']}")
print("\n来源:")
for source in result['sources']:
    print(f"- {source['source']} (页码: {source['page']})")
    print(f"  预览: {source['content_preview']}")
```

## 自定义提示模板

不同的应用场景可能需要不同的提示模板。以下是一些专为RAG系统设计的提示模板示例：

### 专业问答提示模板

```python
professional_template = """作为一名专业顾问，请使用以下参考资料来回答问题。
如果参考资料中没有提供足够信息，请明确指出并基于你的知识提供建议，但要明确标注这部分不是来自参考资料。

参考资料:
{context}

用户问题: {question}

请提供专业、全面且有条理的回答:"""

professional_rag = BasicRAG(
    retriever=retriever,
    llm=llm,
    prompt_template=professional_template
)
```

### 摘要提示模板

```python
summary_template = """根据以下参考文档，对问题提供简洁的摘要回答。
请将回答限制在3-5个要点内，并确保包含所有关键信息。

参考文档:
{context}

问题: {question}

简洁摘要:"""

summary_rag = BasicRAG(
    retriever=retriever,
    llm=llm,
    prompt_template=summary_template
)
```

## 查询重写策略

用户的原始查询可能不是最优的检索查询。实现查询重写可以提高检索质量：

```python
# 查询重写示例

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class QueryRewriter:
    """查询重写器，用于优化原始查询"""
    
    def __init__(self, llm):
        """初始化查询重写器"""
        self.llm = llm
        
        # 查询重写提示模板
        self.rewrite_template = PromptTemplate.from_template(
            """你是一个专业的搜索查询优化器。你的任务是将用户的原始查询重写为更适合向量检索系统的格式。
            重写的查询应该：
            1. 包含所有重要的关键词和概念
            2. 去除不必要的词语（如"请告诉我"、"我想知道"等）
            3. 扩展缩写和专业术语
            4. 保持简洁明了
            5. 不要添加原始查询中没有的新概念
            
            原始查询: {query}
            
            重写后的查询:"""
        )
        
        # 构建重写链
        self.rewrite_chain = (
            self.rewrite_template
            | self.llm
            | StrOutputParser()
        )
    
    def rewrite_query(self, query: str) -> str:
        """
        重写查询
        
        Args:
            query: 原始查询
            
        Returns:
            重写后的查询
        """
        return self.rewrite_chain.invoke({"query": query})
```

将查询重写器集成到RAG系统：

```python
# 集成查询重写器

class EnhancedRAG(BasicRAG):
    """增强型RAG系统，包含查询重写功能"""
    
    def __init__(
        self, 
        retriever: BaseRetriever,
        llm: BaseLLM,
        prompt_template: Optional[str] = None,
        use_query_rewriting: bool = True
    ):
        """初始化增强型RAG系统"""
        super().__init__(retriever, llm, prompt_template)
        self.use_query_rewriting = use_query_rewriting
        
        if use_query_rewriting:
            self.query_rewriter = QueryRewriter(llm)
    
    def query(self, question: str) -> str:
        """处理用户查询"""
        if self.use_query_rewriting:
            # 重写查询
            rewritten_query = self.query_rewriter.rewrite_query(question)
            print(f"原始查询: {question}")
            print(f"重写查询: {rewritten_query}")
            
            # 使用重写后的查询进行检索，但在提示中使用原始问题
            docs = self.retriever.get_relevant_documents(rewritten_query)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # 构建输入
            prompt_input = {
                "context": context,
                "question": question
            }
            
            # 生成回答
            return self.prompt.format(**prompt_input) | self.llm | StrOutputParser()
        else:
            # 使用基础RAG逻辑
            return super().query(question)
```

## 处理长文档问题

RAG系统面临的一个挑战是处理长文档。当检索到的文档超过LLM的上下文窗口时，我们需要特殊处理：

```python
# 长文档处理示例

class LongDocumentRAG:
    """处理长文档的RAG系统"""
    
    def __init__(self, retriever, llm, max_context_length=3000):
        self.retriever = retriever
        self.llm = llm
        self.max_context_length = max_context_length
        
        # 默认提示模板
        self.prompt = PromptTemplate.from_template(
            """使用以下上下文来回答问题。如果你不知道答案，就说你不知道。
            
上下文:
{context}

问题: {question}

回答:"""
        )
    
    def query(self, question: str) -> str:
        """处理用户查询"""
        # 检索文档
        docs = self.retriever.get_relevant_documents(question)
        
        # 按相关性排序
        # 注意：假设检索器已经按相关性排序
        
        # 合并文档内容直到达到最大上下文长度
        context = ""
        for doc in docs:
            # 检查添加当前文档是否会超过最大长度
            if len(context) + len(doc.page_content) + 2 < self.max_context_length:
                context += doc.page_content + "\n\n"
            else:
                # 如果添加完整文档会超长，尝试添加部分内容
                remaining_length = self.max_context_length - len(context) - 2
                if remaining_length > 200:  # 确保添加的内容有意义
                    context += doc.page_content[:remaining_length] + "..."
                break
        
        # 构建输入
        prompt_input = {
            "context": context,
            "question": question
        }
        
        # 生成回答
        return self.prompt.format(**prompt_input) | self.llm | StrOutputParser()
```

## 基础RAG性能评估

要评估RAG系统的性能，我们可以从多个方面进行衡量：

```python
# RAG评估示例

def evaluate_rag_system(rag_system, test_questions, ground_truth):
    """
    评估RAG系统
    
    Args:
        rag_system: RAG系统实例
        test_questions: 测试问题列表
        ground_truth: 参考答案列表
    """
    results = []
    
    for i, question in enumerate(test_questions):
        # 处理查询
        start_time = time.time()
        answer = rag_system.query(question)
        end_time = time.time()
        
        # 计算响应时间
        response_time = end_time - start_time
        
        # 获取检索到的文档
        retrieved_docs = rag_system.get_retrieved_documents(question)
        
        # 记录结果
        result = {
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth[i],
            "response_time": response_time,
            "num_docs_retrieved": len(retrieved_docs)
        }
        
        results.append(result)
    
    # 计算平均响应时间
    avg_response_time = sum(r["response_time"] for r in results) / len(results)
    print(f"平均响应时间: {avg_response_time:.2f}秒")
    
    # 输出结果
    for result in results:
        print("\n" + "="*50)
        print(f"问题: {result['question']}")
        print(f"系统回答: {result['answer']}")
        print(f"参考答案: {result['ground_truth']}")
        print(f"响应时间: {result['response_time']:.2f}秒")
    
    return results
```

## 实用技巧与最佳实践

### 1. 文档去重

确保检索到的文档不包含重复内容：

```python
def deduplicate_documents(documents):
    """去除重复文档"""
    unique_docs = []
    seen_contents = set()
    
    for doc in documents:
        # 使用文档内容的哈希作为去重标识
        content_hash = hash(doc.page_content)
        
        if content_hash not in seen_contents:
            unique_docs.append(doc)
            seen_contents.add(content_hash)
    
    return unique_docs
```

### 2. 动态调整检索数量

根据查询复杂度动态调整检索文档数量：

```python
def get_dynamic_k(query, min_k=2, max_k=8):
    """根据查询复杂度动态确定k值"""
    # 简单启发式方法：根据查询长度和复杂词汇数量
    query_words = query.split()
    query_length = len(query_words)
    
    # 计算复杂词汇数量（这里简化为长度>5的词）
    complex_words = sum(1 for word in query_words if len(word) > 5)
    
    # 计算k值
    k = min(max(min_k, 2 + query_length//10 + complex_words//2), max_k)
    
    return k
```

### 3. 查询分解

对于复杂查询，可以将其分解为多个子查询并合并结果：

```python
def decompose_query(query, llm):
    """将复杂查询分解为多个子查询"""
    decompose_prompt = PromptTemplate.from_template(
        """你需要将以下复杂问题分解为2-3个简单的子问题，以便更好地检索相关信息。
        这些子问题应该：
        1. 涵盖原始问题的不同方面
        2. 使用简单、直接的语言
        3. 独立可回答
        4. 按逻辑顺序排列
        
        复杂问题: {question}
        
        请直接列出分解后的子问题，每行一个，不要添加编号或其他文本。
        """
    )
    
    # 生成子查询
    sub_queries_text = decompose_prompt.format(question=query) | llm | StrOutputParser()
    
    # 将结果分割为列表
    sub_queries = [q.strip() for q in sub_queries_text.split('\n') if q.strip()]
    
    return sub_queries
```

## 下一步

在本章中，我们构建了一个基础的RAG系统，并介绍了一些增强技术。这个系统已经可以处理简单的RAG应用场景，但在实际应用中，我们可能需要更复杂的技术来提高系统性能。在下一章中，我们将探讨高级RAG技术，包括多查询检索、重排序、上下文压缩等，以进一步提高RAG系统的效果。 