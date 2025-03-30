# 高级RAG技术

本章将介绍一系列高级RAG技术，用于增强基础RAG系统的性能和功能。

## 多查询生成

### 概念

多查询生成是指从原始用户查询生成多个不同的查询变体，然后分别执行检索并合并结果的技术。这种方法可以扩大检索范围，捕获更多相关信息。

### 实现

```python
# src/rag/advanced_rag.py

from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.retrievers.base import BaseRetriever

class MultiQueryGenerator:
    """多查询生成器"""
    
    def __init__(self, llm: BaseLLM, num_queries: int = 3):
        """
        初始化多查询生成器
        
        Args:
            llm: 大语言模型实例
            num_queries: 生成的查询数量
        """
        self.llm = llm
        self.num_queries = num_queries
        
        # 定义输出模型
        class QueryList(BaseModel):
            queries: List[str] = Field(description=f"生成的{num_queries}个查询列表")
        
        # 创建输出解析器
        self.parser = PydanticOutputParser(pydantic_object=QueryList)
        
        # 创建多查询生成提示
        template = f"""你是一个AI助手，专门负责生成多样化的搜索查询。
        
对于给定的原始查询，请生成{num_queries}个不同的查询变体，这些变体应该：
1. 表达相同的信息需求
2. 使用不同的措辞和角度
3. 可能包含同义词或相关概念
4. 覆盖不同的信息方面

原始查询: {{original_query}}

{self.parser.get_format_instructions()}
"""
        
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["original_query"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
    
    def generate_queries(self, original_query: str) -> List[str]:
        """
        生成多个查询变体
        
        Args:
            original_query: 原始查询
            
        Returns:
            生成的查询列表
        """
        # 生成查询
        result = self.llm.invoke(self.prompt.format(original_query=original_query))
        
        # 解析结果
        try:
            queries = self.parser.parse(result).queries
        except Exception as e:
            print(f"解析失败: {e}")
            # 返回包含原始查询的列表作为后备方案
            return [original_query]
        
        # 确保原始查询也包含在列表中
        if original_query not in queries:
            queries.append(original_query)
        
        return queries
```

### 使用示例

```python
# 多查询RAG使用示例

class MultiQueryRAG:
    """基于多查询的RAG系统"""
    
    def __init__(
        self, 
        retriever: BaseRetriever,
        llm: BaseLLM,
        num_queries: int = 3,
        unique_docs: bool = True
    ):
        """初始化多查询RAG系统"""
        self.retriever = retriever
        self.llm = llm
        self.query_generator = MultiQueryGenerator(llm, num_queries)
        self.unique_docs = unique_docs
        
        # 创建基础RAG提示
        self.prompt = PromptTemplate.from_template(
            """使用以下上下文来回答问题。如果你不知道答案，就说你不知道。

上下文:
{context}

问题: {question}

回答:"""
        )
    
    def query(self, question: str) -> str:
        """处理用户查询"""
        # 生成多个查询变体
        queries = self.query_generator.generate_queries(question)
        print(f"生成的查询变体: {queries}")
        
        # 使用每个查询变体检索文档
        all_docs = []
        for query in queries:
            docs = self.retriever.get_relevant_documents(query)
            all_docs.extend(docs)
        
        # 去重(如果需要)
        if self.unique_docs:
            unique_docs = []
            seen_contents = set()
            
            for doc in all_docs:
                if doc.page_content not in seen_contents:
                    unique_docs.append(doc)
                    seen_contents.add(doc.page_content)
            
            all_docs = unique_docs
        
        # 合并文档内容
        context = "\n\n".join([doc.page_content for doc in all_docs])
        
        # 生成回答
        prompt_input = {
            "context": context,
            "question": question
        }
        
        return self.llm.invoke(self.prompt.format(**prompt_input))
```

## 检索增强技术

### 1. 密集检索与混合检索

```python
# 混合检索实现

from typing import List, Dict, Any
import numpy as np
from langchain.schema import Document
from langchain.retrievers import BM25Retriever
from langchain.retrievers.base import BaseRetriever

class HybridRetriever(BaseRetriever):
    """混合检索器，结合向量检索与BM25检索"""
    
    def __init__(
        self,
        vector_retriever: BaseRetriever,
        texts: List[str],
        k: int = 4,
        alpha: float = 0.5
    ):
        """
        初始化混合检索器
        
        Args:
            vector_retriever: 向量检索器
            texts: 文本列表用于BM25
            k: 返回结果数量
            alpha: 向量检索权重，1-alpha为BM25权重
        """
        self.vector_retriever = vector_retriever
        self.bm25_retriever = BM25Retriever.from_texts(texts)
        self.k = k
        self.alpha = alpha
        super().__init__()
    
    def _get_relevant_documents(
        self,
        query: str,
        run_manager=None,
    ) -> List[Document]:
        """获取相关文档"""
        # 获取向量检索结果
        vector_docs = self.vector_retriever.get_relevant_documents(query)
        
        # 获取BM25检索结果
        bm25_docs = self.bm25_retriever.get_relevant_documents(query)
        
        # 合并结果
        # 创建文档ID到(文档,分数)的映射
        results_map = {}
        
        # 向量检索结果
        for i, doc in enumerate(vector_docs):
            doc_id = hash(doc.page_content)
            # 使用位置作为相对分数(倒序)
            score = self.alpha * (len(vector_docs) - i) / len(vector_docs)
            results_map[doc_id] = (doc, score)
        
        # BM25检索结果
        for i, doc in enumerate(bm25_docs):
            doc_id = hash(doc.page_content)
            # 使用位置作为相对分数(倒序)
            bm25_score = (1 - self.alpha) * (len(bm25_docs) - i) / len(bm25_docs)
            
            if doc_id in results_map:
                # 现有文档，增加分数
                existing_doc, existing_score = results_map[doc_id]
                results_map[doc_id] = (existing_doc, existing_score + bm25_score)
            else:
                # 新文档
                results_map[doc_id] = (doc, bm25_score)
        
        # 排序并返回前k个结果
        sorted_results = sorted(
            results_map.values(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [doc for doc, _ in sorted_results[:self.k]]
```

### 2. 重排序技术

```python
# 检索结果重排序实现

from typing import List, Tuple, Dict, Any
from langchain.schema import Document
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate

class CrossEncoderReranker:
    """基于交叉编码器的重排序器"""
    
    def __init__(self, llm: BaseLLM):
        """初始化重排序器"""
        self.llm = llm
        
        # 相关性评分提示
        self.relevance_prompt = PromptTemplate.from_template(
            """评估以下文档与查询的相关性。
            
查询: {query}

文档: {document}

在1到10的范围内给文档评分，其中:
1 = 完全不相关
5 = 部分相关
10 = 非常相关

仅返回数字分数，不要有任何解释。
"""
        )
    
    def rerank_documents(
        self,
        query: str,
        documents: List[Document],
        top_n: int = None
    ) -> List[Document]:
        """
        重排序文档
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_n: 返回前n个结果
            
        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []
        
        if top_n is None:
            top_n = len(documents)
        
        # 为每个文档计算相关性分数
        scored_docs = []
        
        for doc in documents:
            # 格式化提示
            prompt_input = {
                "query": query,
                "document": doc.page_content
            }
            
            # 获取相关性分数
            score_text = self.llm.invoke(self.relevance_prompt.format(**prompt_input))
            
            # 解析分数
            try:
                score = float(score_text.strip())
            except ValueError:
                # 如果解析失败，给一个中等分数
                score = 5.0
            
            scored_docs.append((doc, score))
        
        # 按分数排序
        sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        
        # 返回前top_n个文档
        return [doc for doc, _ in sorted_docs[:top_n]]
```

## 上下文压缩

当检索到的文档过长或包含大量不相关内容时，上下文压缩可以提高效率：

```python
# 上下文压缩实现

from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.llms.base import BaseLLM

class ContextCompressor:
    """上下文压缩器，提取最相关的内容"""
    
    def __init__(self, llm: BaseLLM, max_tokens: int = 2000):
        """
        初始化上下文压缩器
        
        Args:
            llm: 大语言模型实例
            max_tokens: 最大输出标记数
        """
        self.llm = llm
        self.max_tokens = max_tokens
    
    def compress_documents(
        self,
        query: str,
        documents: List[Document]
    ) -> List[Document]:
        """
        压缩文档列表
        
        Args:
            query: 查询文本
            documents: 要压缩的文档列表
            
        Returns:
            压缩后的文档列表
        """
        # 定义压缩提示
        compress_prompt = f"""你是一个专业的文档提取系统。你的任务是从以下文档中提取与查询最相关的部分。
        
查询: {query}

文档:
{{document}}

请提取并返回与查询直接相关的文本片段。保留所有重要的信息，但删除不相关内容。提取的内容不应超过原文档的50%，但要确保包含所有可能有助于回答查询的关键信息。仅返回提取的内容，不要添加任何额外解释。
"""
        
        compressed_docs = []
        
        for doc in documents:
            # 获取原始元数据
            metadata = doc.metadata.copy() if doc.metadata else {}
            
            # 使用LLM压缩文档
            prompt_input = compress_prompt.format(document=doc.page_content)
            compressed_content = self.llm.invoke(prompt_input)
            
            # 创建新文档
            compressed_doc = Document(
                page_content=compressed_content,
                metadata={
                    **metadata,
                    "compressed": True,
                    "original_length": len(doc.page_content),
                    "compressed_length": len(compressed_content)
                }
            )
            
            compressed_docs.append(compressed_doc)
        
        return compressed_docs
```

## 生成增强技术

### 1. 可控生成

```python
# 可控生成实现

class ControlledGenerationRAG:
    """可控生成RAG系统"""
    
    def __init__(self, retriever, llm):
        """初始化可控生成RAG"""
        self.retriever = retriever
        self.llm = llm
    
    def query_with_style(
        self,
        question: str,
        style: str = "detailed",
        format: str = "paragraph"
    ) -> str:
        """
        按特定风格生成回答
        
        Args:
            question: 用户问题
            style: 回答风格，可选值包括"detailed"、"concise"、"simple"、"technical"等
            format: 回答格式，可选值包括"paragraph"、"bullet_points"、"numbered_list"、"table"等
            
        Returns:
            生成的回答
        """
        # 检索文档
        docs = self.retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 构建风格化提示
        style_descriptions = {
            "detailed": "详细且全面的回答，包含所有相关细节",
            "concise": "简明扼要的回答，只包含最重要的信息",
            "simple": "使用简单易懂的语言，避免专业术语",
            "technical": "使用专业术语和精确表述的技术性回答",
            "educational": "以教育性方式解释概念，包含例子和解释"
        }
        
        format_instructions = {
            "paragraph": "以连贯的段落形式呈现",
            "bullet_points": "使用要点列表格式",
            "numbered_list": "使用编号列表格式",
            "table": "在适用的情况下使用表格格式组织信息",
            "qa_pairs": "使用问答对的形式组织信息"
        }
        
        # 获取风格和格式说明
        style_desc = style_descriptions.get(style, style_descriptions["detailed"])
        format_desc = format_instructions.get(format, format_instructions["paragraph"])
        
        # 构建提示
        prompt = f"""根据以下上下文回答问题。

上下文:
{context}

问题: {question}

请提供一个{style_desc}，并{format_desc}。如果上下文中没有足够信息，请承认你不知道，不要编造信息。
"""
        
        # 生成回答
        return self.llm.invoke(prompt)
```

### 2. 逐步推理

```python
# 逐步推理实现

class ChainOfThoughtRAG:
    """基于逐步推理的RAG系统"""
    
    def __init__(self, retriever, llm):
        """初始化逐步推理RAG"""
        self.retriever = retriever
        self.llm = llm
        
        # 逐步推理提示
        self.cot_prompt = PromptTemplate.from_template(
            """根据以下上下文回答问题。请一步一步思考，确保你的推理逻辑清晰。

上下文:
{context}

问题: {question}

请按以下步骤思考:
1. 分析问题，确定需要从上下文中提取的关键信息
2. 从上下文中提取相关信息和事实
3. 使用这些信息逐步推理
4. 得出最终结论

逐步思考:"""
        )
    
    def query(self, question: str) -> str:
        """执行逐步推理查询"""
        # 检索文档
        docs = self.retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 构建输入
        prompt_input = {
            "context": context,
            "question": question
        }
        
        # 生成逐步推理回答
        cot_reasoning = self.llm.invoke(self.cot_prompt.format(**prompt_input))
        
        return cot_reasoning
```

## 多模态RAG

随着模型能力的提升，多模态RAG变得越来越重要：

```python
# 多模态RAG示意实现

class MultimodalRAG:
    """多模态RAG系统"""
    
    def __init__(self, text_retriever, image_analyzer, llm):
        """初始化多模态RAG"""
        self.text_retriever = text_retriever
        self.image_analyzer = image_analyzer  # 假设这是能处理图像的组件
        self.llm = llm
    
    def query(self, question: str, image_paths: List[str] = None) -> str:
        """处理多模态查询"""
        # 检索文本文档
        text_docs = self.text_retriever.get_relevant_documents(question)
        text_context = "\n\n".join([doc.page_content for doc in text_docs])
        
        # 处理图像(如果有)
        image_context = ""
        if image_paths:
            for image_path in image_paths:
                # 分析图像，获取描述
                image_description = self.image_analyzer.analyze_image(image_path)
                image_context += f"图像描述: {image_description}\n\n"
        
        # 合并上下文
        combined_context = text_context
        if image_context:
            combined_context += "\n\n" + image_context
        
        # 构建提示
        prompt = f"""根据以下文本和图像信息回答问题。

信息:
{combined_context}

问题: {question}

回答:"""
        
        # 生成回答
        return self.llm.invoke(prompt)
```

## Agent增强RAG

将RAG与Agent架构结合可以使系统更加智能：

```python
# Agent增强RAG示意实现

class AgentRAG:
    """基于Agent的RAG系统"""
    
    def __init__(self, retriever, llm, tools=None):
        """初始化Agent RAG"""
        self.retriever = retriever
        self.llm = llm
        self.tools = tools or []  # 各种工具函数
    
    def process_query(self, question: str) -> str:
        """处理查询"""
        # 分析问题，决定是否需要检索
        analysis_prompt = f"""分析以下问题，并确定解答它需要哪些步骤。
        
问题: {question}

请考虑:
1. 是否需要从知识库检索信息?
2. 是否需要计算或推理?
3. 是否需要调用特定工具?

以JSON格式返回你的分析结果:
{{
  "needs_retrieval": true/false,
  "needs_calculation": true/false,
  "tools_needed": ["工具1", "工具2", ...],
  "reasoning": "你的推理过程"
}}
"""
        
        # 执行分析
        analysis_result = self.llm.invoke(analysis_prompt)
        
        try:
            # 解析分析结果(简化示例)
            import json
            analysis = json.loads(analysis_result)
            
            # 根据分析结果决定行动
            if analysis.get("needs_retrieval", True):
                # 执行检索
                docs = self.retriever.get_relevant_documents(question)
                context = "\n\n".join([doc.page_content for doc in docs])
            else:
                context = "无需检索外部知识。"
            
            # 调用工具(如果需要)
            tool_results = []
            for tool_name in analysis.get("tools_needed", []):
                if tool_name in self.tools:
                    tool_result = self.tools[tool_name](question)
                    tool_results.append(f"工具 '{tool_name}' 结果: {tool_result}")
            
            # 构建最终提示
            final_prompt = f"""根据以下信息回答问题。

检索到的信息:
{context}

{"工具结果:\n" + "\n".join(tool_results) if tool_results else ""}

问题: {question}

请提供全面且准确的回答。
"""
            
            # 生成最终回答
            return self.llm.invoke(final_prompt)
            
        except Exception as e:
            # 解析失败时的后备方案
            print(f"分析失败: {e}")
            # 执行标准RAG流程
            docs = self.retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            prompt = f"""根据以下上下文回答问题。

上下文:
{context}

问题: {question}

回答:"""
            
            return self.llm.invoke(prompt)
```

## 自适应RAG

自适应RAG能够根据查询和上下文动态调整系统行为：

```python
# 自适应RAG示意实现

class AdaptiveRAG:
    """自适应RAG系统"""
    
    def __init__(self, retrievers, llm):
        """
        初始化自适应RAG
        
        Args:
            retrievers: 检索器字典，格式为{名称: 检索器}
            llm: 大语言模型实例
        """
        self.retrievers = retrievers
        self.llm = llm
    
    def query(self, question: str) -> str:
        """处理查询"""
        # 分析查询类型
        query_analysis_prompt = f"""分析以下问题，并确定最适合回答它的检索策略。

问题: {question}

可用的检索策略:
{", ".join(self.retrievers.keys())}

请选择最合适的策略并解释原因:
"""
        
        # 获取分析结果
        analysis = self.llm.invoke(query_analysis_prompt)
        
        # 简化的策略选择(实际应用中可能需要更复杂的解析)
        selected_retriever = None
        for retriever_name in self.retrievers.keys():
            if retriever_name.lower() in analysis.lower():
                selected_retriever = retriever_name
                break
        
        # 默认选择第一个检索器
        if not selected_retriever:
            selected_retriever = list(self.retrievers.keys())[0]
        
        print(f"选择的检索策略: {selected_retriever}")
        
        # 使用选择的检索器
        retriever = self.retrievers[selected_retriever]
        docs = retriever.get_relevant_documents(question)
        
        # 合并上下文
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 构建提示
        prompt = f"""根据以下上下文回答问题。

上下文:
{context}

问题: {question}

回答:"""
        
        # 生成回答
        return self.llm.invoke(prompt)
```

## 下一步

在本章中，我们探讨了多种高级RAG技术，这些技术可以显著提升RAG系统的性能。每种技术都有其特定的应用场景和优势。在下一章中，我们将学习如何评估和优化RAG系统，包括定义评估指标、识别性能瓶颈和调整系统参数，以确保RAG系统能够达到最佳性能。 