"""
Langchain 检索 (Retrieval) 教程
============================

本教程介绍如何使用Langchain的检索功能实现高效的信息检索
"""

# 导入必要的库
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import (
    ContextualCompressionRetriever,
    SelfQueryRetriever
)
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 设置环境变量
api_key = os.environ.get("ALIBABA_API_KEY", "你的API密钥")
os.environ["ALIBABA_API_KEY"] = api_key

print("检索 (Retrieval) 教程")
print("=" * 50)

# 准备示例文档
documents = [
    Document(page_content="检索是RAG应用中的关键步骤，它决定了系统能够提供多少相关信息。", 
             metadata={"category": "基础知识", "importance": "高"}),
    Document(page_content="向量检索是最常见的检索方式，它基于文本的语义相似性进行检索。", 
             metadata={"category": "技术原理", "importance": "中"}),
    Document(page_content="自查询检索允许系统从自然语言查询中提取结构化查询，用于过滤和检索文档。", 
             metadata={"category": "高级技术", "importance": "高"}),
    Document(page_content="上下文压缩检索可以减少检索到的无关信息，提高检索精度。", 
             metadata={"category": "高级技术", "importance": "中"}),
    Document(page_content="混合检索结合了多种检索策略的优点，可以提高检索性能。", 
             metadata={"category": "高级技术", "importance": "高"}),
    Document(page_content="检索增强生成（RAG）通过检索外部知识来增强大语言模型的回答。", 
             metadata={"category": "应用场景", "importance": "高"}),
    Document(page_content="Langchain提供了多种检索器接口，可以轻松实现复杂的检索功能。", 
             metadata={"category": "工具介绍", "importance": "中"}),
    Document(page_content="构建个性化检索系统需要考虑用户的特定需求和数据特性。", 
             metadata={"category": "最佳实践", "importance": "中"})
]

print(f"准备了 {len(documents)} 个示例文档")

# 设置嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="shibing624/text2vec-base-chinese",  # 中文嵌入模型
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 创建向量存储
vectorstore = FAISS.from_documents(documents, embeddings)

print("已创建FAISS向量存储")

# 1. 基本检索器
print("\n1. 基本检索器 (Retriever)")
print("-" * 40)

# 1.1 从向量存储创建检索器
retriever = vectorstore.as_retriever(
    search_type="similarity",  # similarity（默认）, mmr, similarity_score_threshold
    search_kwargs={
        "k": 2  # 返回的文档数量
    }
)

query = "什么是检索增强生成"
retrieved_docs = retriever.invoke(query)

print(f"查询: '{query}'")
print(f"检索到 {len(retrieved_docs)} 个文档:")
for i, doc in enumerate(retrieved_docs):
    print(f"  {i+1}. {doc.page_content}")
    print(f"     元数据: {doc.metadata}")

# 2. 上下文压缩检索
print("\n2. 上下文压缩检索 (Contextual Compression Retriever)")
print("-" * 40)
print("注意: 上下文压缩检索需要LLM支持")

# 设置LLM
llm = QianfanChatEndpoint(
    model="qwen-max",  # 使用阿里云的通义千问模型
    qianfan_api_key=api_key,
    streaming=True
)

# 创建压缩器
compressor = LLMChainExtractor.from_llm(llm)

# 创建上下文压缩检索器
compression_retriever = ContextualCompressionRetriever(
    base_retriever=retriever,
    base_compressor=compressor
)

query = "RAG系统如何帮助LLM"
compressed_docs = compression_retriever.invoke(query)

print(f"查询: '{query}'")
print(f"压缩检索到 {len(compressed_docs)} 个文档:")
for i, doc in enumerate(compressed_docs):
    print(f"  {i+1}. {doc.page_content}")
    print(f"     (经过上下文压缩，只保留了最相关的内容)")

# 3. 元数据过滤检索
print("\n3. 元数据过滤检索 (Metadata Filtering)")
print("-" * 40)

# 元数据过滤检索
filtered_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3,
        "filter": {"importance": "高"}  # 只检索重要性为"高"的文档
    }
)

query = "检索技术"
filtered_docs = filtered_retriever.invoke(query)

print(f"查询: '{query}'，过滤条件: importance=高")
print(f"过滤检索到 {len(filtered_docs)} 个文档:")
for i, doc in enumerate(filtered_docs):
    print(f"  {i+1}. {doc.page_content}")
    print(f"     元数据: {doc.metadata}")

# 4. 基于RAG的简单问答系统
print("\n4. 基于RAG的简单问答系统")
print("-" * 40)

# 创建RAG提示模板
template = """基于以下检索到的上下文信息回答问题。如果上下文中没有相关信息，请回答"我没有足够的信息来回答这个问题"。

上下文：
{context}

问题：{question}

回答："""

prompt = PromptTemplate.from_template(template)

# 创建RAG链
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 查询并获取回答
query = "RAG如何增强大语言模型的能力？"
response = rag_chain.invoke(query)

print(f"问题: {query}")
print(f"RAG回答: {response}")

# 5. 高级检索技巧
print("\n5. 高级检索技巧")
print("-" * 40)

# 5.1 查询改写
print("\n5.1 查询改写 (Query Transformation)")
print("查询改写可以将用户查询转换为更有效的检索查询")

query_transform_prompt = """
你是一个查询优化专家。你的任务是将用户的原始查询改写为可以更好地用于向量检索的查询。
原始查询：{query}
改写后的查询：
"""

query_transformer = (
    PromptTemplate.from_template(query_transform_prompt)
    | llm
    | StrOutputParser()
)

original_query = "RAG是什么"
transformed_query = query_transformer.invoke({"query": original_query})

print(f"原始查询: {original_query}")
print(f"改写后的查询: {transformed_query}")

# 5.2 多查询检索
print("\n5.2 多查询检索 (Multi-Query Retrieval)")
print("多查询检索通过生成多个不同的查询来增加召回率")

print("""
# 多查询检索示例代码
from langchain.retrievers.multi_query import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

query = "如何实现高效的信息检索"
docs = retriever.get_relevant_documents(query)
""")

# 6. 检索器最佳实践
print("\n6. 检索器最佳实践")
print("-" * 40)

best_practices = [
    "1. 选择合适的嵌入模型：针对中文数据，选择专门的中文嵌入模型",
    "2. 调整分块大小：文档分块的大小直接影响检索效果",
    "3. 使用上下文压缩：减少无关信息，提高检索质量",
    "4. 元数据过滤：利用结构化信息提高检索精度",
    "5. 查询改写：将用户查询转换为更适合检索的形式",
    "6. 混合检索策略：结合关键词和语义检索的优点",
    "7. 评估和优化：定期评估检索系统性能并进行调整"
]

for practice in best_practices:
    print(practice)

print("\n检索总结")
print("=" * 30)
print("1. 检索是RAG系统的核心组件，直接影响生成质量")
print("2. Langchain提供了多种检索器实现，可以根据需求选择合适的检索策略")
print("3. 上下文压缩和元数据过滤可以提高检索质量")
print("4. 查询改写和多查询检索可以提高召回率")
print("5. 构建高质量检索系统需要综合考虑多种因素并持续优化") 