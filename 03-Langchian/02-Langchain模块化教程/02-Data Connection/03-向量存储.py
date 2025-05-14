"""
Langchain 向量存储 (Vector Stores) 教程
=====================================

本教程介绍如何使用Langchain的向量存储功能存储和检索向量化的文本
"""

# 导入必要的库
import os
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 设置环境变量（如果需要的话）
api_key = os.environ.get("ALIBABA_API_KEY", "你的API密钥")
os.environ["ALIBABA_API_KEY"] = api_key

print("向量存储 (Vector Stores) 教程")
print("=" * 50)

# 准备示例文档
documents = [
    Document(page_content="向量数据库是AI应用的重要组成部分"),
    Document(page_content="Langchain提供了多种向量存储的集成"),
    Document(page_content="向量检索可以帮助我们找到语义相关的内容"),
    Document(page_content="通过嵌入模型，我们可以将文本转换为向量"),
    Document(page_content="大语言模型可以与向量数据库结合使用"),
    Document(page_content="RAG是检索增强生成的缩写"),
    Document(page_content="检索增强生成可以增强大语言模型的知识"),
    Document(page_content="中文文本同样可以进行向量化和检索")
]

print(f"准备了 {len(documents)} 个示例文档")

# 1. 使用Hugging Face嵌入模型
print("\n1. 使用Hugging Face嵌入模型")
print("-" * 40)
print("注意: 需要安装sentence-transformers: pip install sentence-transformers")

embeddings = HuggingFaceEmbeddings(
    model_name="shibing624/text2vec-base-chinese",  # 中文嵌入模型
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

print("使用shibing624/text2vec-base-chinese模型进行中文文本嵌入")

# 2. 使用Chroma向量存储
print("\n2. 使用Chroma向量存储")
print("-" * 40)
print("注意: 需要安装chromadb: pip install chromadb")

# 创建Chroma向量存储
chroma_db = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"  # 持久化存储目录
)

print("已创建Chroma向量存储并持久化到./chroma_db目录")

# 进行相似性搜索
query = "向量数据库有什么用"
results = chroma_db.similarity_search(query, k=2)

print(f"查询: '{query}'")
print(f"找到 {len(results)} 个相关文档:")
for i, doc in enumerate(results):
    print(f"  {i+1}. {doc.page_content}")
    
# 3. 使用FAISS向量存储
print("\n3. 使用FAISS向量存储")
print("-" * 40)
print("注意: 需要安装faiss-cpu: pip install faiss-cpu")

# 创建FAISS向量存储
faiss_db = FAISS.from_documents(
    documents=documents,
    embedding=embeddings
)

print("已创建FAISS向量存储")

# 进行相似性搜索
query = "检索增强生成"
results = faiss_db.similarity_search(query, k=2)

print(f"查询: '{query}'")
print(f"找到 {len(results)} 个相关文档:")
for i, doc in enumerate(results):
    print(f"  {i+1}. {doc.page_content}")

# 保存和加载FAISS索引
faiss_db.save_local("faiss_index")
print("已保存FAISS索引到faiss_index目录")

loaded_faiss = FAISS.load_local("faiss_index", embeddings)
print("已从faiss_index目录加载FAISS索引")

# 4. 向量存储的高级用法
print("\n4. 向量存储的高级用法")
print("-" * 40)

# 4.1 带元数据的检索
documents_with_metadata = [
    Document(page_content="向量数据库是AI应用的重要组成部分", metadata={"source": "书籍", "page": 1}),
    Document(page_content="Langchain提供了多种向量存储的集成", metadata={"source": "文档", "page": 5}),
    Document(page_content="向量检索可以帮助我们找到语义相关的内容", metadata={"source": "博客", "page": 2}),
]

meta_db = FAISS.from_documents(documents_with_metadata, embeddings)

# 带元数据过滤的检索
print("\n4.1 带元数据过滤的检索")
meta_results = meta_db.similarity_search(
    "向量检索",
    k=1,
    filter={"source": "博客"}
)

print(f"查询带元数据过滤，找到结果: {meta_results[0].page_content}")
print(f"元数据: {meta_results[0].metadata}")

# 4.2 最大边际相关性(MMR)检索
print("\n4.2 最大边际相关性(MMR)检索")
print("MMR在相关性和多样性之间取得平衡")

mmr_results = faiss_db.max_marginal_relevance_search(
    "大语言模型和向量数据库",
    k=2,
    fetch_k=3,
    lambda_mult=0.5  # 0是最大多样性，1是最大相关性
)

print(f"MMR查询结果:")
for i, doc in enumerate(mmr_results):
    print(f"  {i+1}. {doc.page_content}")

# 5. 完整的RAG流程示例
print("\n5. 完整的RAG流程示例")
print("-" * 40)

print("""
# 完整RAG流程示例代码

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import QianfanLLMEndpoint  # 或者其他模型

# 1. 加载文档
loader = TextLoader("my_knowledge_base.txt", encoding="utf-8")
documents = loader.load()

# 2. 分割文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(documents)

# 3. 嵌入并存储在向量数据库中
embeddings = HuggingFaceEmbeddings(
    model_name="shibing624/text2vec-base-chinese"
)
vectorstore = FAISS.from_documents(splits, embeddings)

# 4. 创建检索器
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# 5. 创建LLM
llm = QianfanLLMEndpoint(
    streaming=True,
    model_name="ERNIE-Bot-4"
)

# 6. 创建QA链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 7. 查询并获取回答
query = "解释一下什么是向量数据库"
result = qa_chain({"query": query})
print(result["result"])
""")

print("\n向量存储总结")
print("=" * 30)
print("1. 向量存储是RAG应用的核心组件")
print("2. Langchain支持多种向量存储，包括Chroma、FAISS、Pinecone等")
print("3. 向量存储需要配合嵌入模型使用，将文本转换为向量")
print("4. 在选择向量存储时，需要考虑速度、规模和功能需求")
print("5. 元数据过滤和MMR等高级功能可以提高检索质量") 