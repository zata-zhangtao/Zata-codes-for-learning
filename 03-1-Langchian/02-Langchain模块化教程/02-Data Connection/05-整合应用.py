"""
Langchain Data Connection 整合应用教程
===================================

本教程演示如何组合使用Langchain的Data Connection模块各个组件，构建一个完整的RAG应用
"""

# 导入必要的库
import os
import tempfile
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# 设置环境变量
api_key = os.environ.get("ALIBABA_API_KEY", "你的API密钥")
os.environ["ALIBABA_API_KEY"] = api_key

print("Langchain Data Connection 整合应用教程")
print("=" * 60)

# 创建一个示例知识库文件
knowledge_base_text = """
# 人工智能基础知识

## 什么是人工智能

人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，致力于创造能够模拟人类智能行为的机器。
人工智能研究包括机器学习、深度学习、自然语言处理、计算机视觉等多个领域。

## 机器学习简介

机器学习是人工智能的一个核心子领域，它使计算机能够在没有明确编程的情况下学习和改进。
机器学习算法可以分为监督学习、无监督学习和强化学习三大类。

### 监督学习

监督学习是指通过标记数据进行训练的机器学习方法。算法学习输入和输出之间的映射关系。
常见的监督学习算法包括线性回归、逻辑回归、决策树、随机森林和神经网络等。

### 无监督学习

无监督学习是指在没有标记数据的情况下，算法自主发现数据中的模式和结构。
常见的无监督学习算法包括聚类算法（如K-means）和降维算法（如PCA）。

### 强化学习

强化学习是一种通过与环境交互并获得反馈来学习的方法。
智能体通过尝试不同的行动并观察结果来学习最优策略。

## 深度学习基础

深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的表示。
深度学习在图像识别、自然语言处理和语音识别等领域取得了突破性进展。

### 神经网络结构

神经网络由输入层、隐藏层和输出层组成。每一层由多个神经元构成，神经元之间通过权重连接。
深度神经网络是指具有多个隐藏层的神经网络。

### 常见深度学习模型

常见的深度学习模型包括卷积神经网络（CNN）、循环神经网络（RNN）、长短期记忆网络（LSTM）和Transformer等。

## 大语言模型

大语言模型（Large Language Models，简称LLM）是一种基于深度学习的自然语言处理模型。
这些模型通过在大量文本数据上进行预训练，能够生成连贯、流畅且具有上下文理解能力的文本。

### 架构

现代大语言模型主要基于Transformer架构，它采用注意力机制来处理序列数据。
Transformer模型由编码器和解码器组成，可以有效地并行处理文本数据。

### 应用场景

大语言模型的应用场景包括文本生成、对话系统、内容摘要、代码生成、翻译等多个领域。

## 人工智能的伦理考量

随着人工智能技术的发展，我们需要考虑多种伦理问题，包括隐私保护、算法偏见、责任归属和就业影响等。
确保人工智能的发展符合人类价值观和道德准则是至关重要的。
"""

# 创建临时文件
with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
    f.write(knowledge_base_text)
    knowledge_base_file = f.name

print(f"创建了临时知识库文件: {knowledge_base_file}")

# 1. 文档加载和处理阶段
print("\n1. 文档加载和处理阶段")
print("-" * 40)

# 1.1 加载文档
loader = TextLoader(knowledge_base_file, encoding="utf-8")
documents = loader.load()
print(f"已加载文档，共 {len(documents)} 个")

# 1.2 文档分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,        # 每个块的最大字符数
    chunk_overlap=50,      # 相邻块之间的重叠字符数
    separators=["\n## ", "\n### ", "\n", " ", ""],  # 分隔符顺序很重要
    length_function=len
)

splits = text_splitter.split_documents(documents)
print(f"文档已分割成 {len(splits)} 个块")
print(f"第一个块的内容: {splits[0].page_content[:100]}...")

# 2. 文档向量化和存储阶段
print("\n2. 文档向量化和存储阶段")
print("-" * 40)

# 2.1 设置嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="shibing624/text2vec-base-chinese",  # 中文嵌入模型
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("已设置中文嵌入模型: shibing624/text2vec-base-chinese")

# 2.2 创建向量存储
vectorstore = FAISS.from_documents(splits, embeddings)
print("已创建FAISS向量存储")

# 3. 设置检索器
print("\n3. 设置检索器")
print("-" * 40)

# 3.1 创建基本检索器
retriever = vectorstore.as_retriever(
    search_type="mmr",  # 使用最大边际相关性搜索
    search_kwargs={
        "k": 3,  # 返回的文档数量
        "fetch_k": 5,  # 初始获取的文档数
        "lambda_mult": 0.7  # 控制多样性的参数，越小多样性越高
    }
)
print("已创建MMR检索器，兼顾相关性和多样性")

# 3.2 设置LLM
llm = QianfanChatEndpoint(
    model="qwen-max",  # 使用阿里云的通义千问模型
    qianfan_api_key=api_key,
    streaming=True
)
print("已设置阿里云通义千问大语言模型")

# 3.3 创建上下文压缩检索器
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_retriever=retriever,
    base_compressor=compressor
)
print("已创建上下文压缩检索器，可以减少无关信息")

# 4. 创建RAG应用
print("\n4. 创建RAG应用")
print("-" * 40)

# 4.1 创建检索增强生成提示模板
template = """你是一位专业的人工智能教育专家，使用简洁明了的语言回答问题。
基于以下检索到的信息回答问题。如果无法从检索到的信息中得到答案，请说"我没有足够的信息回答这个问题"。

检索到的信息:
{context}

问题: {question}

回答:"""

prompt = PromptTemplate.from_template(template)

# 4.2 创建格式化文档函数
def format_docs(docs):
    return "\n\n".join([f"文档 {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

# 4.3 构建RAG链
rag_chain = (
    {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("已创建完整的RAG链，将检索器、提示模板和LLM连接在一起")

# 5. 使用RAG应用回答问题
print("\n5. 使用RAG应用回答问题")
print("-" * 40)

# 准备一些问题
questions = [
    "什么是强化学习？",
    "深度神经网络和普通神经网络有什么区别？",
    "大语言模型的主要应用场景有哪些？",
    "人工智能面临哪些伦理挑战？"
]

# 使用RAG回答问题
print("使用RAG应用回答多个问题:\n")
for i, question in enumerate(questions):
    print(f"问题 {i+1}: {question}")
    answer = rag_chain.invoke(question)
    print(f"回答: {answer}\n")

# 6. 改进和优化RAG应用
print("\n6. RAG应用的改进和优化建议")
print("-" * 40)

improvements = [
    "1. 使用更先进的嵌入模型：可以使用支持多语言的模型如BAAI/bge-large-zh等",
    "2. 增加评估机制：定期评估检索质量和回答准确性",
    "3. 实现查询改写：在检索前对用户查询进行优化",
    "4. 添加多知识源支持：整合多个知识库提高覆盖面",
    "5. 加入人类反馈机制：收集用户反馈不断改进系统",
    "6. 实现历史对话跟踪：支持多轮对话和上下文理解",
    "7. 优化文档分块策略：根据具体内容特点调整分块方式"
]

for improvement in improvements:
    print(improvement)

# 7. 总结
print("\n总结")
print("=" * 40)
print("1. Data Connection模块是Langchain的核心组件之一，为LLM提供外部知识访问能力")
print("2. 完整的RAG应用包括文档加载、文本分割、嵌入向量化、向量存储和检索等环节")
print("3. 通过组合这些组件，可以构建强大的知识增强型AI应用")
print("4. 选择合适的组件和参数配置对系统性能至关重要")
print("5. RAG技术显著提升了LLM在特定领域的知识准确性和回答质量")

# 清理临时文件
import os
if os.path.exists(knowledge_base_file):
    os.remove(knowledge_base_file)
print("\n已清理临时文件") 