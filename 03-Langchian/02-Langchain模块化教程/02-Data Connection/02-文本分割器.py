"""
Langchain 文本分割器 (Text Splitters) 教程
=========================================

本教程介绍如何使用Langchain的文本分割器将长文本分割成可管理的块
"""

# 导入必要的库
import os
from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter
)

# 设置环境变量（如果需要的话）
api_key = os.environ.get("ALIBABA_API_KEY", "你的API密钥")
os.environ["ALIBABA_API_KEY"] = api_key

print("文本分割器 (Text Splitters) 教程")
print("=" * 50)

# 创建一个长文本用于演示
long_text = """
# Langchain 框架介绍

Langchain是一个用于开发由大型语言模型（LLM）驱动的应用程序的框架。

## 核心组件

Langchain框架包含几个核心组件：

### 1. 模型 IO (Model IO)
这个组件处理与LLM的交互。它支持多种模型提供商，如OpenAI、Anthropic、百度、阿里等。

### 2. 数据连接 (Data Connection)
这个组件允许LLM与其他数据源交互。它包括：
- 文档加载器：从各种来源加载文档
- 文本分割器：将长文本分割成可管理的块
- 向量存储：存储和检索向量化的文本
- 检索器：实现高效的信息检索

### 3. 链 (Chains)
链允许将多个组件组合在一起以创建更复杂的应用程序。

### 4. 记忆 (Memory)
记忆组件允许存储和检索与特定应用程序相关的信息。

### 5. 代理 (Agents)
代理使用LLM来确定应该采取哪些行动以及按什么顺序采取这些行动。
"""

# 1. 基于字符的分割器
print("\n1. 基于字符的分割器 (CharacterTextSplitter)")
print("-" * 40)

char_splitter = CharacterTextSplitter(
    separator="\n\n",  # 按两个换行符分割
    chunk_size=200,    # 每个块的最大字符数
    chunk_overlap=50,  # 相邻块之间的重叠字符数
    length_function=len  # 用于计算块长度的函数
)

# 从文本创建文档
document = Document(page_content=long_text)

# 分割文档
char_split_docs = char_splitter.split_documents([document])

print(f"文档被分割成 {len(char_split_docs)} 个块")
print(f"第一个块的内容: {char_split_docs[0].page_content[:100]}...")
print(f"块的元数据: {char_split_docs[0].metadata}")

# 2. 递归字符分割器
print("\n2. 递归字符分割器 (RecursiveCharacterTextSplitter)")
print("-" * 40)

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=20,
    separators=["\n# ", "\n## ", "\n### ", "\n", " ", ""]  # 分隔符顺序很重要
)

recursive_split_docs = recursive_splitter.split_documents([document])

print(f"文档被递归分割成 {len(recursive_split_docs)} 个块")
print(f"第一个块的内容: {recursive_split_docs[0].page_content[:100]}...")
print(f"第二个块的内容: {recursive_split_docs[1].page_content[:100]}...")

# 3. 基于Token的分割器
print("\n3. 基于Token的分割器 (TokenTextSplitter)")
print("-" * 40)
print("注意: 这种分割器基于token而不是字符，更适合LLM使用")

token_splitter = TokenTextSplitter(
    chunk_size=100,  # 每个块的最大token数
    chunk_overlap=10  # 相邻块之间的重叠token数
)

token_split_docs = token_splitter.split_documents([document])

print(f"文档被分割成 {len(token_split_docs)} 个token块")
print(f"第一个块的内容: {token_split_docs[0].page_content[:100]}...")

# 4. Markdown标题分割器
print("\n4. Markdown标题分割器 (MarkdownHeaderTextSplitter)")
print("-" * 40)
print("注意: 这种分割器根据Markdown标题分割文本，并在元数据中保留标题层次")

headers_to_split_on = [
    ("#", "标题1"),
    ("##", "标题2"),
    ("###", "标题3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
markdown_split_docs = markdown_splitter.split_text(long_text)

print(f"Markdown文档被分割成 {len(markdown_split_docs)} 个块")
for i, doc in enumerate(markdown_split_docs[:3]):
    print(f"块 {i+1} 的元数据: {doc.metadata}")
    print(f"块 {i+1} 的内容: {doc.page_content[:50]}...")

# 5. 分割长文档的完整流程
print("\n5. 分割长文档的完整流程")
print("-" * 40)

print("""
# 完整流程示例

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 加载文档
loader = TextLoader("my_long_document.txt")
documents = loader.load()

# 2. 分割文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(documents)

# 3. 现在可以使用这些分割后的文档进行向量化和检索
print(f"文档被分割成 {len(splits)} 个块，可以用于向量化和检索")
""")

print("\n文本分割器总结")
print("=" * 30)
print("1. 选择适当的分割器取决于文本的结构和用例")
print("2. RecursiveCharacterTextSplitter通常是一个很好的通用选择")
print("3. 合适的chunk_size和chunk_overlap对检索性能至关重要")
print("4. 对于结构化文本(如Markdown)，使用专门的分割器可以保留文档结构")
print("5. 适当的文本分割是构建高效检索系统的基础") 