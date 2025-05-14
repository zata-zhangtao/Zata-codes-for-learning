"""
Langchain 文档加载器 (Document Loaders) 教程
===========================================

本教程介绍如何使用Langchain的文档加载器从各种来源加载文档
"""

# 导入必要的库
import os
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    CSVLoader, 
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader
)

# 设置环境变量（如果需要的话）
api_key = os.environ.get("ALIBABA_API_KEY", "你的API密钥")
os.environ["ALIBABA_API_KEY"] = api_key

print("文档加载器 (Document Loaders) 教程")
print("=" * 50)

# 1. 简单文本文件加载器
print("\n1. 文本文件加载器 (TextLoader)")
print("-" * 30)

# 创建一个临时的文本文件用于演示
with open("sample.txt", "w", encoding="utf-8") as f:
    f.write("这是一个示例文本文件。\n")
    f.write("Langchain是一个强大的LLM应用开发框架。\n")
    f.write("文档加载器帮助我们从各种来源获取数据。")

# 使用TextLoader加载文本文件
text_loader = TextLoader("sample.txt", encoding="utf-8")
text_documents = text_loader.load()

print(f"加载了 {len(text_documents)} 个文档")
print(f"文档内容: {text_documents[0].page_content[:50]}...")
print(f"文档元数据: {text_documents[0].metadata}")

# 2. PDF文件加载器
print("\n2. PDF文件加载器 (PyPDFLoader)")
print("-" * 30)
print("注意: 需要安装pypdf库: pip install pypdf")
print("PyPDFLoader用法示例:")
print("""
pdf_loader = PyPDFLoader("example.pdf")
pdf_documents = pdf_loader.load()
print(f"加载了 {len(pdf_documents)} 页PDF")
print(f"第一页内容: {pdf_documents[0].page_content[:50]}...")
""")

# 3. CSV文件加载器
print("\n3. CSV文件加载器 (CSVLoader)")
print("-" * 30)
print("注意: CSV加载器会将每一行转换为一个文档")
print("CSVLoader用法示例:")
print("""
csv_loader = CSVLoader("data.csv")
csv_documents = csv_loader.load()
print(f"加载了 {len(csv_documents)} 个文档 (行)")
""")

# 4. HTML文件加载器
print("\n4. HTML文件加载器 (UnstructuredHTMLLoader)")
print("-" * 30)
print("注意: 需要安装unstructured库: pip install unstructured")
print("UnstructuredHTMLLoader用法示例:")
print("""
html_loader = UnstructuredHTMLLoader("page.html")
html_documents = html_loader.load()
print(f"HTML内容: {html_documents[0].page_content[:50]}...")
""")

# 5. Markdown文件加载器
print("\n5. Markdown文件加载器 (UnstructuredMarkdownLoader)")
print("-" * 30)
print("UnstructuredMarkdownLoader用法示例:")
print("""
md_loader = UnstructuredMarkdownLoader("readme.md")
md_documents = md_loader.load()
print(f"Markdown内容: {md_documents[0].page_content[:50]}...")
""")

# 6. 网页加载器
print("\n6. 网页加载器 (WebBaseLoader)")
print("-" * 30)
print("注意: 需要安装requests和beautifulsoup4库")
print("""
from langchain_community.document_loaders import WebBaseLoader

web_loader = WebBaseLoader("https://python.langchain.com/docs/modules/data_connection/document_loaders/")
web_documents = web_loader.load()
print(f"网页内容: {web_documents[0].page_content[:50]}...")
""")

# 7. 目录加载器
print("\n7. 目录加载器 (DirectoryLoader)")
print("-" * 30)
print("""
from langchain_community.document_loaders import DirectoryLoader

# 加载目录中所有的.txt文件
dir_loader = DirectoryLoader("./my_documents/", glob="**/*.txt", loader_cls=TextLoader)
dir_documents = dir_loader.load()
print(f"加载了 {len(dir_documents)} 个文本文档")
""")

print("\n文档加载器总结")
print("=" * 30)
print("1. Langchain提供了丰富的文档加载器，可以从各种来源加载文档")
print("2. 每个加载器返回的是Document对象的列表，每个Document具有content和metadata属性")
print("3. 可以根据数据来源选择合适的加载器")
print("4. 某些加载器可能需要安装额外的依赖")

# 清理临时文件
import os
if os.path.exists("sample.txt"):
    os.remove("sample.txt") 