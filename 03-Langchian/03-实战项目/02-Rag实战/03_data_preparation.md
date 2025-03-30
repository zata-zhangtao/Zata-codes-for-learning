# 数据准备

数据准备是构建有效RAG系统的关键步骤。本章将介绍如何收集、清洗和分割文本数据，为后续的嵌入和检索过程做好准备。

## 数据收集

### 数据来源

RAG系统可以使用多种来源的数据：

1. **文档库**：PDF、Word、PowerPoint、Markdown、HTML等文档
2. **网站内容**：通过爬虫或API获取的网页内容
3. **数据库**：结构化数据库中的文本内容
4. **API**：通过API获取的第三方数据
5. **开放数据集**：如Wikipedia、ArXiv等

### 示例数据

为了本教程，我们将使用一些示例数据进行演示。您可以使用自己的数据替换这些示例：

1. 创建一个示例PDF文档
2. 准备一些网页内容
3. 添加一些CSV或JSON格式的结构化数据

```bash
# 在data/raw目录中下载示例数据
mkdir -p data/raw/articles
mkdir -p data/raw/docs
mkdir -p data/raw/websites

# 下载示例PDF文件（如果需要）
# wget -O data/raw/docs/sample_document.pdf https://example.com/sample.pdf
```

## 数据加载

我们需要从不同的数据源加载数据。LangChain提供了多种文档加载器，使这个过程变得简单。

### 创建通用文档加载器

首先，让我们创建一个通用的文档加载器模块：

```python
# src/data/loader.py

from typing import List, Optional
import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader
)
from langchain.schema import Document

class DocumentLoader:
    """通用文档加载器，支持多种文件格式"""
    
    def __init__(self):
        # 文件扩展名到加载器的映射
        self.loader_map = {
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".txt": TextLoader,
            ".csv": CSVLoader,
            ".html": UnstructuredHTMLLoader,
            ".htm": UnstructuredHTMLLoader,
            ".md": UnstructuredMarkdownLoader,
            ".pptx": UnstructuredPowerPointLoader,
            ".ppt": UnstructuredPowerPointLoader,
        }
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        加载单个文档
        
        Args:
            file_path: 文档路径
            
        Returns:
            包含文档内容的Document对象列表
        """
        # 获取文件扩展名
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()
        
        # 检查是否支持该文件类型
        if file_extension not in self.loader_map:
            raise ValueError(f"不支持的文件类型: {file_extension}")
        
        # 加载文档
        loader_class = self.loader_map[file_extension]
        loader = loader_class(file_path)
        
        return loader.load()
    
    def load_documents_from_directory(
        self, 
        directory_path: str, 
        file_extensions: Optional[List[str]] = None
    ) -> List[Document]:
        """
        从目录中加载所有支持的文档
        
        Args:
            directory_path: 目录路径
            file_extensions: 要加载的文件扩展名列表，如果为None则加载所有支持的扩展名
            
        Returns:
            包含所有文档内容的Document对象列表
        """
        documents = []
        
        # 如果没有指定文件扩展名，使用所有支持的扩展名
        if file_extensions is None:
            file_extensions = list(self.loader_map.keys())
            
        # 遍历目录中的所有文件
        for dirpath, _, filenames in os.walk(directory_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                _, file_extension = os.path.splitext(file_path)
                file_extension = file_extension.lower()
                
                # 如果文件扩展名在指定的扩展名列表中，加载该文件
                if file_extension in file_extensions:
                    try:
                        docs = self.load_document(file_path)
                        documents.extend(docs)
                        print(f"成功加载 {file_path}")
                    except Exception as e:
                        print(f"加载 {file_path} 失败: {e}")
        
        return documents
```

### 使用示例

```python
# 使用文档加载器的示例

from src.data.loader import DocumentLoader

# 初始化加载器
loader = DocumentLoader()

# 加载单个文档
pdf_docs = loader.load_document("data/raw/docs/sample_document.pdf")
print(f"已加载 {len(pdf_docs)} 个文档片段")

# 从目录加载所有文档
all_docs = loader.load_documents_from_directory("data/raw", [".pdf", ".txt", ".md"])
print(f"总共加载了 {len(all_docs)} 个文档片段")
```

## 数据清洗

从文档中提取的原始文本通常需要清洗，以去除不必要的内容、格式化问题和噪声。

### 创建文本处理器

```python
# src/data/processor.py

import re
from typing import List
from langchain.schema import Document

class TextProcessor:
    """文本处理器，用于清洗和规范化文本"""
    
    def __init__(self):
        pass
    
    def clean_text(self, text: str) -> str:
        """
        清洗文本
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        # 移除多余的空白字符
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        
        # 移除URL（可选）
        # cleaned_text = re.sub(r'https?://\S+', '', cleaned_text)
        
        # 移除特殊字符（根据需求调整）
        # cleaned_text = re.sub(r'[^\w\s\.\,\?\!]', '', cleaned_text)
        
        return cleaned_text
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        处理文档列表
        
        Args:
            documents: 文档列表
            
        Returns:
            处理后的文档列表
        """
        processed_documents = []
        
        for doc in documents:
            # 清洗页面内容
            cleaned_content = self.clean_text(doc.page_content)
            
            # 创建新的文档对象，保留原始的元数据
            processed_doc = Document(
                page_content=cleaned_content,
                metadata=doc.metadata
            )
            
            processed_documents.append(processed_doc)
            
        return processed_documents
    
    def filter_documents(self, documents: List[Document], min_length: int = 50) -> List[Document]:
        """
        过滤文档，去除过短或空的文档
        
        Args:
            documents: 文档列表
            min_length: 最小文档长度
            
        Returns:
            过滤后的文档列表
        """
        return [doc for doc in documents if len(doc.page_content) >= min_length]
```

### 使用示例

```python
# 使用文本处理器的示例

from src.data.loader import DocumentLoader
from src.data.processor import TextProcessor

# 加载文档
loader = DocumentLoader()
raw_docs = loader.load_documents_from_directory("data/raw")

# 处理文档
processor = TextProcessor()
cleaned_docs = processor.process_documents(raw_docs)
filtered_docs = processor.filter_documents(cleaned_docs, min_length=100)

print(f"原始文档数: {len(raw_docs)}")
print(f"清洗后文档数: {len(cleaned_docs)}")
print(f"过滤后文档数: {len(filtered_docs)}")
```

## 文本分割

大型文档通常需要分割成较小的块，以便：
1. 符合嵌入模型的最大输入长度限制
2. 创建更精细的检索单元
3. 优化语义搜索的效果

### 创建文本分割器

```python
# src/data/splitter.py

from typing import List
from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)

class TextSplitter:
    """文本分割器，将文档分割成较小的块"""
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200, 
        splitter_type: str = "recursive"
    ):
        """
        初始化文本分割器
        
        Args:
            chunk_size: 每个文本块的目标大小
            chunk_overlap: 相邻块之间的重叠字符数
            splitter_type: 分割器类型，可选值为"recursive"、"character"、"token"
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 选择分割器类型
        if splitter_type == "recursive":
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        elif splitter_type == "character":
            self.splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="\n"
            )
        elif splitter_type == "token":
            self.splitter = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            raise ValueError(f"不支持的分割器类型: {splitter_type}")
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        分割文档
        
        Args:
            documents: 要分割的文档列表
            
        Returns:
            分割后的文档列表
        """
        return self.splitter.split_documents(documents)
    
    def split_text(self, text: str) -> List[str]:
        """
        分割单个文本字符串
        
        Args:
            text: 要分割的文本
            
        Returns:
            分割后的文本列表
        """
        return self.splitter.split_text(text)
```

### 使用示例

```python
# 使用文本分割器的示例

from src.data.loader import DocumentLoader
from src.data.processor import TextProcessor
from src.data.splitter import TextSplitter

# 加载文档
loader = DocumentLoader()
raw_docs = loader.load_documents_from_directory("data/raw")

# 处理文档
processor = TextProcessor()
cleaned_docs = processor.process_documents(raw_docs)

# 分割文档
splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(cleaned_docs)

print(f"原始文档数: {len(raw_docs)}")
print(f"处理后文档数: {len(cleaned_docs)}")
print(f"分割后文档块数: {len(split_docs)}")

# 查看一个分割后的块示例
if split_docs:
    print("\n示例块:")
    print("内容:", split_docs[0].page_content[:100] + "...")
    print("元数据:", split_docs[0].metadata)
```

## 元数据增强

为了提高检索效果，我们可以为每个文档块添加有用的元数据：

```python
# 元数据增强示例

from typing import List
import hashlib
from datetime import datetime
from langchain.schema import Document

def enhance_metadata(documents: List[Document]) -> List[Document]:
    """
    增强文档元数据
    
    Args:
        documents: 文档列表
        
    Returns:
        增强了元数据的文档列表
    """
    enhanced_docs = []
    
    for i, doc in enumerate(documents):
        # 获取现有元数据或创建空字典
        metadata = doc.metadata.copy() if doc.metadata else {}
        
        # 添加块ID
        metadata["chunk_id"] = i
        
        # 添加内容哈希（用于去重）
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
        metadata["content_hash"] = content_hash
        
        # 添加处理时间戳
        metadata["processed_at"] = datetime.now().isoformat()
        
        # 计算文本长度
        metadata["char_count"] = len(doc.page_content)
        metadata["word_count"] = len(doc.page_content.split())
        
        # 创建新文档
        enhanced_doc = Document(
            page_content=doc.page_content,
            metadata=metadata
        )
        
        enhanced_docs.append(enhanced_doc)
    
    return enhanced_docs
```

## 完整的数据处理管道

将上述所有步骤组合在一起，创建一个完整的数据处理管道：

```python
# 完整的数据处理管道示例

import os
import json
from src.data.loader import DocumentLoader
from src.data.processor import TextProcessor
from src.data.splitter import TextSplitter

class DataPipeline:
    """完整的数据处理管道"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_length: int = 50
    ):
        self.loader = DocumentLoader()
        self.processor = TextProcessor()
        self.splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.min_chunk_length = min_chunk_length
    
    def process(self, input_dir: str, output_dir: str = None, file_extensions: list = None):
        """
        处理输入目录中的所有文档
        
        Args:
            input_dir: 输入目录路径
            output_dir: 输出目录路径，如果为None则不保存结果
            file_extensions: 要处理的文件扩展名列表
        
        Returns:
            处理后的文档列表
        """
        # 第1步：加载文档
        print(f"正在从 {input_dir} 加载文档...")
        raw_docs = self.loader.load_documents_from_directory(input_dir, file_extensions)
        print(f"已加载 {len(raw_docs)} 个文档片段")
        
        # 第2步：清洗文档
        print("正在清洗文档...")
        cleaned_docs = self.processor.process_documents(raw_docs)
        
        # 第3步：过滤过短的文档
        filtered_docs = self.processor.filter_documents(cleaned_docs, min_length=self.min_chunk_length)
        print(f"清洗和过滤后剩余 {len(filtered_docs)} 个文档片段")
        
        # 第4步：分割文档
        print("正在分割文档...")
        split_docs = self.splitter.split_documents(filtered_docs)
        print(f"分割后得到 {len(split_docs)} 个文档块")
        
        # 第5步：增强元数据
        print("正在增强元数据...")
        processed_docs = enhance_metadata(split_docs)
        
        # 第6步：保存处理后的文档（如果指定了输出目录）
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            print(f"正在将处理后的文档保存到 {output_dir}...")
            
            # 将文档转换为可序列化的字典
            docs_dicts = []
            for doc in processed_docs:
                doc_dict = {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                docs_dicts.append(doc_dict)
            
            # 保存为JSON文件
            output_path = os.path.join(output_dir, "processed_documents.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(docs_dicts, f, ensure_ascii=False, indent=2)
            
            print(f"已将处理后的文档保存到 {output_path}")
        
        return processed_docs
```

### 使用数据管道

```python
# 使用数据处理管道

from src.data import DataPipeline

# 创建数据管道
pipeline = DataPipeline(
    chunk_size=800,
    chunk_overlap=100,
    min_chunk_length=50
)

# 处理文档
processed_docs = pipeline.process(
    input_dir="data/raw",
    output_dir="data/processed",
    file_extensions=[".pdf", ".txt", ".md", ".html"]
)

# 查看处理结果
print(f"总共处理了 {len(processed_docs)} 个文档块")
```

## 加载处理后的数据

如果您已经保存了处理后的数据，可以使用以下函数加载它们：

```python
def load_processed_documents(file_path: str) -> List[Document]:
    """
    加载处理后的文档
    
    Args:
        file_path: 文档JSON文件路径
        
    Returns:
        Document对象列表
    """
    with open(file_path, "r", encoding="utf-8") as f:
        docs_dicts = json.load(f)
    
    documents = []
    for doc_dict in docs_dicts:
        doc = Document(
            page_content=doc_dict["content"],
            metadata=doc_dict["metadata"]
        )
        documents.append(doc)
    
    return documents
```

## 下一步

在本章中，我们学习了如何加载、清洗和分割文本数据，为RAG系统准备高质量的文档块。这些处理后的文档块将在下一章中用于创建向量嵌入，这是搭建高效检索系统的基础。 