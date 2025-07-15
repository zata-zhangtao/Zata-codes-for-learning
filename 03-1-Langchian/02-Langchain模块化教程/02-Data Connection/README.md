# Langchain Data Connection 模块教程

本目录包含Langchain的数据连接(Data Connection)模块教程，该模块是Langchain框架的核心组件之一，负责连接LLM与外部数据源。

## 模块概述

Data Connection模块主要包括四个核心组件:

1. **文档加载器(Document Loaders)**: 从各种来源加载文档
2. **文本分割器(Text Splitters)**: 将长文本分割成可管理的块
3. **向量存储(Vector Stores)**: 存储和检索向量化的文本
4. **检索器(Retrievers)**: 实现高效的信息检索

这些组件共同构成了RAG(检索增强生成)应用的基础，使LLM能够访问和利用外部知识。

## 教程文件

本目录包含以下教程文件:

| 文件名 | 描述 |
|-------|------|
| [01-文档加载器.py](./01-文档加载器.py) | 介绍如何使用Langchain加载各种格式的文档(文本、PDF、CSV、HTML等) |
| [02-文本分割器.py](./02-文本分割器.py) | 演示各种文本分割策略，包括字符分割、递归分割、Token分割和Markdown分割 |
| [03-向量存储.py](./03-向量存储.py) | 讲解向量数据库的使用，包括Chroma和FAISS等工具 |
| [04-检索Retrieval.py](./04-检索Retrieval.py) | 探索各种检索策略，如相似性搜索、MMR、上下文压缩等 |
| [05-整合应用.py](./05-整合应用.py) | 综合运用以上组件构建完整的RAG应用 |
| [06-网页爬虫.py](./06-网页爬虫.py) | 介绍如何使用Langchain进行网页抓取和内容处理，将网页数据整合到RAG应用中 |

## 使用方法

1. 确保安装了所需的依赖:
```bash
pip install langchain langchain-community langchain-text-splitters faiss-cpu sentence-transformers
```

2. 对于网页爬虫功能还需要安装:
```bash
pip install requests beautifulsoup4 selenium webdriver-manager
```

3. 对于阿里云API:
```python
import os
os.environ["ALIBABA_API_KEY"] = "你的API密钥"
```

4. 运行相应的教程文件:
```bash
python 01-文档加载器.py
```

## 核心概念

- **文档(Document)**: Langchain中的基本数据单位，包含文本内容和元数据
- **分块(Chunking)**: 将大型文档分解为更小的部分，以适应向量存储和检索的需要
- **嵌入(Embedding)**: 将文本转换为向量表示，以便进行语义搜索
- **检索增强生成(RAG)**: 结合检索和生成的技术，使LLM能够基于外部知识生成回答
- **网页爬虫(Web Scraping)**: 从互联网自动获取和处理信息的技术，为LLM提供最新数据

## 注意事项

- 对于中文文本处理，推荐使用专门的中文嵌入模型，如`shibing624/text2vec-base-chinese`
- 分块大小和重叠度会显著影响检索质量，需要根据具体应用场景调整
- 向量存储需要足够的内存资源，特别是对于大型文档集合
- 进行网页爬虫时，请遵守网站的robots.txt规则，合理设置请求延迟

## 延伸阅读

- [Langchain官方文档 - Data Connection](https://python.langchain.com/docs/modules/data_connection/)
- [RAG应用开发最佳实践](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [网页爬虫最佳实践](https://python.langchain.com/docs/modules/data_connection/document_loaders/web_base/) 