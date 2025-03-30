# 环境配置

本节将指导您设置一个完整的RAG系统开发环境，包括安装必要的依赖、配置API密钥以及组织项目结构。

## 安装依赖

### 1. Python环境

确保您的系统已安装Python 3.8或更高版本。推荐使用conda或venv创建虚拟环境：

```bash
# 使用conda创建虚拟环境
conda create -n rag-tutorial python=3.10
conda activate rag-tutorial

# 或使用venv
python -m venv rag-env
# Windows激活
rag-env\Scripts\activate
# Linux/Mac激活
source rag-env/bin/activate
```

### 2. 安装基本依赖

创建一个`requirements.txt`文件，包含以下依赖：

```
# LLM和嵌入模型
langchain>=0.1.0
langchain-community>=0.0.15
langchain-openai>=0.0.5
openai>=1.6.0
sentence-transformers>=2.2.2

# 向量数据库
chromadb>=0.4.22
faiss-cpu>=1.7.4

# 数据处理
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
unstructured>=0.11.0
pypdf>=3.17.0
docx2txt>=0.8
python-pptx>=0.6.21
beautifulsoup4>=4.12.0

# Web应用
streamlit>=1.28.0
gradio>=4.12.0
flask>=2.3.0

# 评估
ragas>=0.1.0

# 工具库
tqdm>=4.66.0
matplotlib>=3.7.0
```

安装这些依赖：

```bash
pip install -r requirements.txt
```

## API密钥配置

### OpenAI API密钥

RAG系统通常需要使用大语言模型和嵌入模型，本教程将使用OpenAI的服务。

1. 访问 [OpenAI平台](https://platform.openai.com/) 并注册账号
2. 导航到API密钥部分，创建新的API密钥
3. 在项目中创建`.env`文件并添加密钥：

```
OPENAI_API_KEY=你的OpenAI密钥
```

4. 添加`.env`到`.gitignore`文件以确保不会意外提交密钥

### 加载API密钥的代码

```python
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 获取API密钥
openai_api_key = os.getenv("OPENAI_API_KEY")

# 检查API密钥是否存在
if not openai_api_key:
    raise ValueError("请设置OPENAI_API_KEY环境变量")
```

## 项目结构

为了保持项目的组织性和可维护性，我们采用以下项目结构：

```
rag-tutorial/
│
├── data/                      # 数据目录
│   ├── raw/                   # 原始数据
│   ├── processed/             # 处理后的数据
│   └── embeddings/            # 嵌入向量存储
│
├── src/                       # 源代码
│   ├── data/                  # 数据处理模块
│   │   ├── __init__.py
│   │   ├── loader.py          # 文档加载器
│   │   ├── processor.py       # 文本处理
│   │   └── splitter.py        # 文本分割
│   │
│   ├── embeddings/            # 嵌入模块
│   │   ├── __init__.py
│   │   └── embedder.py        # 文本嵌入器
│   │
│   ├── vector_store/          # 向量数据库模块
│   │   ├── __init__.py
│   │   └── store.py           # 向量存储实现
│   │
│   ├── retriever/             # 检索模块
│   │   ├── __init__.py
│   │   └── retriever.py       # 检索实现
│   │
│   ├── llm/                   # 大语言模型模块
│   │   ├── __init__.py
│   │   └── model.py           # LLM接口
│   │
│   ├── rag/                   # RAG系统模块
│   │   ├── __init__.py
│   │   ├── basic_rag.py       # 基础RAG实现
│   │   └── advanced_rag.py    # 高级RAG实现
│   │
│   └── utils/                 # 工具函数
│       ├── __init__.py
│       └── helpers.py         # 辅助函数
│
├── notebooks/                 # Jupyter笔记本
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_analysis.ipynb
│   └── 03_rag_evaluation.ipynb
│
├── app/                       # 应用程序
│   ├── api/                   # API服务
│   │   ├── __init__.py
│   │   └── app.py             # Flask API
│   │
│   └── ui/                    # 用户界面
│       ├── streamlit_app.py   # Streamlit应用
│       └── gradio_app.py      # Gradio应用
│
├── tests/                     # 测试
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_embeddings.py
│   └── test_rag.py
│
├── config/                    # 配置文件
│   └── config.yaml            # 项目配置
│
├── .env                       # 环境变量（不提交到版本控制）
├── .gitignore                 # Git忽略文件
├── requirements.txt           # 项目依赖
├── setup.py                   # 安装脚本
└── README.md                  # 项目说明
```

## 初始化项目

创建以上目录结构：

```bash
# 在项目根目录中执行
mkdir -p data/{raw,processed,embeddings}
mkdir -p src/{data,embeddings,vector_store,retriever,llm,rag,utils}
mkdir -p notebooks
mkdir -p app/{api,ui}
mkdir -p tests
mkdir -p config

# 创建必要的__init__.py文件
touch src/__init__.py
touch src/data/__init__.py
touch src/embeddings/__init__.py
touch src/vector_store/__init__.py
touch src/retriever/__init__.py
touch src/llm/__init__.py
touch src/rag/__init__.py
touch src/utils/__init__.py
touch app/api/__init__.py
touch tests/__init__.py

# 创建配置文件
touch config/config.yaml
touch .env
touch .gitignore
touch setup.py
```

## 创建setup.py

```python
from setuptools import setup, find_packages

setup(
    name="rag-tutorial",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # 这里列出的依赖项应与requirements.txt相同
        "langchain>=0.1.0",
        "langchain-community>=0.0.15",
        "langchain-openai>=0.0.5",
        "openai>=1.6.0",
        "sentence-transformers>=2.2.2",
        "chromadb>=0.4.22",
        "faiss-cpu>=1.7.4",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "unstructured>=0.11.0",
        "pypdf>=3.17.0",
        "docx2txt>=0.8",
        "python-pptx>=0.6.21",
        "beautifulsoup4>=4.12.0",
        "streamlit>=1.28.0",
        "gradio>=4.12.0",
        "flask>=2.3.0",
        "ragas>=0.1.0",
        "tqdm>=4.66.0",
        "matplotlib>=3.7.0",
        "python-dotenv>=1.0.0",
    ],
    author="RAG实战教程",
    author_email="example@example.com",
    description="检索增强生成(RAG)实战教程",
    keywords="rag, llm, retrieval, embeddings",
    python_requires=">=3.8",
)
```

## 创建.gitignore文件

```
# 环境变量
.env

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# 虚拟环境
venv/
ENV/
rag-env/

# Jupyter Notebook
.ipynb_checkpoints

# 向量数据库和嵌入
data/embeddings/
*.index
*.bin

# 编辑器配置
.vscode/
.idea/
*.swp
*.swo

# 操作系统文件
.DS_Store
Thumbs.db
```

## 配置文件示例

在`config/config.yaml`中添加以下内容：

```yaml
# 模型配置
models:
  llm:
    provider: "openai"
    model_name: "gpt-3.5-turbo"
    temperature: 0.0
    max_tokens: 500
  embedding:
    provider: "openai"
    model_name: "text-embedding-3-small"
    dims: 1536

# 数据处理配置
data:
  chunk_size: 1000
  chunk_overlap: 200
  
# 向量数据库配置
vector_store:
  type: "chroma"
  persist_directory: "data/embeddings/chroma"
  distance_metric: "cosine"

# 检索配置
retrieval:
  top_k: 4
  search_type: "similarity"
  score_threshold: 0.7

# 评估配置
evaluation:
  metrics:
    - "faithfulness"
    - "answer_relevancy"
    - "context_precision"
```

## 下一步

完成环境配置后，您已经准备好开始RAG系统开发的旅程。在下一章，我们将学习如何准备和处理数据，这是构建有效RAG系统的基础步骤。 