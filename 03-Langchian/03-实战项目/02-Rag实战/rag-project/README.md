# RAG系统实现项目

这是一个基于检索增强生成(Retrieval-Augmented Generation, RAG)技术的完整实现，提供了从文档处理、嵌入生成到向量检索和回答生成的全流程功能。

## 功能特点

- 多种文件格式支持（PDF, TXT, DOCX, HTML, CSV）
- 灵活的文本分块策略
- 可配置的嵌入模型（OpenAI或HuggingFace）
- 高效的向量存储与检索
- 完整的RESTful API
- 详细的日志和错误处理

## 项目结构

```
rag-project/
├── api.py                  # API服务
├── document_processor.py   # 文档处理模块
├── embedding_manager.py    # 嵌入管理模块
├── rag_system.py           # 主RAG系统模块
├── utils.py                # 工具函数
├── vector_store.py         # 向量存储模块
├── requirements.txt        # 依赖项
├── .env                    # 环境变量配置
├── documents/              # 文档目录
└── db/                     # 向量数据库目录
```

## 安装与设置

### 前提条件

- Python 3.8+
- 访问OpenAI API的密钥（或其他嵌入/LLM提供商）

### 安装步骤

1. 克隆仓库（或下载项目文件）:

```bash
git clone https://github.com/yourusername/rag-project.git
cd rag-project
```

2. 创建并激活虚拟环境:

```bash
# 创建虚拟环境
python -m venv rag-env

# 激活环境
# Windows
rag-env\Scripts\activate
# Mac/Linux
source rag-env/bin/activate
```

3. 安装依赖:

```bash
pip install -r requirements.txt
```

4. 配置环境变量:

编辑`.env`文件，设置您的API密钥和其他配置:

```
OPENAI_API_KEY=your_openai_api_key
DEBUG=True
```

## 使用方法

### 准备文档

1. 将要处理的文档（PDF, TXT, DOCX等）放入`documents`目录。

### 启动API服务

```bash
python api.py
```

服务将在http://localhost:8000启动，可以通过http://localhost:8000/docs访问API文档。

### API端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/index` | POST | 索引documents目录中的所有文档 |
| `/api/query` | POST | 处理用户查询 |
| `/api/documents` | POST | 添加单个文档 |
| `/api/stats` | GET | 获取系统统计信息 |
| `/api/clear` | POST | 清空系统数据 |
| `/api/reload` | POST | 重新加载系统 |
| `/api/customize-prompt` | POST | 自定义提示模板 |

### 示例查询

使用curl发送查询:

```bash
curl -X POST "http://localhost:8000/api/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "你的问题", "top_k": 5}'
```

或使用Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/query",
    json={"query": "你的问题", "top_k": 5}
)

print(response.json())
```

## 系统配置

可以在`rag_system.py`中调整以下配置参数:

- `docs_dir`: 文档目录
- `db_dir`: 向量数据库目录
- `embedding_model`: 嵌入模型名称
- `llm_model`: 语言模型名称
- `chunk_size`: 文本分块大小
- `chunk_overlap`: 文本分块重叠大小
- `temperature`: 模型温度参数

## 高级用法

### 自定义提示模板

可以通过API自定义回答生成的提示模板:

```bash
curl -X POST "http://localhost:8000/api/customize-prompt" \
     -H "Content-Type: application/json" \
     -d '{
         "template": "基于以下信息回答问题。如果信息不足，请说明。\n\n信息: {context}\n\n问题: {query}\n\n回答:"
     }'
```

### 过滤查询结果

可以使用元数据过滤查询结果:

```bash
curl -X POST "http://localhost:8000/api/query" \
     -H "Content-Type: application/json" \
     -d '{
         "query": "你的问题", 
         "top_k": 5,
         "filter": {"file_type": "pdf"}
     }'
```

## 扩展与自定义

### 添加新的文档加载器

在`document_processor.py`中的`DocumentProcessor`类中扩展`loader_map`字典，添加新的文件类型支持。

### 使用不同的嵌入模型

在初始化`RAGSystem`时指定不同的嵌入模型:

```python
rag = RAGSystem(
    embedding_model="sentence-transformers/all-mpnet-base-v2"
)
```

### 使用不同的向量数据库

当前实现使用ChromaDB作为向量存储。如需使用其他向量数据库，可以修改`vector_store.py`模块。

## 许可

此项目采用MIT许可证。 