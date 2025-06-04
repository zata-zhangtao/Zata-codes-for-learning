# 搜索主题回答系统

基于LangGraph和通义模型的搜索主题回答系统，能够根据用户提供的主题从搜索引擎检索相关信息，将内容存储起来，并能够基于存储的信息回答用户的后续问题。

## 特性

- 主题搜索：连接搜索引擎API，获取相关网页内容
- 内容存储：将内容处理并向量化存储
- 智能问答：基于RAG架构回答用户问题
- 会话管理：支持多轮对话上下文
- 基于LangGraph的工作流：模块化、可扩展的工作流设计

## 技术栈

- **后端框架**：FastAPI
- **LLM**：通义大模型（Tongyi）
- **工作流引擎**：LangGraph
- **向量数据库**：Chroma
- **数据库**：MongoDB
- **搜索API**：SerpAPI/Google Search API

## 快速开始

### 前置条件

- Python 3.9+
- MongoDB
- 通义API密钥
- 搜索引擎API密钥

### 安装

1. 克隆仓库

```bash
git clone https://github.com/yourusername/topic-search-qa.git
cd topic-search-qa
```

2. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # 在Windows上使用: venv\Scripts\activate
```

3. 安装依赖

```bash
pip install -r requirements.txt
```

4. 配置环境变量

复制示例环境文件并填入您的API密钥

```bash
cp sample.env .env
```

编辑`.env`文件，填入必要的API密钥和配置。

### 运行应用

```bash
uvicorn app.main:app --reload
```

应用将在`http://localhost:8000`上运行。

## API使用说明

### 创建新主题

```bash
curl -X POST "http://localhost:8000/api/topics" \
     -H "Content-Type: application/json" \
     -d '{"topic": "人工智能的发展历程"}'
```

### 提问

```bash
curl -X POST "http://localhost:8000/api/topics/{topic_id}/questions" \
     -H "Content-Type: application/json" \
     -d '{"question": "人工智能的冬夏交替是什么？"}'
```

### 获取主题历史

```bash
curl -X GET "http://localhost:8000/api/topics/{topic_id}/history"
```

## 项目结构

```
app/
├── api/                    # API接口
│   └── routers/            # 路由定义
├── core/                   # 核心配置
├── graph/                  # LangGraph工作流定义
│   ├── nodes.py            # 工作流节点
│   └── workflow.py         # 工作流图定义
├── models/                 # 数据模型
├── schemas/                # Pydantic模式
├── services/               # 服务层
│   ├── db_service.py       # 数据库服务
│   ├── llm_service.py      # 大模型服务
│   ├── search_service.py   # 搜索服务
│   ├── topic_service.py    # 主题服务
│   ├── question_service.py # 问题服务
│   └── vector_store_service.py  # 向量存储服务
└── main.py                 # 应用入口
```

## 开发

### 添加新的工作流节点

1. 在`app/graph/nodes.py`中添加新节点函数
2. 在`app/graph/workflow.py`中注册节点并设置边

### 自定义搜索引擎

修改`app/services/search_service.py`以集成不同的搜索引擎API。

## 贡献

欢迎贡献代码、报告问题或提出建议！

## 许可证

MIT 