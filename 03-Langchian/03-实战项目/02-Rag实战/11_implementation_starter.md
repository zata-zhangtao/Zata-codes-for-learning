# RAG系统实现起步指南

## 目录
- [环境准备](#环境准备)
- [基础RAG系统实现](#基础RAG系统实现)
- [后续优化方向](#后续优化方向)

## 环境准备

创建一个新的虚拟环境并安装必要的依赖：

```bash
# 创建虚拟环境
python -m venv rag-env

# 激活环境
# Windows
rag-env\Scripts\activate
# Mac/Linux
source rag-env/bin/activate

# 安装基础依赖
pip install -r requirements.txt
```

`requirements.txt`文件内容：

```
langchain>=0.0.267
openai>=0.27.8
chromadb>=0.4.6
sentence-transformers>=2.2.2
pypdf>=3.15.1
tiktoken>=0.4.0
fastapi>=0.100.0
uvicorn>=0.23.1
python-dotenv>=1.0.0
```

## 基础RAG系统实现

以下是一个简单但完整的RAG系统实现，包含所有核心组件：

```python
# rag_system.py

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# 加载环境变量
load_dotenv()

class SimpleRAG:
    def __init__(self, 
                 docs_dir: str = "documents",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 persist_directory: str = "db"):
        """
        初始化RAG系统
        
        Args:
            docs_dir: 文档目录
            chunk_size: 分块大小
            chunk_overlap: 分块重叠部分
            persist_directory: 向量数据库持久化目录
        """
        self.docs_dir = docs_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_directory = persist_directory
        
        # 初始化嵌入模型
        self.embeddings = OpenAIEmbeddings()
        
        # 初始化语言模型
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo"
        )
        
        # 初始化向量存储
        if os.path.exists(persist_directory):
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
        else:
            self.vector_store = None
    
    def load_and_index_documents(self) -> None:
        """加载并索引文档"""
        documents = []
        
        # 加载PDF文件
        for filename in os.listdir(self.docs_dir):
            if filename.endswith(".pdf"):
                file_path = os.path.join(self.docs_dir, filename)
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
        
        # 文本分块
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        splits = text_splitter.split_documents(documents)
        
        # 创建向量存储
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.vector_store.persist()
        
        print(f"已处理并索引{len(splits)}个文档片段")
    
    def query(self, question: str, top_k: int = 4) -> Dict[str, Any]:
        """查询RAG系统
        
        Args:
            question: 用户问题
            top_k: 检索文档数量
            
        Returns:
            包含答案和来源的字典
        """
        if not self.vector_store:
            raise ValueError("请先加载并索引文档")
            
        # 检索相关文档
        retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.get_relevant_documents(question)
        
        # 准备上下文
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 准备提示模板
        template = """
        你是一个知识库助手。使用以下文档片段来回答用户的问题。
        如果你无法从文档中找到答案，请如实说明你无法回答，而不要编造信息。
        
        文档片段:
        {context}
        
        问题: {question}
        
        回答:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # 创建QA链
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        # 执行查询
        result = qa_chain({"query": question})
        
        # 提取来源
        sources = []
        for doc in result["source_documents"]:
            if hasattr(doc, "metadata") and "source" in doc.metadata:
                source = doc.metadata["source"]
                if source not in [s["source"] for s in sources]:
                    sources.append({
                        "source": source,
                        "page": doc.metadata.get("page", None)
                    })
        
        return {
            "answer": result["result"],
            "sources": sources
        }
```

## 使用示例

创建`app.py`文件，实现一个简单的API接口：

```python
# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from rag_system import SimpleRAG
import os

app = FastAPI(title="简单RAG API")

# 初始化RAG系统
rag = SimpleRAG(
    docs_dir="documents",
    persist_directory="db"
)

# 如果向量库目录不存在，加载并索引文档
if not os.path.exists("db"):
    rag.load_and_index_documents()

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 4

class Source(BaseModel):
    source: str
    page: Optional[int] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]

@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        result = rag.query(
            question=request.question,
            top_k=request.top_k
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reload")
async def reload_documents():
    try:
        rag.load_and_index_documents()
        return {"status": "success", "message": "文档重新加载并索引成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 项目结构

创建如下的项目结构：

```
rag-project/
├── .env                  # 环境变量文件
├── requirements.txt      # 依赖文件
├── rag_system.py         # RAG系统实现
├── app.py                # API服务
├── documents/            # 文档目录
│   └── sample.pdf        # 示例文档
└── db/                   # 向量数据库目录
```

`.env`文件内容：

```
OPENAI_API_KEY=your_openai_api_key
```

## 启动服务

```bash
# 运行API服务
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

访问`http://localhost:8000/docs`查看和测试API。

## 后续优化方向

基础RAG系统实现后，可以考虑以下优化方向：

1. **改进文档处理**:
   - 添加更多文档格式支持
   - 实现更智能的分块策略
   - 提取和利用文档结构信息

2. **增强检索能力**:
   - 实现混合检索策略
   - 添加查询重写功能
   - 实现结果重排序

3. **提升生成质量**:
   - 优化提示模板
   - 实现提示工程策略
   - 添加后处理逻辑

4. **增加系统能力**:
   - 实现多模态处理
   - 添加历史对话管理
   - 实现用户反馈机制

5. **部署与规模化**:
   - 添加缓存机制
   - 实现监控和日志系统
   - 优化性能和资源使用

参考本教程其他章节获取详细实现指导。

---

通过这个起步指南，你可以快速构建一个功能完整的基础RAG系统，并根据实际需求逐步添加高级功能和优化措施。关键是理解RAG的核心组件和工作流程，然后在此基础上进行迭代改进。 