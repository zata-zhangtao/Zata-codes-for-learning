"""
RAG系统API服务
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from rag_system import RAGSystem

# 配置日志
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag_api")

# 初始化FastAPI应用
app = FastAPI(
    title="RAG系统API",
    description="检索增强生成系统API",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境应限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化RAG系统
rag_system = RAGSystem(
    docs_dir="documents",
    db_dir="db"
)

# 数据模型
class QueryRequest(BaseModel):
    query: str = Field(..., description="用户查询")
    top_k: Optional[int] = Field(5, description="检索文档数量")
    filter: Optional[Dict[str, Any]] = Field(None, description="过滤条件")
    return_sources: Optional[bool] = Field(True, description="是否返回来源")

class DocumentRequest(BaseModel):
    text: str = Field(..., description="文档文本")
    metadata: Optional[Dict[str, Any]] = Field(None, description="文档元数据")

class CustomPromptRequest(BaseModel):
    template: str = Field(..., description="提示模板")

# API路由
@app.get("/")
async def root():
    """API根路径，返回简单欢迎信息"""
    return {
        "message": "欢迎使用RAG系统API",
        "docs_url": "/docs",
        "status": "operational"
    }

@app.post("/api/query")
async def query(request: QueryRequest):
    """
    处理用户查询
    
    Args:
        request: 查询请求
        
    Returns:
        包含回答和来源的结果
    """
    try:
        result = rag_system.process_query(
            query=request.query,
            top_k=request.top_k,
            filter_dict=request.filter,
            return_sources=request.return_sources
        )
        return result
    except Exception as e:
        logger.error(f"处理查询时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/index")
async def index_documents():
    """
    索引文档目录中的所有文档
    
    Returns:
        索引操作结果
    """
    try:
        result = rag_system.index_documents()
        return result
    except Exception as e:
        logger.error(f"索引文档时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents")
async def add_document(request: DocumentRequest):
    """
    添加单个文档
    
    Args:
        request: 文档请求
        
    Returns:
        添加操作结果
    """
    try:
        result = rag_system.add_document(
            text=request.text,
            metadata=request.metadata
        )
        return result
    except Exception as e:
        logger.error(f"添加文档时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    """
    获取系统统计信息
    
    Returns:
        系统统计信息
    """
    try:
        return rag_system.get_stats()
    except Exception as e:
        logger.error(f"获取统计信息时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clear")
async def clear_data():
    """
    清空系统数据
    
    Returns:
        操作结果
    """
    try:
        return rag_system.clear_data()
    except Exception as e:
        logger.error(f"清空数据时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reload")
async def reload_system():
    """
    重新加载系统
    
    Returns:
        操作结果
    """
    try:
        return rag_system.reload()
    except Exception as e:
        logger.error(f"重新加载系统时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/customize-prompt")
async def customize_prompt(request: CustomPromptRequest):
    """
    自定义提示模板
    
    Args:
        request: 提示模板请求
        
    Returns:
        操作结果
    """
    try:
        rag_system.customize_prompt(request.template)
        return {"status": "success", "message": "成功自定义提示模板"}
    except Exception as e:
        logger.error(f"自定义提示模板时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 主入口
if __name__ == "__main__":
    # 如果数据库目录不存在，尝试索引文档
    if not os.path.exists("db") or os.path.isdir("db") and not os.listdir("db"):
        try:
            logger.info("数据库目录不存在或为空，尝试索引文档...")
            rag_system.index_documents()
        except Exception as e:
            logger.warning(f"自动索引文档失败，但API将继续启动: {str(e)}")
    
    # 启动API服务
    uvicorn.run(
        "api:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    ) 