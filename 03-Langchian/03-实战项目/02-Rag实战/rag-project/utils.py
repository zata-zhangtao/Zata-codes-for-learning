"""
工具函数模块，提供通用功能
"""

import os
import time
import logging
import hashlib
from typing import List, Dict, Any, Optional, Union

# 配置日志
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag_utils")

def generate_document_id(content: str, source: str) -> str:
    """
    为文档生成唯一ID
    
    Args:
        content: 文档内容
        source: 文档来源
        
    Returns:
        唯一ID字符串
    """
    content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
    source_hash = hashlib.md5(source.encode("utf-8")).hexdigest()
    return f"{source_hash[:8]}_{content_hash[:8]}"

def measure_time(func):
    """
    函数执行时间测量装饰器
    
    Args:
        func: 要测量的函数
        
    Returns:
        包装后的函数
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"函数 {func.__name__} 执行时间: {end_time - start_time:.4f}秒")
        return result
    return wrapper

def get_file_extension(file_path: str) -> str:
    """
    获取文件扩展名
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件扩展名(小写，不含点)
    """
    _, ext = os.path.splitext(file_path)
    return ext.lower().lstrip(".")

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    将列表分割成固定大小的块
    
    Args:
        lst: 要分割的列表
        chunk_size: 每个块的大小
        
    Returns:
        分割后的块列表
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def safe_get(data: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    安全地从嵌套字典中获取值
    
    Args:
        data: 字典数据
        key_path: 键路径，使用.分隔
        default: 未找到时的默认值
        
    Returns:
        找到的值或默认值
    """
    keys = key_path.split(".")
    result = data
    
    for key in keys:
        if isinstance(result, dict) and key in result:
            result = result[key]
        else:
            return default
            
    return result

def truncate_text(text: str, max_length: int = 100) -> str:
    """
    截断文本并添加省略号
    
    Args:
        text: 要截断的文本
        max_length: 最大长度
        
    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..." 