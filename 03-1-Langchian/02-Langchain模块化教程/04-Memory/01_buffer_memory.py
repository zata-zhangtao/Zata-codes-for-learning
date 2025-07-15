#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ConversationBufferMemory示例 - 最基本的记忆类型
这种记忆类型会存储所有的对话历史
"""

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 从环境变量获取API密钥
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("请设置DASHSCOPE_API_KEY环境变量")

# 初始化大模型
# 使用阿里云百炼平台的API
llm = ChatOpenAI(
    model="qwen-max",  # 或其他百炼平台支持的模型
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    streaming=True
)

def main():
    """演示ConversationBufferMemory的基本用法"""
    
    print("=== ConversationBufferMemory 示例 ===")
    
    # 创建记忆组件
    # ConversationBufferMemory会存储所有的对话历史
    memory = ConversationBufferMemory()
    
    # 创建对话链
    conversation = ConversationChain(
        llm=llm, 
        memory=memory,
        verbose=True  # 设置为True可以看到更多细节
    )
    
    # 第一轮对话
    print("\n第一轮对话:")
    response = conversation.invoke("我叫小明，今年25岁，来自北京。")
    print(f"AI: {response['response']}")
    
    # 第二轮对话
    print("\n第二轮对话:")
    response = conversation.invoke("你还记得我的年龄吗？")
    print(f"AI: {response['response']}")
    
    # 第三轮对话
    print("\n第三轮对话:")
    response = conversation.invoke("我来自哪里？")
    print(f"AI: {response['response']}")
    
    # 查看内存中存储的内容
    print("\n查看记忆中存储的内容:")
    print(memory.load_memory_variables({}))

if __name__ == "__main__":
    main() 