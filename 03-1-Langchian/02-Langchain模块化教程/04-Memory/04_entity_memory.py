#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ConversationEntityMemory示例 - 实体记忆类型
这种记忆类型会跟踪对话中提及的各种实体的信息
"""

from langchain.memory import ConversationEntityMemory
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
    """演示ConversationEntityMemory的基本用法"""
    
    print("=== ConversationEntityMemory 示例 ===")
    
    # 创建实体记忆组件
    # 会自动识别和跟踪对话中的实体
    memory = ConversationEntityMemory(llm=llm)
    
    # 创建对话链
    conversation = ConversationChain(
        llm=llm, 
        memory=memory,
        verbose=True
    )
    
    # 进行一系列对话，提及不同的实体
    
    # 第一轮对话：提及"小王"
    print("\n第一轮对话:")
    response = conversation.invoke("我的同事小王喜欢打篮球，每周末都会去打球。")
    print(f"AI: {response['response']}")
    
    # 第二轮对话：提及"小李"
    print("\n第二轮对话:")
    response = conversation.invoke("我的朋友小李是一名软件工程师，擅长Python编程。")
    print(f"AI: {response['response']}")
    
    # 第三轮对话：提及"北京"
    print("\n第三轮对话:")
    response = conversation.invoke("我计划下个月去北京旅游，北京有很多历史景点。")
    print(f"AI: {response['response']}")
    
    # 查询关于之前提到的实体的信息
    print("\n第四轮对话 (询问关于小王的信息):")
    response = conversation.invoke("你还记得小王的爱好是什么吗？")
    print(f"AI: {response['response']}")
    
    # 查看实体记忆中存储的内容
    print("\n查看实体记忆中存储的内容:")
    print("存储的实体:")
    for entity, info in memory.entity_store.items():
        print(f"- {entity}: {info}")
    
    # 添加更多关于已有实体的信息
    print("\n第五轮对话 (添加关于小李的更多信息):")
    response = conversation.invoke("小李最近在学习机器学习，希望在AI领域有所发展。")
    print(f"AI: {response['response']}")
    
    # 再次查看实体存储的变化
    print("\n再次查看实体记忆中存储的内容:")
    print("存储的实体:")
    for entity, info in memory.entity_store.items():
        print(f"- {entity}: {info}")

if __name__ == "__main__":
    main() 