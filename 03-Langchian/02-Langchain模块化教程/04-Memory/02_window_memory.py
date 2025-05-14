#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ConversationBufferWindowMemory示例 - 窗口记忆类型
这种记忆类型只保留最近k轮对话历史
"""

from langchain.memory import ConversationBufferWindowMemory
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
    """演示ConversationBufferWindowMemory的基本用法"""
    
    print("=== ConversationBufferWindowMemory 示例 ===")
    
    # 创建窗口记忆组件
    # k=2表示只保留最近2轮对话
    memory = ConversationBufferWindowMemory(k=2)
    
    # 创建对话链
    conversation = ConversationChain(
        llm=llm, 
        memory=memory,
        verbose=True
    )
    
    # 第一轮对话
    print("\n第一轮对话:")
    response = conversation.invoke("我叫小红，今年28岁。")
    print(f"AI: {response['response']}")
    
    # 第二轮对话
    print("\n第二轮对话:")
    response = conversation.invoke("我喜欢旅游和摄影。")
    print(f"AI: {response['response']}")
    
    # 第三轮对话
    print("\n第三轮对话:")
    response = conversation.invoke("我最近想去云南旅游。")
    print(f"AI: {response['response']}")
    
    # 查看内存中存储的内容
    print("\n查看记忆中存储的内容:")
    print(memory.load_memory_variables({}))
    
    # 第四轮对话 - 测试窗口记忆效果
    print("\n第四轮对话 (测试是否记得第一轮对话内容):")
    response = conversation.invoke("你还记得我的年龄吗？")
    print(f"AI: {response['response']}")
    print("注意：由于窗口大小设置为2，模型应该已经'遗忘'了第一轮中提到的年龄信息")

if __name__ == "__main__":
    main() 