#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ConversationSummaryMemory示例 - 摘要记忆类型
这种记忆类型会用LLM对对话历史进行总结，从而节省token
"""

from langchain.memory import ConversationSummaryMemory
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
    """演示ConversationSummaryMemory的基本用法"""
    
    print("=== ConversationSummaryMemory 示例 ===")
    
    # 创建摘要记忆组件
    # llm_model用于生成摘要
    memory = ConversationSummaryMemory(llm=llm)
    
    # 创建对话链
    conversation = ConversationChain(
        llm=llm, 
        memory=memory,
        verbose=True
    )
    
    # 进行一系列对话
    conversation_history = [
        "我叫张伟，是一名大学教授，专注于人工智能研究。",
        "我最近在研究大模型在教育领域的应用。",
        "我想开发一个能够帮助学生个性化学习的AI助手。",
        "这个AI助手需要能够理解学生的学习风格和知识水平。",
        "然后根据学生的情况提供定制化的学习材料和建议。"
    ]
    
    # 进行多轮对话
    for i, message in enumerate(conversation_history):
        print(f"\n第{i+1}轮对话:")
        print(f"用户: {message}")
        response = conversation.invoke(message)
        print(f"AI: {response['response']}")
    
    # 查看摘要内存中存储的内容
    print("\n查看摘要记忆中存储的内容:")
    print(memory.load_memory_variables({}))
    
    # 测试摘要效果
    print("\n测试摘要效果 (询问之前的信息):")
    response = conversation.invoke("你能概括一下我的研究兴趣是什么吗？")
    print(f"AI: {response['response']}")

if __name__ == "__main__":
    main() 