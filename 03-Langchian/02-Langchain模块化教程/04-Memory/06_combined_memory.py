#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
组合记忆类型示例 - 结合多种记忆类型的使用
使用CombinedMemory可以同时使用多种不同类型的记忆
"""

from langchain.memory import CombinedMemory, ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
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
    """演示如何组合多种记忆类型"""
    
    print("=== 组合记忆类型 (CombinedMemory) 示例 ===")
    
    # 创建不同类型的记忆组件
    
    # 1. 常规对话缓冲记忆 - 存储短期记忆
    conv_memory = ConversationBufferMemory(
        memory_key="chat_history",  # 这里的键需要与prompt模板中的变量名匹配
        input_key="input"  # 指定输入键
    )
    
    # 2. 摘要记忆 - 存储长期摘要
    summary_memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="summary",  # 这里的键需要与prompt模板中的变量名匹配
        input_key="input"  # 指定输入键
    )
    
    # 组合这些记忆
    combined_memory = CombinedMemory(
        memories=[conv_memory, summary_memory]
    )
    
    # 创建自定义提示模板，确保包含所有记忆键
    template = """以下是与AI助手的友好对话。
    
对话摘要：{summary}

对话历史：
{chat_history}

人类: {input}
AI助手:"""

    prompt = PromptTemplate(
        input_variables=["summary", "chat_history", "input"],
        template=template
    )
    
    # 创建对话链
    conversation = ConversationChain(
        llm=llm,
        memory=combined_memory,
        prompt=prompt,
        verbose=True
    )
    
    # 进行多轮对话
    print("\n第一轮对话:")
    response = conversation.invoke(
        {"input": "我叫李明，是一名软件开发者，主要用Java和Python开发企业应用。"}
    )
    print(f"AI: {response['response']}")
    
    print("\n第二轮对话:")
    response = conversation.invoke(
        {"input": "我最近在学习微服务架构和容器化技术。"}
    )
    print(f"AI: {response['response']}")
    
    print("\n第三轮对话:")
    response = conversation.invoke(
        {"input": "我希望在未来能够成为一名技术架构师。"}
    )
    print(f"AI: {response['response']}")
    
    # 查看不同类型记忆中存储的内容
    print("\n查看对话缓冲记忆中的内容:")
    print(conv_memory.load_memory_variables({}))
    
    print("\n查看摘要记忆中的内容:")
    print(summary_memory.load_memory_variables({}))
    
    # 测试记忆效果
    print("\n测试记忆效果 (询问关于用户的信息):")
    response = conversation.invoke(
        {"input": "你能总结一下我的技术背景和职业目标吗？"}
    )
    print(f"AI: {response['response']}")

if __name__ == "__main__":
    main() 