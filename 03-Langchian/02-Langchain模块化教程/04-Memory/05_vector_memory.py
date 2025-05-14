#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VectorStoreRetrieverMemory示例 - 向量存储记忆类型
这种记忆类型将对话存储到向量数据库中，可以进行语义检索
"""

from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings  # 实际项目中可以替换为对应的嵌入模型
from langchain.chains import ConversationChain
from langchain_community.embeddings import DashScopeEmbeddings

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 从环境变量获取API密钥
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("请设置DASHSCOPE_API_KEY环境变量")

# 由于DashScope可能不提供嵌入API，这里先用OpenAI的嵌入模型进行示例
# 实际使用时可以替换为其他支持的嵌入模型
embedding_model  = DashScopeEmbeddings(
    model="text-embedding-v2",
    # other params...
)
# 初始化大模型
# 使用阿里云百炼平台的API
llm = ChatOpenAI(
    model="qwen-max",  # 或其他百炼平台支持的模型
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    streaming=True
)

def main():
    """演示VectorStoreRetrieverMemory的基本用法"""
    
    print("=== VectorStoreRetrieverMemory 示例 ===")
    print("注意：此示例需要提供支持嵌入(Embedding)功能的API")
    
    # 创建FAISS向量存储
    # 首先需要创建一个空的向量存储
    vectorstore = FAISS.from_texts(
        ["初始化向量存储"], embedding=embedding_model
    )
    
    # 创建检索器
    retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
    
    # 创建向量存储记忆
    memory = VectorStoreRetrieverMemory(
        retriever=retriever,
        memory_key="history"
    )
    
    # 添加一些历史信息到记忆中
    memory.save_context(
        {"input": "我叫王磊，是一名数据科学家。"},
        {"output": "很高兴认识你，王磊！你作为数据科学家一定有很多有趣的经验。"}
    )
    memory.save_context(
        {"input": "我在一家科技公司工作，主要负责机器学习模型的开发。"},
        {"output": "听起来很棒！机器学习是一个非常有前景的领域。你主要使用哪些技术栈？"}
    )
    memory.save_context(
        {"input": "我主要使用Python，TensorFlow和PyTorch进行模型开发。"},
        {"output": "Python是数据科学的主流语言，TensorFlow和PyTorch也是非常强大的深度学习框架。"}
    )
    
    # 创建对话链
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    # 查询之前的对话
    print("\n测试向量记忆检索能力:")
    response = conversation.invoke("请告诉我我的工作是什么？")
    print(f"AI: {response['response']}")
    
    # 添加新的对话
    print("\n添加新的对话内容:")
    response = conversation.invoke("我最近在研究大语言模型的应用。")
    print(f"AI: {response['response']}")
    
    # 再次检索，测试相关性搜索
    print("\n再次测试向量记忆检索能力:")
    response = conversation.invoke("我使用哪些编程语言和框架？")
    print(f"AI: {response['response']}")
    
    # 查看内存中检索到的内容
    print("\n查看向量记忆中检索到的内容:")
    print(memory.load_memory_variables({"input": "技术栈"}))

if __name__ == "__main__":
    main() 