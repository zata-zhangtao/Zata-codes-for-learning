# Langchain Memory 模块教程

Memory模块是Langchain框架中的重要组成部分，它使得LLM应用能够记住之前的对话历史，从而实现有状态的对话。本教程将介绍各种Memory类型及其用法，所有示例均使用阿里云DashScope API。

## Memory模块主要组件

Langchain的Memory模块包含多种记忆类型，每种都有不同的存储和检索对话历史的方式：

1. **ConversationBufferMemory**：最基本的记忆类型，存储所有对话历史。
2. **ConversationBufferWindowMemory**：仅保留最近k轮对话。
3. **ConversationSummaryMemory**：使用LLM对长对话进行总结以节省Token。
4. **ConversationSummaryBufferMemory**：结合窗口和摘要的混合记忆类型。
5. **ConversationTokenBufferMemory**：基于token数量而非对话轮数来限制记忆。
6. **ConversationEntityMemory**：跟踪对话中提及的实体信息。
7. **VectorStoreRetrieverMemory**：将对话存储在向量数据库中，实现语义检索。

## 教程文件说明

本目录包含以下Python教程文件：

- `01_buffer_memory.py` - 基础对话缓冲记忆示例
- `02_window_memory.py` - 窗口记忆示例
- `03_summary_memory.py` - 摘要记忆示例
- `04_entity_memory.py` - 实体记忆示例
- `05_vector_memory.py` - 向量存储记忆示例
- `06_combined_memory.py` - 组合多种记忆类型示例

## 环境配置

使用前请确保已设置阿里云百炼API密钥：

```python
import os
os.environ["DASHSCOPE_API_KEY"] = "你的阿里云API密钥"
```

或创建`.env`文件并添加：

```
DASHSCOPE_API_KEY=你的阿里云API密钥
```

## 安装依赖

```bash
pip install langchain langchain-openai python-dotenv
``` 