
## Langchain的模块

Langchain是一个强大的框架，用于开发由大型语言模型（LLM）驱动的应用程序。它由以下几个核心模块组成：

### 01-模型 IO（Model IO）
- LLMs：大型语言模型接口，如OpenAI、Anthropic、Llama等
- 嵌入模型（Embeddings）：将文本转换为向量表示
- 聊天模型（Chat Models）：专门用于对话的模型接口
- 输出解析器（Output Parsers）：将模型输出转换为结构化数据

### 02-数据连接（Data Connection）
- 文档加载器（Document Loaders）：从各种来源加载文档
- 文本分割器（Text Splitters）：将长文本分割成可管理的块
- 向量存储（Vector Stores）：存储和检索向量化的文本
- 检索器（Retrievers）：实现高效的信息检索

### 03-链（Chains）
- 顺序链（Sequential Chains）：按顺序执行多个步骤
- 路由链（Router Chains）：根据输入动态选择执行路径
- 转换链（Transform Chains）：处理和转换数据
- 问答链（QA Chains）：专门用于问答任务的链

### 04-记忆（Memory）
- 对话记忆（Conversation Memory）：存储和检索对话历史
- 向量存储记忆（Vector Store Memory）：基于语义相似性的记忆系统
- 实体记忆（Entity Memory）：跟踪对话中提到的实体
- 缓冲记忆（Buffer Memory）：简单的短期记忆机制

### 05-代理（Agents）
- 工具（Tools）：代理可以使用的功能模块
- 工具包（Toolkits）：相关工具的集合
- 代理类型（Agent Types）：不同的决策和规划策略
- 代理执行器（Agent Executors）：控制代理的执行流程

### 06-回调（Callbacks）
- 处理器（Handlers）：处理各种事件的回调函数
- 追踪（Tracing）：跟踪和记录执行过程
- 日志记录（Logging）：记录系统运行信息
- 自定义回调（Custom Callbacks）：根据需求定制回调功能



## 增强langchain能力

### Tools

### LangGraph