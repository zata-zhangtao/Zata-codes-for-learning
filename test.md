### langchain的事件系统


在 Langchain 中，当你运行一个 LLM、Chain、Agent 或 Tool 时，会经历一系列的步骤（例如，开始调用 LLM，LLM 返回结果，Chain 开始执行，Tool 被调用等）。事件系统（通过回调处理器 CallbackHandler 实现）允许你在这些步骤发生时执行自定义代码。这对于日志记录、监控、调试、数据收集、UI 更新等都非常有用。


