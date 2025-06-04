from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatTongyi

# MessagesState 是 LangGraph 中的一个内置状态管理类，位于 langgraph.graph 模块，
# 用于处理消息集合，常用于对话型工作流或聊天机器人等场景。以下是对 MessagesState 用法的详细中文介绍，
# 包括其用途、结构和实际示例。

# 定义处理消息并生成回复的节点
def chatbot_node(state: MessagesState):
    # 获取当前状态中的消息
    messages = state["messages"]
    
    # 初始化 LLM（例如 OpenAI 的聊天模型）
    llm = ChatTongyi(model="qwen-max")
    
    # 根据对话历史生成回复
    response = llm.invoke(messages)
    
    # 返回更新后的状态，包含新的 AI 消息
    return {"messages": [response]}

# 创建使用 MessagesState 的 StateGraph
graph = StateGraph(MessagesState)

# 添加聊天机器人节点
graph.add_node("chatbot", chatbot_node)

# 定义边：START -> chatbot -> END
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

# 编译图
app = graph.compile()

# 使用初始消息运行图
initial_state = {"messages": [HumanMessage(content="你好！今天能帮我什么？")]}
result = app.invoke(initial_state)

# 打印最终状态
for message in result["messages"]:
    print(f"{message.__class__.__name__}: {message.content}")
