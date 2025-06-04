from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.chat_models import ChatTongyi
from langgraph.graph import StateGraph, END

# 1. 定义状态
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# 2. 定义LLM和节点
# 确保你已经设置了 OPENAI_API_KEY 环境变量
llm = ChatTongyi(model="qwen-max")

def call_language_model(state: AgentState):
    print("--- 调用语言模型 ---")
    messages = state['messages']
    response = llm.invoke(messages)
    print(f"LLM 响应: {response.content}")
    return {"messages": [response]}

# 3. 构建图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("language_model", call_language_model)

# 设置入口点
workflow.set_entry_point("language_model")

# 添加结束点 (这里简化，LLM调用后就结束本次交互)
workflow.add_edge("language_model", END)

# 编译图
app = workflow.compile()

# 4. 运行对话
print("开始对话 (输入 '退出' 来结束)")
while True:
    user_input_text = input("你: ")
    if user_input_text.lower() == '退出':
        print("对话结束.")
        break

    inputs = {"messages": [HumanMessage(content=user_input_text)]}
    # LangGraph 的 stream 方法会处理状态的累积 (因为我们在 AgentState 中定义了 operator.add)
    # 但对于一个持续的对话，你通常会在每次调用 invoke 或 stream 之前，将新的用户消息添加到历史消息中
    # 在这个简单的循环中，我们每次都传入包含当前用户消息的 inputs
    # 为了实现持续对话，我们需要从上一次的状态中获取 'messages' 并添加新的 HumanMessage

    # 更合适的做法是维护一个完整的对话历史
    # current_messages = [] # 在循环外初始化
    # ...
    # current_messages.append(HumanMessage(content=user_input_text))
    # inputs = {"messages": current_messages}
    # for s_event in app.stream(inputs):
    #    # ... 更新 current_messages ...
    #    if "__end__" not in s_event:
    #        current_messages = s_event[next(iter(s_event))]['messages']


    # 为了简单起见，以下示例每次都只处理当前输入，不显式维护外部历史记录
    # LangGraph 的 StateGraph 和 operator.add 会在图内部处理消息的累积（如果图被设计为多次调用并传递状态）
    # 但对于一个典型的聊天机器人循环，你需要从上一次运行的最终状态中获取消息。

    # 简单的运行和打印结果
    for event in app.stream(inputs):
        for key, value in event.items():
            if key != "__end__":
                print(f"节点 '{key}' 的输出: {value}")
                if 'messages' in value and isinstance(value['messages'][-1], AIMessage):
                    print(f"机器人: {value['messages'][-1].content}")
        print("---")