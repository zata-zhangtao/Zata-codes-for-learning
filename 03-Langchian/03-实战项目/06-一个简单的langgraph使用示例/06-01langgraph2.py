import os
from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage # Added ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

# Ensure DASHSCOPE_API_KEY is set in your environment
if not os.getenv("DASHSCOPE_API_KEY"):
    # You can set it directly for testing if not globally set:
    # os.environ["DASHSCOPE_API_KEY"] = "your_actual_dashscope_api_key"
    # if not os.getenv("DASHSCOPE_API_KEY"): # Check again after potential direct set
    raise ValueError("DASHSCOPE_API_KEY environment variable not set. Please set it before running.")

# **FIX: Rename the tool function**
@tool
def get_weather_updates(query: str) -> str: # Renamed from 'search'
    """
    查询城市当前天气 (Query current weather for a city)
    Use this tool to find out the current weather for a given city.
    """
    print(f"--- Tool 'get_weather_updates' called with query: {query} ---")
    query_lower = query.lower()
    if "上海" in query_lower or 'shanghai' in query_lower:
        return "now is 30 celsius, foggy"
    elif "北京" in query_lower or 'beijing' in query_lower:
        return "now is 20 celsius, sunny"
    else:
        return f"Weather information for {query} not available with this tool. Only Shanghai and Beijing are supported."

# Update the tools list with the new function name
tools = [get_weather_updates] # Use the new tool name

tool_node = ToolNode(tools)

# Initialize the model
model = ChatOpenAI(
    model="qwen-max", # or "qwen-turbo", "qwen-plus" etc. depending on availability
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    # It's good practice to set a temperature if you expect consistent tool use decisions
    temperature=0
)

# Bind the tools to the model
model_with_tools = model.bind_tools(tools)

def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls and len(last_message.tool_calls) > 0:
        print(f"--- LLM decided to use tools: {last_message.tool_calls} ---")
        return "tools"
    print("--- LLM decided NOT to use tools, or no tool calls found. Ending or proceeding. ---")
    return END

def call_model(state: MessagesState):
    messages = state["messages"]
    print(f"--- Calling model with {len(messages)} messages. Last message type: {type(messages[-1])} ---")
    # Ensure we are passing the right message types
    # Dashscope might be particular, ensure HumanMessage, AIMessage, ToolMessage are used correctly.
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END  # Make sure END is explicitly mapped
    }
)
workflow.add_edge("tools", "agent")

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# Use a string for thread_id
thread_id = "chat_thread_42"
config = {"configurable": {"thread_id": thread_id}}

print("\nInvoking for Shanghai...")
shanghai_input = {"messages": [HumanMessage(content="what's the weather in Shanghai?")]}
try:
    final_state_shanghai = app.invoke(shanghai_input, config=config)

    if final_state_shanghai["messages"] and isinstance(final_state_shanghai["messages"][-1], AIMessage):
        result_shanghai = final_state_shanghai["messages"][-1].content
        print(f"Final response for Shanghai: {result_shanghai}")
    else:
        print(f"Unexpected final state for Shanghai: {final_state_shanghai['messages']}")

    # print("\nFull final state for Shanghai:")
    # for i, msg in enumerate(final_state_shanghai["messages"]):
    #     print(f"  MSG {i}: {type(msg).__name__}: ", end="")
    #     if hasattr(msg, 'content'): print(f"Content: '{msg.content}' ", end="")
    #     if hasattr(msg, 'tool_calls') and msg.tool_calls: print(f"ToolCalls: {msg.tool_calls} ", end="")
    #     if hasattr(msg, 'tool_call_id') and msg.tool_call_id: print(f"ToolCallID: {msg.tool_call_id} ", end="")
    #     print()


    print("\nInvoking for Beijing (same thread)...")
    # For subsequent calls in the same thread, LangGraph appends the new message.
    # The 'messages' key in the input to invoke should represent the *new* messages for this turn.
    beijing_input_message = HumanMessage(content="what's the weather in Beijing?")
    final_state_beijing = app.invoke({"messages": [beijing_input_message]}, config=config)


    if final_state_beijing["messages"] and isinstance(final_state_beijing["messages"][-1], AIMessage):
        result_beijing = final_state_beijing["messages"][-1].content
        print(f"Final response for Beijing: {result_beijing}")
    else:
        print(f"Unexpected final state for Beijing: {final_state_beijing['messages']}")

    # print("\nFull final state for Beijing (includes Shanghai interaction):")
    # for i, msg in enumerate(final_state_beijing["messages"]):
    #     print(f"  MSG {i}: {type(msg).__name__}: ", end="")
    #     if hasattr(msg, 'content'): print(f"Content: '{msg.content}' ", end="")
    #     if hasattr(msg, 'tool_calls') and msg.tool_calls: print(f"ToolCalls: {msg.tool_calls} ", end="")
    #     if hasattr(msg, 'tool_call_id') and msg.tool_call_id: print(f"ToolCallID: {msg.tool_call_id} ", end="")
    #     print()

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()