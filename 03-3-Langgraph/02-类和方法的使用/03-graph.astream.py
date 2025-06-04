import asyncio
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END

# 1. 定义状态 (Define State)
# 状态是图在执行过程中传递的数据结构。
class MyState(TypedDict):
    input_value: str
    result_a: str
    result_b: str
    history: Sequence[str] # 用于追踪执行顺序

# 2. 定义图节点 (Define Graph Nodes)
# 节点是图中的处理单元，它们接收状态，执行操作，并返回对状态的更新。

async def node_A(state: MyState):
    print("--- Executing Node A ---")
    await asyncio.sleep(1) # 模拟异步操作
    value = state["input_value"]
    processed_value = f"Node A processed: {value}"
    print(f"Node A output: {processed_value}")
    return {"result_a": processed_value, "history": state["history"] + ["Node A"]}

async def node_B(state: MyState):
    print("--- Executing Node B ---")
    await asyncio.sleep(1) # 模拟异步操作
    value = state["result_a"] # 从 node_A 获取结果
    processed_value = f"Node B processed: {value}"
    print(f"Node B output: {processed_value}")
    return {"result_b": processed_value, "history": state["history"] + ["Node B"]}

# 3. 构建图 (Build the Graph)
workflow = StateGraph(MyState)

# 添加节点
workflow.add_node("A", node_A)
workflow.add_node("B", node_B)

# 定义边 (如何从一个节点到另一个节点)
workflow.add_edge("A", "B")

# 设置入口和出口
workflow.set_entry_point("A")
workflow.set_finish_point("B") # 或者使用 END

# 编译图
graph = workflow.compile()

# 4. 定义初始状态和配置 (Define Initial State and Config)
initial_state = {"input_value": "Hello LangGraph!", "history": []}
config = {"recursion_limit": 5} # 示例配置

# 5. 异步流式执行 (Asynchronously Stream Execution with stream_mode="values")
async def main():
    print("Starting graph execution with stream_mode='values'...\n")
    async for step_output in graph.astream(
        input=initial_state, config=config, stream_mode="values"
    ):
        # 当 stream_mode="values" 时, step_output 是每个被调用节点返回的字典。
        # 注意：它不是完整的状态对象，而是该步骤中刚刚执行的节点的输出。
        # 通常，这会是一个字典，包含了该节点更新的状态字段。
        print(f"--- Streamed Output (Value from Node) ---")
        print(f"{step_output}\n")

    print("\n--- Final State (after full execution, if needed) ---")
    # 如果需要完整的最终状态，可以单独调用 ainvoke
    final_state = await graph.ainvoke(input=initial_state, config=config)
    print(final_state)

if __name__ == "__main__":
    asyncio.run(main())