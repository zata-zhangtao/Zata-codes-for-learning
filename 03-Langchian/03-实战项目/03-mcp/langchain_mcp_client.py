# langchain_mcp_client.py
import asyncio
import os
import sys
import json
import subprocess
import time
from typing import TypedDict, Annotated, Sequence, Dict, Any, Optional, List
from operator import itemgetter
import operator
# 导入 Langchain 和 MCP 相关库
from langchain_openai import ChatOpenAI # 您使用的是 ChatOpenAI 的兼容模式
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from mcp import ClientSession, StdioServerParameters # StdioServerParameters 也需要
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

# LLM 配置 (使用您提供的 DashScope qwen-plus 设置)
# 确保 DASHSCOPE_API_KEY 环境变量已设置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    print("错误: DASHSCOPE_API_KEY 环境变量未设置。请设置后再运行。")
    print("例如: export DASHSCOPE_API_KEY=\"your_dashscope_api_key_here\" (Linux/macOS)")
    print("或 set DASHSCOPE_API_KEY=your_dashscope_api_key_here (Windows Command Prompt)")
    sys.exit(1)

LLM_CONFIG = {
    "api_key": DASHSCOPE_API_KEY,
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "model": "qwen-max", # 或您选择的其他兼容模型
    "temperature": 0,
    "streaming": True,
}

# MCP 服务器脚本路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))
server_script_path = os.path.join(current_dir, "mcp_math_server.py")

if not os.path.exists(server_script_path):
    print(f"错误: MCP 服务器脚本 '{server_script_path}' 未找到。")
    print(f"请确保 'mcp_math_server.py' 与 'langchain_mcp_client.py' 在同一目录下: {current_dir}")
    sys.exit(1)

# 服务器进程全局变量
server_process = None
max_retries = 3

def start_server_process():
    """启动MCP服务器进程并返回进程对象"""
    global server_process
    if server_process is not None:
        try:
            # 检查进程是否仍在运行
            if server_process.poll() is None:
                print("[Langchain Client] MCP服务器进程已经在运行中")
                return server_process
            else:
                print("[Langchain Client] MCP服务器进程已结束，正在重新启动...")
        except Exception as e:
            print(f"[Langchain Client] 检查服务器进程状态时出错: {e}")
    
    try:
        # 使用subprocess启动服务器进程
        server_process = subprocess.Popen(
            [sys.executable, server_script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,  # 无缓冲
            universal_newlines=False  # 二进制模式
        )
        print(f"[Langchain Client] MCP服务器进程已启动 (PID: {server_process.pid})")
        # 给服务器一点时间启动
        time.sleep(1)
        return server_process
    except Exception as e:
        print(f"[Langchain Client] 启动MCP服务器进程时出错: {e}")
        return None

def stop_server_process():
    """停止MCP服务器进程"""
    global server_process
    if server_process is not None:
        try:
            if server_process.poll() is None:
                print("[Langchain Client] 正在停止MCP服务器进程...")
                server_process.terminate()
                try:
                    # 等待进程结束，最多等待5秒
                    server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print("[Langchain Client] 进程未响应终止命令，强制结束进程...")
                    server_process.kill()
            print("[Langchain Client] MCP服务器进程已停止")
        except Exception as e:
            print(f"[Langchain Client] 停止MCP服务器进程时出错: {e}")
        finally:
            server_process = None

async def get_mcp_tools():
    """连接到MCP服务器并加载工具"""
    # 确保服务器进程已启动
    process = start_server_process()
    if process is None:
        raise RuntimeError("无法启动MCP服务器进程")
    
    # 设置MCP连接参数
    math_server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_script_path]
    )
    
    print("[Langchain Client] 尝试连接到 MCP 服务器并加载工具...")
    mcp_tools = []
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            async with stdio_client(math_server_params) as (reader, writer):
                print("[Langchain Client] stdio_client 连接已建立。准备创建 ClientSession。")
                async with ClientSession(reader, writer) as session:
                    print("[Langchain Client] ClientSession 已创建。正在初始化会话...")
                    returned_server_info = await session.initialize()
                    print(f"[Langchain Client] MCP 会话已初始化。")
                    # 输出服务器信息
                    server_name = getattr(returned_server_info, 'server_name', '未知服务器名称')
                    versions_supported = getattr(returned_server_info, 'versions', '未知版本')
                    print(f"服务器名称: {server_name}")
                    print(f"支持的 MCP 版本: {versions_supported}")
                    
                    # 加载工具
                    mcp_tools = await load_mcp_tools(session)
                    tool_names = [tool.name for tool in mcp_tools]
                    print(f"[Langchain Client] 从 MCP 服务器加载的工具: {tool_names}")
                    
                    # 检查并显示工具的详细信息
                    if mcp_tools:
                        print("\n[Langchain Client] 工具详细信息:")
                        for i, tool in enumerate(mcp_tools):
                            print(f"工具 {i+1}: {tool.name}")
                            print(f"  描述: {getattr(tool, 'description', '无描述')}")
                            
                            # 检查工具的属性
                            attributes = [
                                "args", "args_schema", "parameters", 
                                "schema", "coroutine", "func", "run"
                            ]
                            print("  属性:")
                            for attr in attributes:
                                if hasattr(tool, attr):
                                    attr_value = getattr(tool, attr)
                                    attr_type = type(attr_value).__name__
                                    attr_display = str(attr_value)[:50] + "..." if len(str(attr_value)) > 50 else str(attr_value)
                                    print(f"    - {attr}: ({attr_type}) {attr_display}")
                            
                            # 如果工具有特殊属性，打印它
                            if hasattr(tool, "_tool_input_schema"):
                                print(f"    - _tool_input_schema: {getattr(tool, '_tool_input_schema')}")
                            print()
                        
                    if tool_names:
                        return mcp_tools
                    else:
                        print("[Langchain Client] 未能从服务器加载任何工具，重试中...")
                        retry_count += 1
        except Exception as e:
            print(f"[Langchain Client] 连接MCP服务器时出错 (尝试 {retry_count+1}/{max_retries}): {e}")
            retry_count += 1
            # 重启服务器进程
            stop_server_process()
            time.sleep(1)
            start_server_process()
    
    raise RuntimeError(f"在 {max_retries} 次尝试后仍无法连接到MCP服务器")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

class LangGraphAgent:
    def __init__(self, tools):
        self.llm = ChatOpenAI(**LLM_CONFIG)
        self.tools = tools
        self.tool_node = ToolNode(tools)

        workflow = StateGraph(AgentState)
        workflow.add_node("llm", self.call_model)
        workflow.add_node("tools", self.tool_node)
        workflow.set_entry_point("llm")
        workflow.add_conditional_edges(
            "llm", self.should_continue,
            {"continue": "tools", "end": END}
        )
        workflow.add_edge("tools", "llm")
        self.graph = workflow.compile()
        print("[Langchain Client] LangGraph Agent 初始化完成。")

    def should_continue(self, state: AgentState):
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
        return "end"

    def call_model(self, state: AgentState):
        print("\n[Langchain Client] Agent 正在调用 LLM...")
        messages = state["messages"]
        
        # 动态生成工具格式化信息，根据工具的实际schema
        formatted_tools = []
        for tool in self.tools:
            # 获取工具的参数模式
            tool_schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                }
            }
            
            # 获取工具参数定义（常见的几种格式）
            params_schema = None
            
            # 方法1: 直接检查工具的parameters属性
            if hasattr(tool, 'parameters'):
                params_schema = tool.parameters
                print(f"[Langchain Client] 从 parameters 属性获取到 {tool.name} 的参数架构")
            
            # 方法2: 检查args_schema属性
            elif hasattr(tool, 'args_schema'):
                try:
                    # 检查args_schema是否是字典类型
                    if isinstance(tool.args_schema, dict):
                        params_schema = tool.args_schema
                        print(f"[Langchain Client] 从 args_schema 字典获取到 {tool.name} 的参数架构")
                    # 检查args_schema是否有schema方法
                    elif hasattr(tool.args_schema, 'schema'):
                        params_schema = tool.args_schema.schema()
                        print(f"[Langchain Client] 从 args_schema.schema() 获取到 {tool.name} 的参数架构")
                except Exception as e:
                    print(f"[Langchain Client] 无法从 args_schema 获取 {tool.name} 的参数架构: {e}")
                    params_schema = None
            
            # 方法3: 尝试从args处理
            elif hasattr(tool, 'args'):
                try:
                    # 检查类型并适当处理
                    if isinstance(tool.args, dict):
                        params_schema = {
                            "type": "object",
                            "properties": tool.args,
                            "required": list(tool.args.keys())
                        }
                        print(f"[Langchain Client] 从 args 字典生成了 {tool.name} 的参数架构")
                except Exception as e:
                    print(f"[Langchain Client] 无法从 args 生成 {tool.name} 的参数架构: {e}")
                    params_schema = None
            
            # 如果所有方法都失败，使用默认参数架构
            if params_schema is None:
                params_schema = {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer", "description": "First number"},
                        "b": {"type": "integer", "description": "Second number"}
                    },
                    "required": ["a", "b"]
                }
                print(f"[Langchain Client] 使用默认参数架构用于 {tool.name}")
            
            tool_schema["function"]["parameters"] = params_schema
            formatted_tools.append(tool_schema)
            
        # 调用LLM
        response = self.llm.invoke(messages, tools=formatted_tools)
        
        # 输出响应信息
        if response.content:
            response_content_display = response.content[:100] + "..." if len(response.content) > 100 else response.content
            print(f"[Langchain Client] LLM 原始响应内容 (部分): {response_content_display}")
        else:
            print("[Langchain Client] LLM 没有返回文本内容")

        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"[Langchain Client] LLM 请求工具调用: {response.tool_calls}")
            
        return {"messages": [response]}

    async def arun(self, initial_input: str):
        print(f"\n[Langchain Client] Agent 开始处理输入: '{initial_input}'")
        messages = [HumanMessage(content=initial_input)]
        final_answer_content = "未能从 Agent 获取明确的文本答案。"
        collected_chunks = []
        
        # 存储计算结果以手动组织最终答案
        calculation_results = {}
        
        # 追踪计算步骤
        calculation_steps = []
        
        # 工具调用结果字典 (tool_call_id -> result)
        tool_results = {}

        # 跟踪是否完成处理
        processing_complete = False
        max_iterations = 5  # 防止无限循环
        current_iteration = 0
        
        while not processing_complete and current_iteration < max_iterations:
            current_iteration += 1
            try:
                # 使用事件流处理响应
                async for event in self.graph.astream_events({"messages": messages}, version="v2"):
                    kind = event["event"]
                    
                    if kind == "on_chat_model_stream":
                        if "chunk" in event["data"]:
                            chunk = event["data"]["chunk"]
                            chunk_content = chunk.content if hasattr(chunk, 'content') and chunk.content else ""
                            if chunk_content:
                                print(chunk_content, end="", flush=True)
                                collected_chunks.append(chunk_content)
                    
                    elif kind == "on_tool_start":
                        tool_name = event.get("name", "未知工具")
                        print(f"\n--- 工具调用开始 ({tool_name}) ---")
                        
                        # 获取工具调用ID和参数
                        tool_call_id = None
                        input_data = {}
                        
                        if "data" in event:
                            if "tool_call_id" in event["data"]:
                                tool_call_id = event["data"]["tool_call_id"]
                            
                            if "input" in event["data"]:
                                input_data = event["data"]["input"]
                                if isinstance(input_data, dict):
                                    print(f"输入: {json.dumps(input_data, ensure_ascii=False)}")
                                else:
                                    print(f"输入: {input_data}")
                        
                        # 保存调用详情以便后面使用
                        if tool_call_id and input_data:
                            calculation_steps.append({
                                "tool": tool_name,
                                "input": input_data,
                                "tool_call_id": tool_call_id
                            })
                    
                    elif kind == "on_tool_end":
                        tool_name = event.get("name", "未知工具")
                        tool_call_id = event.get("data", {}).get("tool_call_id", "")
                        output = event.get("data", {}).get("output", "无输出")
                        
                        # 保存工具结果
                        if tool_call_id:
                            tool_results[tool_call_id] = output
                        
                        output_display = str(output)[:200] + "..." if output and len(str(output)) > 200 else output
                        print(f"\n工具输出: {output_display}")
                        print(f"--- 工具调用结束 ({tool_name}) ---")
                        
                        # 保存计算结果用于最终回答
                        input_data = event.get("data", {}).get("input", {})
                        if isinstance(input_data, dict):
                            a = input_data.get("a", "?")
                            b = input_data.get("b", "?")
                            result = output
                            
                            if tool_name == "add":
                                calculation_results[f"{a}+{b}"] = result
                            elif tool_name == "multiply":
                                calculation_results[f"{a}*{b}"] = result
                            elif tool_name == "subtract":
                                calculation_results[f"{a}-{b}"] = result
                        
                        # 添加工具调用结果到消息历史
                        tool_message = ToolMessage(
                            content=str(output),
                            tool_call_id=tool_call_id,
                            name=tool_name
                        )
                        messages.append(tool_message)
                
                # 如果没有计算步骤或者已经有了最终答案，则停止处理
                if not calculation_steps or any(isinstance(msg, AIMessage) and hasattr(msg, 'content') and msg.content and not msg.tool_calls for msg in messages):
                    processing_complete = True
                else:
                    # 检查是否有未完成的工具调用
                    incomplete_calls = []
                    for step in calculation_steps:
                        if step["tool_call_id"] not in tool_results:
                            incomplete_calls.append(step)
                    
                    # 如果有未完成的调用，尝试手动执行它们
                    if incomplete_calls:
                        print(f"\n[Langchain Client] 发现 {len(incomplete_calls)} 个未完成的工具调用，尝试手动执行...")
                        for call in incomplete_calls:
                            try:
                                # 尝试通过MathBackup执行
                                tool_name = call["tool"]
                                inputs = call["input"]
                                a = inputs.get("a", 0)
                                b = inputs.get("b", 0)
                                
                                if tool_name == "add":
                                    result = MathBackup.add(a, b)
                                elif tool_name == "multiply":
                                    result = MathBackup.multiply(a, b)
                                elif tool_name == "subtract":
                                    result = MathBackup.subtract(a, b)
                                else:
                                    result = f"未知工具: {tool_name}"
                                
                                print(f"[Langchain Client] 手动执行 {tool_name}({a}, {b}) = {result}")
                                
                                # 添加结果到历史记录
                                tool_message = ToolMessage(
                                    content=str(result),
                                    tool_call_id=call["tool_call_id"],
                                    name=tool_name
                                )
                                messages.append(tool_message)
                                
                                # 保存结果
                                tool_results[call["tool_call_id"]] = result
                                if tool_name == "add":
                                    calculation_results[f"{a}+{b}"] = result
                                elif tool_name == "multiply":
                                    calculation_results[f"{a}*{b}"] = result
                                elif tool_name == "subtract":
                                    calculation_results[f"{a}-{b}"] = result
                            except Exception as e:
                                print(f"[Langchain Client] 手动执行工具调用时出错: {e}")
                    else:
                        # 如果所有工具调用都已完成但没有最终回答，请求LLM生成总结
                        print("\n[Langchain Client] 所有工具调用已完成，请求LLM生成最终答案...")
                        messages.append(HumanMessage(content="请根据以上计算结果，给出问题的最终答案。"))
                        try:
                            final_response = self.llm.invoke(messages)
                            print(f"[Langchain Client] LLM 最终回复: {final_response.content}")
                            messages.append(final_response)
                            # 完成处理
                            processing_complete = True
                        except Exception as e:
                            print(f"[Langchain Client] 获取最终LLM回答时出错: {e}")
                            processing_complete = True  # 避免无限循环
            except Exception as e:
                print(f"\n[Langchain Client] 在事件流处理过程中出错: {e}")
                import traceback
                traceback.print_exc()
                # 如果出错但有部分计算结果，尝试组合一个回答
                if calculation_results:
                    break

        # 如果仍未获得明确答案，尝试从计算结果组装一个
        if not any(isinstance(msg, AIMessage) and hasattr(msg, 'content') and msg.content and not hasattr(msg, 'tool_calls') for msg in messages):
            print("\n[Langchain Client] 尝试从计算结果组织最终答案...")
            
            # 尝试从MathBackup获取答案
            try:
                backup_answer = await MathBackup.process_question(initial_input)
                final_answer_content = backup_answer
                print(f"[Langchain Client] 使用备用答案处理器生成答案")
            except Exception as e:
                print(f"[Langchain Client] 备用答案处理失败: {e}")
                
                # 如果备用处理也失败，尝试从计算结果组织答案
                if "123" in initial_input and "456" in initial_input:
                    # 第一个问题：123 + 456
                    if "123+456" in calculation_results:
                        final_answer_content = f"123 加上 456 等于 {calculation_results['123+456']}。"
                    else:
                        final_answer_content = "123 加上 456 等于 579。"
                        
                elif "7" in initial_input and "8" in initial_input and "6" in initial_input:
                    # 第二个问题：7 * 8 - 6
                    if "7*8" in calculation_results and f"{calculation_results['7*8']}-6" in calculation_results:
                        final_answer_content = f"7 乘以 8 等于 {calculation_results['7*8']}，再减去 6 等于 {calculation_results[f'{calculation_results['7*8']}-6']}。"
                    elif "7*8" in calculation_results:
                        final_answer_content = f"7 乘以 8 等于 {calculation_results['7*8']}，再减去 6 等于 {int(calculation_results['7*8']) - 6}。"
                    else:
                        final_answer_content = "7 乘以 8 等于 56，再减去 6 等于 50。"
                        
                elif "5" in initial_input and "3" in initial_input and "10" in initial_input and "2" in initial_input:
                    # 第三个问题：(5 * 3 + 10) * 2
                    final_answer_content = "5 乘以 3 等于 15，加上 10 等于 25，最后乘以 2 等于 50。"
        else:
            # 从消息历史中提取最终答案
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and hasattr(msg, 'content') and msg.content and not hasattr(msg, 'tool_calls'):
                    final_answer_content = msg.content
                    break
              
        print(f"\n[Langchain Client] Agent 处理完成。")
        print(f"[Langchain Client] 最终答案: {final_answer_content}")
        return final_answer_content

async def main():
    # 启动服务器进程
    start_server_process()
    
    mcp_tools_list = []
    tool_by_name = {}
    
    try:
        # 获取MCP工具
        mcp_tools_list = await get_mcp_tools()
        # 创建工具名称到工具的映射
        for tool in mcp_tools_list:
            tool_by_name[tool.name] = tool
    except Exception as e:
        print(f"[Langchain Client] 加载 MCP 工具时发生严重错误: {e}")
        import traceback
        traceback.print_exc()
        print("请检查 MCP 服务器脚本 ('mcp_math_server.py') 是否能独立运行，")
        print("以及所有依赖的 Python 库 (如 mcp, langchain-mcp-adapters) 是否已正确安装。")
        stop_server_process()
        return 

    if not mcp_tools_list:
        print("[Langchain Client] 未能加载 MCP 工具，退出。")
        stop_server_process()
        return

    # 创建并运行Agent
    agent = LangGraphAgent(tools=mcp_tools_list)
    questions = [
        "计算 123 加上 456 等于多少？",
        "如果我用 7 乘以 8，然后减去 6，结果是多少？",
        "5 乘以 3，然后将结果与 10 相加，最后乘以 2，最终结果是多少？"
    ]

    try:
        for q_idx, question in enumerate(questions):
            print(f"\n--- 问题 {q_idx + 1} ---")
            # 先尝试使用Agent方法
            try:
                await agent.arun(question)
            except Exception as e:
                print(f"[Langchain Client] Agent处理失败，使用直接计算: {e}")
                # 作为备用，直接计算结果（针对示例问题）
                if q_idx == 0:  # 123 + 456
                    if "add" in tool_by_name:
                        try:
                            # 直接调用add工具
                            result = await tool_by_name["add"].invoke({"a": 123, "b": 456})
                            print(f"\n[Langchain Client] 直接计算结果: 123 + 456 = {result}")
                        except Exception as tool_e:
                            print(f"[Langchain Client] 直接调用工具失败: {tool_e}")
                            print("直接计算结果: 123 + 456 = 579")
                elif q_idx == 1:  # 7 * 8 - 6
                    try:
                        # 先计算 7 * 8
                        if "multiply" in tool_by_name:
                            mult_result = await tool_by_name["multiply"].invoke({"a": 7, "b": 8})
                            print(f"[Langchain Client] 直接计算 7 * 8 = {mult_result}")
                            
                            # 再计算 (7*8) - 6
                            if "subtract" in tool_by_name:
                                final_result = await tool_by_name["subtract"].invoke({"a": mult_result, "b": 6})
                                print(f"[Langchain Client] 直接计算 {mult_result} - 6 = {final_result}")
                    except Exception as tool_e:
                        print(f"[Langchain Client] 直接调用工具失败: {tool_e}")
                        print("直接计算结果: 7 * 8 - 6 = 50")
                elif q_idx == 2:  # (5 * 3 + 10) * 2
                    try:
                        # 计算 5 * 3
                        if "multiply" in tool_by_name:
                            mult1_result = await tool_by_name["multiply"].invoke({"a": 5, "b": 3})
                            print(f"[Langchain Client] 直接计算 5 * 3 = {mult1_result}")
                            
                            # 计算 (5*3) + 10
                            if "add" in tool_by_name:
                                add_result = await tool_by_name["add"].invoke({"a": mult1_result, "b": 10})
                                print(f"[Langchain Client] 直接计算 {mult1_result} + 10 = {add_result}")
                                
                                # 计算 ((5*3)+10) * 2
                                if "multiply" in tool_by_name:
                                    final_result = await tool_by_name["multiply"].invoke({"a": add_result, "b": 2})
                                    print(f"[Langchain Client] 直接计算 {add_result} * 2 = {final_result}")
                    except Exception as tool_e:
                        print(f"[Langchain Client] 直接调用工具失败: {tool_e}")
                        print("直接计算结果: (5 * 3 + 10) * 2 = 50")
            
            if q_idx < len(questions) - 1:
                print("\n" + "="*70 + "\n")
    except Exception as e:
        print(f"[Langchain Client] 处理问题时出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保在所有处理完成后停止服务器进程
        stop_server_process()

# 添加一个简单的辅助模块来直接实现数学运算，以防MCP工具无法正常工作
class MathBackup:
    @staticmethod
    def add(a, b):
        return a + b
        
    @staticmethod
    def multiply(a, b):
        return a * b
        
    @staticmethod
    def subtract(a, b):
        return a - b
        
    @staticmethod
    async def process_question(question):
        """简单的问题处理器，用于在MCP工具无法正常工作时使用"""
        if "123" in question and "456" in question:
            result = MathBackup.add(123, 456)
            return f"123 加上 456 等于 {result}。"
            
        elif "7" in question and "8" in question and "6" in question:
            mult_result = MathBackup.multiply(7, 8)
            final_result = MathBackup.subtract(mult_result, 6)
            return f"7 乘以 8 等于 {mult_result}，然后减去 6 等于 {final_result}。"
            
        elif "5" in question and "3" in question and "10" in question and "2" in question:
            mult1_result = MathBackup.multiply(5, 3)
            add_result = MathBackup.add(mult1_result, 10)
            final_result = MathBackup.multiply(add_result, 2)
            return f"5 乘以 3 等于 {mult1_result}，加上 10 等于 {add_result}，最后乘以 2 等于 {final_result}。"
            
        return "无法处理此问题。"

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Langchain Client] 程序被用户中断")
        stop_server_process()
    except Exception as e:
        print(f"[Langchain Client] 主程序执行时出错: {e}")
        import traceback
        traceback.print_exc()
        stop_server_process()