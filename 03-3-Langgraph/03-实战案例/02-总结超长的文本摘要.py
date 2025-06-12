import os
from typing import TypedDict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatTongyi

# --- 1. 设置环境 (可选, 如果你已经设置了环境变量则不需要) ---
# os.environ["OPENAI_API_KEY"] = "sk-..."
# os.environ["OPENAI_BASE_URL"] = "https://api.example.com/v1" # 如果需要代理


# --- 2. 定义图的状态 (State) ---
# 状态是图在运行时传递的数据结构。它包含了所有需要跟踪的信息。
class GraphState(TypedDict):
    text: str              # 原始长文本
    chunks: List[str]      # 切分后的文本块列表
    summaries: List[str]   # 每个文本块的摘要列表
    final_summary: str     # 最终的摘要


# --- 3. 初始化模型和文本切分器 ---
# 我们需要一个LLM来进行摘要
llm = ChatTongyi(model="qwen-plus", temperature=0)

# 文本切分器，用于将长文本切分成小块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)


# --- 4. 定义图的节点 (Nodes) ---
# 节点是执行具体任务的函数。每个节点接收当前状态作为输入，并返回一个部分更新的状态。

def chunk_text_node(state: GraphState) -> GraphState:
    """
    节点1：将长文本切分成块。
    """
    print("--- 正在切分文本 ---")
    text = state['text']
    chunks = text_splitter.split_text(text)
    return {"chunks": chunks}

def summarize_map_node(state: GraphState) -> GraphState:
    """
    节点2 (Map): 对每个文本块进行独立摘要。
    """
    print("--- 正在对每个块进行摘要 (Map) ---")
    chunks = state['chunks']
    
    # 为每个块生成摘要的提示模板
    map_prompt = ChatPromptTemplate.from_template(
        "简要总结以下文本内容：\n\n{chunk}"
    )
    
    # 构建摘要链
    summarize_chain = map_prompt | llm
    
    # 并行处理所有块的摘要（这里用列表推导式模拟）
    summaries = summarize_chain.batch(
        [{"chunk": chunk} for chunk in chunks], 
        config={"max_concurrency": 5} # 可以设置并发数
    )
    
    # LangChain v0.2.0+ 的 .batch() 返回的是 AIMessage 列表，我们需要提取内容
    cleaned_summaries = [s.content for s in summaries]
    
    return {"summaries": cleaned_summaries}

def summarize_reduce_node(state: GraphState) -> GraphState:
    """
    节点3 (Reduce): 将所有小摘要合并成最终摘要。
    """
    print("--- 正在合并所有摘要 (Reduce) ---")
    summaries = state['summaries']
    
    # 将所有摘要连接起来
    summaries_joined = "\n\n".join(summaries)
    
    # 创建最终摘要的提示模板
    reduce_prompt = ChatPromptTemplate.from_template(
        "你收到了关于一个长文档的多个摘要。请将它们整合成一个连贯、流畅、全面的最终摘要。\n\n"
        "以下是各个部分的摘要：\n{summaries_text}"
    )
    
    # 构建最终摘要链
    reduce_chain = reduce_prompt | llm
    
    final_summary = reduce_chain.invoke({"summaries_text": summaries_joined})
    
    return {"final_summary": final_summary.content}


# --- 5. 构建并编译图 (Graph) ---
# 现在，我们将上面定义的节点和状态连接成一个工作流。

workflow = StateGraph(GraphState)

# 添加节点
workflow.add_node("chunker", chunk_text_node)
workflow.add_node("mapper", summarize_map_node)
workflow.add_node("reducer", summarize_reduce_node)

# 定义边的连接关系
workflow.set_entry_point("chunker")
workflow.add_edge("chunker", "mapper")
workflow.add_edge("mapper", "reducer")
workflow.add_edge("reducer", END) # 最终节点

# 编译图
app = workflow.compile()


# --- 6. 运行图 ---

# 准备一个超长的示例文本（这里为了演示，只用一段重复的文本模拟）
long_text = """
人工智能（AI）正在以前所未有的速度改变世界。从自动驾驶汽车到医疗诊断，AI的应用无处不在。
其核心技术包括机器学习、深度学习和自然语言处理。机器学习使计算机能够从数据中学习规律，而无需进行显式编程。
深度学习是机器学习的一个分支，它利用深度神经网络模型，在图像识别、语音识别等领域取得了巨大成功。
自然语言处理则致力于让计算机能够理解和生成人类语言，Siri和ChatGPT就是最好的例子。
然而，AI的发展也带来了挑战，如数据隐私、算法偏见和就业冲击。解决这些问题需要技术、法律和伦理的共同努力。
""" * 10 # 将文本重复10次来模拟长文本

# 使用图来处理长文本
# .invoke() 的输入是一个字典，键对应状态中的某个字段
initial_state = {"text": long_text}
final_state = app.invoke(initial_state)

# 打印最终结果
print("\n" + "="*50)
print("✅ 最终摘要：")
print(final_state['final_summary'])