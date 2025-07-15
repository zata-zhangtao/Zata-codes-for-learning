from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_community.llms import Tongyi

from dotenv import load_dotenv
load_dotenv()


# 确保已设置 dashcope 的 api 

# 确保已设置 LangSmith 环境变量 (如上一节所述)
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_API_KEY"] = "YOUR_LANGSMITH_API_KEY"
# os.environ["LANGCHAIN_PROJECT"] = "My First Project" # 替换为你的项目名

# 定义模型
llm = Tongyi(model="qwen-plus-2025-04-28", api_key=os.getenv("DASHSCOPE_API_KEY"))

# 定义提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
    ("human", "{text}")
])

# 定义输出解析器
parser = StrOutputParser()

# 构建链
chain = prompt | llm | parser

# 运行链
try:
    result = chain.invoke({
        "input_language": "English",
        "output_language": "Chinese",
        "text": "Hello, how are you?"
    })
    print(result)
except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure your OpenAI API key and LangSmith environment variables are correctly set.")
