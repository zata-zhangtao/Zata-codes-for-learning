from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
load_dotenv()

chatLLM = ChatOpenAI(
    api_key=os.getenv("api_key"),
    base_url=os.getenv("base_url"),
    model="qwen-plus",  # 此处以qwen-plus为例，您可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    # other params...
)


###################调用方法一########################
result = chatLLM.stream("2025年的技术趋势是什么？")
for chunk in result:
    print(chunk.content, end='', flush=True)
###################################################


###################调用方法二##########################
# 定义提示模板
prompt = PromptTemplate(
    input_variables=["question"],
    template="请简洁回答以下问题：{question}"
)

# 创建链
chain = LLMChain(llm=chatLLM, prompt=prompt)

# 运行链
question = "2025年的技术趋势是什么？"
response = chain.run(question)
print(response)
######################################################


###################调用方法三########################
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你是谁？"}]
response = chatLLM.invoke(messages)
print(response.json())
###################################################