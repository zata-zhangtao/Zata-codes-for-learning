from langchain_openai import ChatOpenAI
import os

import asyncio

llm = ChatOpenAI(
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    model = 'qwq-plus',
    api_key = os.getenv("DASHSCOPE_API_KEY"),
    streaming=True
    
)

for chunk in llm.stream("hello"):
    print(chunk.content, end='',flush=True)
    
# for chunk in llm.invoke("hello"):
#     print(chunk)
# res = llm.invoke("hello")

# async def astream_test():
#     async for chunk in await llm.astream("hello"):
#         print(chunk.content, end='', flush=True)


# asyncio.run(astream_test())




async def main():
    chat = llm
    response = await chat.ainvoke("写一首关于编程的短诗。")
    print(response.content)

    responses = await chat.abatch(["1+1=?", "中国的首都是哪里?"])
    for resp in responses:
        print(resp.content)


asyncio.run(main())
