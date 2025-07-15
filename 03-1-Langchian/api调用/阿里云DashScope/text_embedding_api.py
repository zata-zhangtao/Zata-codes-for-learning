# https://python.langchain.com/api_reference/community/embeddings/langchain_community.embeddings.dashscope.DashScopeEmbeddings.html
from langchain_community.embeddings import DashScopeEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()


api_key = os.getenv("api_key")

embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=api_key,
    # other params...
)

text = "This is a test document."

query_result = embeddings.embed_query(text)
print("文本向量长度：", len(query_result), sep='')

doc_results = embeddings.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ])
print("文本向量数量：", len(doc_results), "，文本向量长度：", len(doc_results[0]), sep='')
