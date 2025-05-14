"""
Langchain 网页爬虫 (Web Scraping) 教程
===================================

本教程介绍如何使用Langchain进行网页抓取和内容处理，将网页数据整合到RAG应用中
"""

# 导入必要的库
import os
import tempfile
from langchain_community.document_loaders import (
    WebBaseLoader,
    SeleniumURLLoader,
    RecursiveUrlLoader,
    SitemapLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 设置环境变量
api_key = os.environ.get("ALIBABA_API_KEY", "你的API密钥")
os.environ["ALIBABA_API_KEY"] = api_key

print("Langchain 网页爬虫 (Web Scraping) 教程")
print("=" * 60)

# 1. 基本网页加载器
print("\n1. 基本网页加载器 (WebBaseLoader)")
print("-" * 40)
print("注意: 需要安装requests和bs4: pip install requests beautifulsoup4")

# 定义示例URL
example_url = "https://python.langchain.com/docs/modules/data_connection/"
example_urls = [
    "https://python.langchain.com/docs/modules/data_connection/document_loaders/",
    "https://python.langchain.com/docs/modules/data_connection/text_splitters/"
]

# 使用WebBaseLoader加载单个网页
print("1.1 加载单个网页")
web_loader = WebBaseLoader(example_url)
web_loader.requests_kwargs = {"verify": False}  # 忽略SSL验证，根据需要设置
documents = web_loader.load()

print(f"加载了 {len(documents)} 个文档")
if documents:
    print(f"文档内容预览: {documents[0].page_content[:150]}...")
    print(f"URL元数据: {documents[0].metadata}")

# 使用WebBaseLoader加载多个网页
print("\n1.2 加载多个网页")
urls_loader = WebBaseLoader(example_urls)
urls_documents = urls_loader.load()

print(f"加载了 {len(urls_documents)} 个文档")
for i, doc in enumerate(urls_documents):
    print(f"文档 {i+1} URL: {doc.metadata.get('source', '未知')}")

# 2. 使用Selenium加载动态网页
print("\n2. 使用Selenium加载动态网页 (SeleniumURLLoader)")
print("-" * 40)
print("注意: 需要安装selenium和webdriver: pip install selenium webdriver-manager")

print("""
# Selenium示例代码
from langchain_community.document_loaders import SeleniumURLLoader

# 加载需要JavaScript渲染的页面
selenium_loader = SeleniumURLLoader(
    urls=["https://www.example.com/dynamic-page"],
    continue_on_failure=True,  # 遇到错误继续处理其他URL
    headless=True              # 无头模式运行浏览器
)
selenium_docs = selenium_loader.load()
""")

# 3. 递归URL加载器
print("\n3. 递归URL加载器 (RecursiveUrlLoader)")
print("-" * 40)
print("注意: 对于需要抓取整个网站或特定路径下所有页面的场景")

print("""
# 递归URL加载器示例代码
from langchain_community.document_loaders import RecursiveUrlLoader
from bs4 import BeautifulSoup

# 配置加载器递归抓取网站
recursive_loader = RecursiveUrlLoader(
    url="https://example.com/docs/",
    max_depth=2,  # 最大递归深度
    extractor=lambda x: BeautifulSoup(x, "html.parser").text  # 提取文本
)
recursive_docs = recursive_loader.load()
""")

# 4. 网站地图加载器
print("\n4. 网站地图加载器 (SitemapLoader)")
print("-" * 40)
print("注意: 适用于有sitemap.xml的网站，可以高效地抓取所有页面")

print("""
# 网站地图加载器示例代码
from langchain_community.document_loaders import SitemapLoader

# 从网站地图加载页面
sitemap_loader = SitemapLoader(
    web_path="https://example.com/sitemap.xml",
    filter_urls=["https://example.com/blog/"],  # 可选，只加载匹配的URL
    parsing_function=lambda soup: soup.text  # 自定义解析函数
)
sitemap_docs = sitemap_loader.load()
""")

# 5. 高级网页内容处理
print("\n5. 高级网页内容处理")
print("-" * 40)

# 5.1 网页内容清洗
print("\n5.1 网页内容清洗")
print("网页内容通常包含大量噪音，需要进行清洗和格式化")

print("""
# 自定义BeautifulSoup提取函数示例
from bs4 import BeautifulSoup

def extract_clean_text(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    
    # 删除脚本、样式和导航等无关内容
    for element in soup(['script', 'style', 'nav', 'footer', 'header']):
        element.decompose()
    
    # 提取正文内容
    main_content = soup.find('main') or soup.find('article') or soup.find('body')
    
    # 获取文本并清理格式
    if main_content:
        text = main_content.get_text(separator='\\n', strip=True)
    else:
        text = soup.get_text(separator='\\n', strip=True)
    
    # 清理多余空白行
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    clean_text = '\\n'.join(lines)
    
    return clean_text

# 在WebBaseLoader中使用
loader = WebBaseLoader(
    web_paths=["https://example.com"],
    bs_kwargs={"parser": "html.parser"},
    bs_get_text_kwargs={"strip": True},
    parsing_function=extract_clean_text
)
""")

# 5.2 处理反爬虫机制
print("\n5.2 处理反爬虫机制")
print("许多网站有反爬虫措施，需要特殊处理")

print("""
# 处理反爬虫示例
from langchain_community.document_loaders import WebBaseLoader
import time
import random

# 自定义请求头
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://www.google.com/"
}

# 对多个URL进行处理，添加延迟避免被封
urls = ["https://example.com/page1", "https://example.com/page2"]
documents = []

for url in urls:
    try:
        loader = WebBaseLoader(
            web_paths=[url],
            requests_kwargs={"headers": headers, "timeout": 10}
        )
        docs = loader.load()
        documents.extend(docs)
        
        # 随机延迟，避免频繁请求
        time.sleep(random.uniform(1, 3))
    except Exception as e:
        print(f"处理URL {url} 时出错: {e}")
""")

# 6. 构建网页内容RAG应用
print("\n6. 构建网页内容RAG应用")
print("-" * 40)

print("""
# 完整网页RAG应用示例

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. 加载网页内容
loader = WebBaseLoader([
    "https://example.com/article1",
    "https://example.com/article2"
])
documents = loader.load()

# 2. 分割文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
splits = text_splitter.split_documents(documents)

# 3. 创建向量存储
embeddings = HuggingFaceEmbeddings(
    model_name="shibing624/text2vec-base-chinese"
)
vectorstore = FAISS.from_documents(splits, embeddings)

# 4. 创建检索器
retriever = vectorstore.as_retriever()

# 5. 创建LLM
llm = QianfanChatEndpoint(
    model="qwen-max",
    qianfan_api_key="你的API密钥",
    streaming=True
)

# 6. 创建提示模板
template = \"\"\"基于以下从网页中检索到的信息回答问题。
如果检索到的信息无法回答问题，请说"我无法从提供的网页内容中找到相关信息"。

检索到的信息:
{context}

问题: {question}

回答:\"\"\"

prompt = PromptTemplate.from_template(template)

# 7. 定义格式化函数
def format_docs(docs):
    return "\\n\\n".join([f"来源: {doc.metadata.get('source', '未知')}\\n内容: {doc.page_content}" for doc in docs])

# 8. 构建RAG链
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 9. 使用RAG链回答问题
response = rag_chain.invoke("网页中提到了什么主要内容？")
print(response)
""")

# 7. 网页爬虫的实用技巧
print("\n7. 网页爬虫的实用技巧")
print("-" * 40)

tips = [
    "1. 尊重robots.txt: 检查网站的robots.txt文件，遵守爬取规则",
    "2. 设置合理延迟: 添加随机延迟，避免频繁请求导致IP被封",
    "3. 使用代理IP池: 对于大规模爬取，可以使用代理IP池轮换请求",
    "4. 定制User-Agent: 使用真实浏览器的User-Agent，避免被识别为爬虫",
    "5. 错误处理与重试: 实现异常捕获和重试机制，增强爬取稳定性",
    "6. 存储原始HTML: 保存原始HTML内容，便于后期重新处理",
    "7. 增量爬取: 只抓取新增或更新的内容，减少重复工作",
    "8. 网站地图优先: 优先使用sitemap.xml进行抓取，效率更高",
    "9. API替代爬虫: 如果网站提供API，优先使用API获取数据",
    "10. 日志与监控: 实现完善的日志和监控系统，及时发现问题"
]

for tip in tips:
    print(tip)

# 8. 特定场景实例
print("\n8. 特定场景实例")
print("-" * 40)
print("以下是一些常见网站爬取的特定实现")

print("""
# 例1: 抓取知乎问题与回答
from langchain_community.document_loaders import SeleniumURLLoader

zhihu_loader = SeleniumURLLoader(
    urls=["https://www.zhihu.com/question/12345678"],
    continue_on_failure=True
)
zhihu_content = zhihu_loader.load()

# 例2: 抓取微信公众号文章(通过搜狗搜索)
sogou_weixin_loader = WebBaseLoader(
    web_paths=["https://weixin.sogou.com/weixin?type=2&query=langchain"]
)
weixin_search_results = sogou_weixin_loader.load()

# 例3: 抓取学术论文(通过Arxiv)
from langchain_community.document_loaders import ArxivLoader

arxiv_loader = ArxivLoader(
    query="LLM RAG",
    load_max_docs=10,
    load_all_available_meta=True
)
papers = arxiv_loader.load()
""")

# 9. 内容存储与更新策略
print("\n9. 内容存储与更新策略")
print("-" * 40)

print("""
# 增量爬取与内容更新示例

import hashlib
import json
import os
from datetime import datetime

# 计算内容哈希值，用于判断内容是否变化
def get_content_hash(content):
    return hashlib.md5(content.encode('utf-8')).hexdigest()

# 保存抓取记录
def save_crawl_record(url, content_hash, metadata):
    records_file = "crawl_records.json"
    
    # 读取现有记录
    if os.path.exists(records_file):
        with open(records_file, 'r', encoding='utf-8') as f:
            records = json.load(f)
    else:
        records = {}
    
    # 更新记录
    records[url] = {
        "content_hash": content_hash,
        "last_crawl": datetime.now().isoformat(),
        "metadata": metadata
    }
    
    # 保存记录
    with open(records_file, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

# 检查URL是否需要更新
def need_update(url, content_hash):
    records_file = "crawl_records.json"
    
    if not os.path.exists(records_file):
        return True
    
    with open(records_file, 'r', encoding='utf-8') as f:
        records = json.load(f)
    
    if url not in records:
        return True
    
    # 如果内容哈希值变化，需要更新
    return records[url]["content_hash"] != content_hash
""")

print("\n网页爬虫总结")
print("=" * 30)
print("1. Langchain提供了多种网页加载器，适用于不同的网站抓取需求")
print("2. 对于静态网页，WebBaseLoader通常足够；对于动态网页，需要使用SeleniumURLLoader")
print("3. 对于大型网站，可以使用RecursiveUrlLoader或SitemapLoader进行批量抓取")
print("4. 网页内容需要进行清洗和处理，去除噪音数据")
print("5. 爬虫需要遵守网络爬取规则，添加延迟和错误处理")
print("6. 实现增量爬取策略可以提高效率，避免重复处理")
print("7. 将网页内容整合到RAG应用中，可以为LLM提供最新的网络信息") 