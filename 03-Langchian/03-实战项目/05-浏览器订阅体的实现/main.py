import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient # 更改导入：使用 TavilyClient
import os # 用于从环境变量获取 API 密钥（推荐方式）

# 帮助函数：尝试从 URL 获取并解析页面内容
# 此函数保持不变，因为 Tavily 可能返回链接，我们仍然可能想要抓取原始内容
def get_page_content(url):
    """
    尝试下载并提取给定 URL 的主要文本内容。
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # 如果请求失败则抛出 HTTPError 错误
        
        response.encoding = response.apparent_encoding if response.apparent_encoding else 'utf-8'

        soup = BeautifulSoup(response.text, 'html.parser')

        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        text = ' '.join(soup.stripped_strings)
        
        return text[:2000] # 返回前2000个字符作为预览 (增加了长度以便获取更多上下文)
    except requests.exceptions.RequestException as e:
        return f"请求页面时出错: {e}"
    except Exception as e:
        return f"解析页面内容时出错: {e}"

def search_and_list_results_tavily(api_key, query, num_results=5):
    """
    使用 Tavily API 进行搜索，并列出结果的标题、链接和页面内容摘要。
    """
    try:
        # 初始化 TavilyClient
        # 你可以直接传递 api_key，或者如果设置了 TAVILY_API_KEY 环境变量，则无需传递
        client = TavilyClient(api_key=api_key)

        # 使用 Tavily SDK 进行搜索
        # Tavily 的 search 方法有很多参数可以调整，例如：
        # - search_depth: "basic" (默认) 或 "advanced" (更深入，可能消耗更多信用点)
        # - include_answer: True/False (是否包含 AI 生成的总结性答案)
        # - include_raw_content: True/False (是否包含原始网页内容，如果为True，可能就不需要下面的 get_page_content)
        # - include_images: True/False
        # - max_results: 要返回的结果数量
        response = client.search(
            query=query,
            search_depth="basic",
            include_answer=True, # 尝试获取一个总结性答案
            max_results=num_results,
            # include_raw_content=True # 如果设为True，Tavily会尝试直接返回页面内容
        )
        
        # Tavily API 的响应结构:
        # response 是一个字典，包含键如 'query', 'answer', 'results', 'images' 等。
        # 'results' 是一个列表，每个元素是一个字典，包含 'title', 'url', 'content', 'score', 'raw_content' (如果请求了)。

        print(f"\n为你找到关于 '{query}' 的搜索结果 (使用 Tavily API)：\n" + "="*55)

        if "answer" in response and response["answer"]:
            print(f"Tavily 总结性回答:\n  {response['answer']}\n" + "-"*55)

        if "results" in response and response["results"]:
            for i, result in enumerate(response["results"]): # Tavily直接在response['results']中给出结果列表
                title = result.get("title", "N/A")
                link = result.get("url", "N/A")
                # 'content' 来自 Tavily，通常是该结果页面的一个很好的摘要或相关部分
                tavily_content_snippet = result.get("content", "N/A")
                # 'raw_content' 如果在请求中设置了 include_raw_content=True，Tavily会尝试填充这个字段
                raw_content_from_tavily = result.get("raw_content", None)


                print(f"\n结果 {i+1}:")
                print(f"  标题: {title}")
                print(f"  链接: {link}")
                print(f"  Tavily 提供的片段/内容:\n    {tavily_content_snippet}")

                if raw_content_from_tavily:
                    print(f"  Tavily 提供的原始页面内容 (摘要):\n    {raw_content_from_tavily[:1000]}...\n")
                elif link and link != "N/A":
                    # 如果 Tavily 没有提供原始文本，或者其内容不够，可以尝试自己抓取
                    print(f"  正在尝试通过 requests 获取页面 '{link}' 的详细内容...")
                    page_content_full = get_page_content(link)
                    print(f"  页面详细内容摘要 (requests):\n    {page_content_full}\n")
                else:
                    print("  无法获取此链接的页面内容，或 Tavily 未提供原始内容。")
                print("-"*55)
        else:
            print("没有找到搜索结果。")
            # Tavily的错误处理通常是通过异常进行的，但也可以检查响应本身
            if not response.get("results") and not response.get("answer"):
                 print("响应中既没有结果也没有答案。")


    except Exception as e:
        print(f"执行 Tavily 搜索时发生错误: {e}")

if __name__ == "__main__":
    # 强烈建议将 API 密钥存储为环境变量 TAVILY_API_KEY
    # client = TavilyClient() 会自动查找它
    # 如果不使用环境变量，请取消注释并替换下面的行：
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")# 请替换为你的 Tavily API 密钥

    # 检查环境变量是否设置
    # if os.getenv("TAVILY_API_KEY"):
    #     TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    #     print("已从环境变量 TAVILY_API_KEY 加载 API 密钥。")
    # el
    if not TAVILY_API_KEY: # 或者 not TAVILY_API_KEY
        print("错误：请在代码中设置 TAVILY_API_KEY，或设置 TAVILY_API_KEY 环境变量。")
        print("你需要前往 https://tavily.com/ 注册并获取你的 API 密钥。")
    else:
        keyword = input("请输入你要搜索的关键字: ")
        if keyword:
            # 你可以调整想获取的结果数量，例如 5 个
            search_and_list_results_tavily(TAVILY_API_KEY, keyword, num_results=3)
        else:
            print("关键字不能为空。")