# RAG实战案例分析

## 目录
- [案例1：企业知识库问答系统](#案例1企业知识库问答系统)
- [案例2：法律文档助手](#案例2法律文档助手)
- [案例3：多语言客户支持系统](#案例3多语言客户支持系统)
- [实战最佳实践](#实战最佳实践)

## 案例1：企业知识库问答系统

### 场景描述

某科技公司需要构建内部知识库问答系统，帮助员工快速获取公司政策、产品信息和技术文档内容。系统需要处理多种格式的文档，包括PDF、Word和HTML，并能准确回答员工的问题。

### 技术实现

#### 1. 数据准备与索引

```python
from typing import List, Dict, Any
import os
from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

class DocumentProcessor:
    def __init__(self, docs_dir: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.docs_dir = docs_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
    def load_documents(self) -> List[Dict[str, Any]]:
        """加载多种格式的文档并分块处理"""
        documents = []
        
        for root, _, files in os.walk(self.docs_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if file.endswith('.pdf'):
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                    elif file.endswith('.docx'):
                        loader = Docx2txtLoader(file_path)
                        docs = loader.load()
                    elif file.endswith('.html'):
                        loader = UnstructuredHTMLLoader(file_path)
                        docs = loader.load()
                    else:
                        continue
                        
                    # 文档分块
                    chunks = self.text_splitter.split_documents(docs)
                    
                    # 提取元数据
                    for chunk in chunks:
                        documents.append({
                            "text": chunk.page_content,
                            "metadata": {
                                "source": file_path,
                                "page": chunk.metadata.get("page", 0),
                                "category": self._determine_category(file_path)
                            }
                        })
                        
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")
                    
        return documents
    
    def _determine_category(self, file_path: str) -> str:
        """根据文件路径确定文档类别"""
        path_parts = file_path.split(os.path.sep)
        for part in path_parts:
            if part in ["policies", "hr"]:
                return "公司政策"
            elif part in ["products", "services"]:
                return "产品信息"
            elif part in ["tech", "development"]:
                return "技术文档"
        return "其他"
```

#### 2. 查询处理与对话管理

```python
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import openai
from typing import List, Dict, Any

class EnterpriseKnowledgeBase:
    def __init__(self, api_key: str, collection_name: str = "enterprise_docs"):
        openai.api_key = api_key
        self.embedding_model = SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
        self.client = chromadb.Client()
        
        # 获取或创建集合
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_model
            )
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_model
            )
            
    def index_documents(self, documents: List[Dict[str, Any]]):
        """向向量数据库添加文档"""
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            
            self.collection.add(
                documents=[doc["text"] for doc in batch],
                metadatas=[doc["metadata"] for doc in batch],
                ids=[f"doc_{i+j}" for j in range(len(batch))]
            )
            
    def query(self, question: str, top_k: int = 5, category_filter: str = None) -> Dict[str, Any]:
        """查询知识库并生成回答"""
        # 构建查询过滤器
        filter_dict = {}
        if category_filter:
            filter_dict["category"] = category_filter
            
        # 执行向量搜索
        results = self.collection.query(
            query_texts=[question],
            n_results=top_k,
            where=filter_dict if filter_dict else None
        )
        
        contexts = results["documents"][0]
        sources = [
            {"source": results["metadatas"][0][i]["source"],
             "page": results["metadatas"][0][i]["page"],
             "category": results["metadatas"][0][i]["category"]}
            for i in range(len(contexts))
        ]
        
        # 构建提示
        prompt = f"""
        基于以下知识库内容回答用户问题。只使用提供的信息，不要编造内容。
        如果无法从提供的内容中找到答案，请说明"基于当前的信息无法回答这个问题"。
        
        知识库内容:
        {' '.join(contexts)}
        
        用户问题: {question}
        
        回答:
        """
        
        # 生成回答
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=500,
            temperature=0.2
        )
        
        return {
            "answer": response.choices[0].text.strip(),
            "sources": sources
        }
```

#### 3. 实现Web界面

```python
import streamlit as st
import pandas as pd
from enterprise_kb import EnterpriseKnowledgeBase, DocumentProcessor
import os

# 配置页面
st.set_page_config(page_title="企业知识库助手", layout="wide")
st.title("企业知识库问答系统")

# 初始化会话状态
if "kb" not in st.session_state:
    api_key = os.getenv("OPENAI_API_KEY", "")
    st.session_state.kb = EnterpriseKnowledgeBase(api_key=api_key)
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 侧边栏 - 文档处理
with st.sidebar:
    st.header("文档管理")
    
    docs_dir = st.text_input("文档目录路径", value="./company_docs")
    
    if st.button("加载并索引文档"):
        with st.spinner("正在处理文档..."):
            processor = DocumentProcessor(docs_dir=docs_dir)
            documents = processor.load_documents()
            st.session_state.kb.index_documents(documents)
            st.success(f"成功索引 {len(documents)} 个文档片段")
    
    # 过滤选项
    st.header("查询选项")
    category = st.selectbox(
        "按类别过滤",
        options=["全部", "公司政策", "产品信息", "技术文档", "其他"],
        index=0
    )
    
    st.markdown("---")
    if st.button("清空聊天记录"):
        st.session_state.chat_history = []

# 主界面 - 聊天
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # 显示参考来源
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("查看参考来源"):
                sources_df = pd.DataFrame(message["sources"])
                st.dataframe(sources_df)

# 用户输入
question = st.chat_input("请输入您的问题")
if question:
    # 添加用户消息到历史
    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })
    
    # 显示用户消息
    with st.chat_message("user"):
        st.write(question)
    
    # 显示助手消息
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            category_filter = None if category == "全部" else category
            response = st.session_state.kb.query(
                question=question,
                category_filter=category_filter
            )
            
            st.write(response["answer"])
            
            # 显示参考来源
            with st.expander("查看参考来源"):
                sources_df = pd.DataFrame(response["sources"])
                st.dataframe(sources_df)
    
    # 添加助手消息到历史
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response["answer"],
        "sources": response["sources"]
    })
```

### 性能优化与经验总结

1. **索引优化**：按文档类别建立多个集合，提高检索精度和速度。
2. **查询预处理**：实现问题分类和意图识别，优化检索策略。
3. **用户反馈循环**：收集用户反馈，持续优化系统回答质量。

## 案例2：法律文档助手

### 场景描述

某律师事务所需要一个系统来快速分析和检索大量法律文件、案例和法规条文，帮助律师准备案件和提供法律建议。系统需要处理专业法律术语，并提供准确的引用和来源。

### 技术实现关键点

1. **专业领域预处理**

```python
# 法律术语提取器示例
def extract_legal_entities(text):
    """提取文本中的法律实体，如法规名称、案例编号等"""
    legal_entities = {
        "laws": [],
        "cases": [],
        "terms": []
    }
    
    # 法规匹配
    law_pattern = r"《(.*?法|.*?条例|.*?规定)》"
    legal_entities["laws"] = re.findall(law_pattern, text)
    
    # 案例号匹配
    case_pattern = r"(\(\d{4}\).*号)"
    legal_entities["cases"] = re.findall(case_pattern, text)
    
    # 其他自定义法律术语匹配...
    
    return legal_entities
```

2. **特定领域嵌入方法**

```python
# 使用特定领域的嵌入模型
from sentence_transformers import SentenceTransformer

class LegalEmbedder:
    def __init__(self):
        # 选择适合法律文本的预训练模型
        # 或微调通用模型以适应法律领域
        self.model = SentenceTransformer("distiluse-base-multilingual-cased")
        
    def fine_tune(self, legal_texts, labels):
        """微调模型以更好地理解法律文本"""
        # 微调代码...
        pass
        
    def embed(self, texts):
        """生成文本嵌入"""
        return self.model.encode(texts)
```

3. **层级检索策略**

```python
def hierarchical_retrieval(query, top_k=10):
    """实现层级检索策略"""
    # 1. 粗筛选 - 使用轻量级模型或关键词匹配
    relevant_docs_ids = first_stage_retrieval(query)
    
    # 2. 精筛选 - 对粗筛选结果使用更复杂的模型
    reranked_docs = rerank_documents(query, relevant_docs_ids)
    
    # 3. 根据文档类型划分结果
    categorized_results = {
        "statutes": [],
        "cases": [],
        "commentaries": []
    }
    
    for doc in reranked_docs[:top_k]:
        doc_type = determine_document_type(doc)
        categorized_results[doc_type].append(doc)
        
    return categorized_results
```

### 关键成果与经验教训

- **精确性提升**: 使用专业术语提取和匹配方法使法律相关文档的检索准确率从75%提高到92%
- **效率改进**: 系统帮助律师减少70%的案例准备时间
- **挑战**: 需要处理法律文件的引用格式多样性和跨文档关系

## 案例3：多语言客户支持系统

### 场景描述

国际电商平台需要构建多语言客户支持系统，能够理解和回答不同语言的产品查询、订单问题和退款请求，同时保持回答的一致性。

### 核心技术实现

1. **多语言处理管道**

```python
from transformers import MarianMTModel, MarianTokenizer

class MultilingualProcessor:
    def __init__(self):
        # 加载英语作为枢纽语言的翻译模型
        self.translators = {}
        self.supported_languages = ["en", "zh", "es", "fr", "de", "ja"]
        
        # 初始化翻译模型
        for lang in self.supported_languages:
            if lang != "en":
                # 加载目标语言到英语的模型
                model_name = f"Helsinki-NLP/opus-mt-{lang}-en"
                self.translators[f"{lang}_to_en"] = {
                    "tokenizer": MarianTokenizer.from_pretrained(model_name),
                    "model": MarianMTModel.from_pretrained(model_name)
                }
                
                # 加载英语到目标语言的模型
                model_name = f"Helsinki-NLP/opus-mt-en-{lang}"
                self.translators[f"en_to_{lang}"] = {
                    "tokenizer": MarianTokenizer.from_pretrained(model_name),
                    "model": MarianMTModel.from_pretrained(model_name)
                }
    
    def detect_language(self, text):
        """检测输入文本的语言"""
        # 语言检测逻辑
        # 可以使用如fastText或langdetect库
        pass
        
    def translate_to_english(self, text, source_lang):
        """将文本翻译成英语"""
        if source_lang == "en":
            return text
            
        translator = self.translators.get(f"{source_lang}_to_en")
        if not translator:
            raise ValueError(f"不支持的源语言: {source_lang}")
            
        tokenizer = translator["tokenizer"]
        model = translator["model"]
        
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        translated = model.generate(**inputs)
        
        return tokenizer.decode(translated[0], skip_special_tokens=True)
        
    def translate_from_english(self, text, target_lang):
        """将英语文本翻译成目标语言"""
        if target_lang == "en":
            return text
            
        translator = self.translators.get(f"en_to_{target_lang}")
        if not translator:
            raise ValueError(f"不支持的目标语言: {target_lang}")
            
        tokenizer = translator["tokenizer"]
        model = translator["model"]
        
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        translated = model.generate(**inputs)
        
        return tokenizer.decode(translated[0], skip_special_tokens=True)
```

2. **多语言RAG系统**

```python
class MultilingualCustomerSupportRAG:
    def __init__(self, openai_api_key):
        self.language_processor = MultilingualProcessor()
        self.vector_store = setup_vector_store()
        self.openai_api_key = openai_api_key
        
    def process_query(self, query, user_language):
        """处理用户查询并返回多语言回答"""
        # 1. 检测语言（如果未提供）
        if not user_language:
            user_language = self.language_processor.detect_language(query)
            
        # 2. 将查询翻译成英语
        english_query = self.language_processor.translate_to_english(query, user_language)
        
        # 3. 检索相关文档
        relevant_docs = self.vector_store.similarity_search(english_query, k=5)
        
        # 4. 生成英文回答
        context = "\n".join([doc.page_content for doc in relevant_docs])
        english_answer = self.generate_answer(english_query, context)
        
        # 5. 将回答翻译回用户语言
        translated_answer = self.language_processor.translate_from_english(
            english_answer, user_language
        )
        
        return {
            "original_query": query,
            "answer": translated_answer,
            "detected_language": user_language,
            "sources": [doc.metadata for doc in relevant_docs]
        }
        
    def generate_answer(self, query, context):
        """使用LLM生成回答"""
        prompt = f"""
        您是一位客户支持专家。请根据提供的产品信息回答客户查询。
        仅使用提供的信息回答，如果无法找到答案，请说明您没有足够信息。
        
        产品信息:
        {context}
        
        客户查询: {query}
        
        您的回答:
        """
        
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=300,
            temperature=0.2
        )
        
        return response.choices[0].text.strip()
```

### 关键成果与挑战

- **覆盖面**: 系统支持6种语言，覆盖了平台90%的用户群体
- **一致性**: 通过"翻译-检索-生成-翻译"的流程保证了不同语言回答的内容一致性
- **挑战**: 翻译质量对系统效果影响显著，特别是专业术语的翻译

## 实战最佳实践

### 1. 数据处理关键策略

* **领域适应性文本分块**: 根据文档类型动态调整分块大小（法律文件按段落，技术文档按章节）
* **元数据增强**: 为每个块附加丰富元数据（来源、类型、重要度、时效性）
* **数据质量控制**: 实现自动化检测和过滤低质量内容的机制

### 2. 检索策略优化

| 检索策略 | 适用场景 | 实施难度 | 效果提升 |
|---------|---------|---------|---------|
| 混合检索 | 通用知识库 | 中等 | ★★★☆☆ |
| 多阶段检索 | 大规模文档集 | 中高 | ★★★★☆ |
| 查询改写 | 用户查询不明确 | 低 | ★★★★☆ |
| 语义路由 | 跨领域知识库 | 高 | ★★★★★ |

### 3. 通用优化技巧

* **缓存机制**: 实现多级缓存（查询缓存、检索结果缓存、生成结果缓存）
* **反馈循环**: 收集用户反馈并用于系统优化
* **系统监控**: 实施全面的监控指标（延迟、召回率、准确度、用户满意度）
* **A/B测试**: 通过对照实验评估不同策略效果

### 4. 经验教训汇总

* **RAG系统最重要的是数据质量**，而非复杂算法
* **分块策略**直接影响检索和生成质量
* **领域知识注入**可显著提升专业场景下的表现
* **反馈机制**是持续改进的关键 