# RAG Tutorial with LangChain

This tutorial demonstrates how to build a Retrieval-Augmented Generation (RAG) application using LangChain. The implementation shows how to create a question-answering system that can retrieve relevant information from documents and generate accurate responses.

## Features

- Document loading and text splitting
- Vector embeddings using DashScope's text-embedding-v2 model
- FAISS vector store for efficient similarity search
- Question-answering using LangChain's RetrievalQA chain with Qwen-Plus model
- Source document retrieval and display

## Prerequisites

- Python 3.8 or higher
- DashScope API key (for accessing models)

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root and add your DashScope API key:
   ```
   DASHSCOPE_API_KEY=your_api_key_here
   ```

## Usage

1. Place your document(s) in the project directory (the example uses `example_doc.txt`)
2. Run the tutorial:
   ```bash
   python rag_tutorial.py
   ```

## How it Works

1. **Document Processing**: The system loads and splits documents into manageable chunks
2. **Embedding Generation**: Text chunks are converted into vector embeddings using DashScope's text-embedding-v2 model
3. **Vector Store Creation**: Embeddings are stored in a FAISS vector database
4. **Question Answering**: The system retrieves relevant documents and generates answers using the Qwen-Plus model

## Customization

You can modify the following parameters in the code:
- Chunk size and overlap in the text splitter
- Number of retrieved documents (k value)
- LLM parameters (temperature, max length)

## Note

This tutorial uses the following models:
- Embeddings: DashScope's `text-embedding-v2`
- LLM: DashScope's `qwen-plus`

You can change these models by modifying the respective parameters in the code.

# 简介
本文件夹是AI Agent的学习记录

## ipynb文件查看方式
很多ipynb文件在github中无法直接查看，可以使用下面的方式查看
1. nbviewer查看

https://nbviewer.org/

比如下面这个文件,github打不开
![alt text](assets/readme/image-1.png)
可以把url地址复制到上面的网站中查看
![alt text](assets/readme/image.png)


## 使用教程
进入每一个目录下面之后，根据对应的requirements.txt安装依赖包


## 项目中常用的函数说明

### from dotenv import load_dotenv
load_dotenv函数用于加载环境变量，如果文件所在的路径下面不存在.env文件会到上一级目录寻找，如果加载不成功也不会报错，所以使用的时候最好打印一下输出
