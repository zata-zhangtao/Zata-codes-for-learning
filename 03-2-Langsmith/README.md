# LangSmith Tutorial & Examples

A comprehensive tutorial project demonstrating how to use LangSmith for tracing and monitoring AI applications, both with and without LangChain.

## 🚀 Overview

This project provides hands-on examples of LangSmith integration for observability and monitoring of AI applications. It covers:

- **LangSmith with LangChain**: Traditional integration using LangChain components
- **LangSmith without LangChain**: Standalone implementation using the `@traceable` decorator
- **Translation Service**: Real-world example using Tongyi/Qwen models
- **API Integration**: Best practices for tracing external API calls
- **Error Handling**: Comprehensive error tracing and debugging

## 📁 Project Structure

```
03-2-Langsmith/
├── 01-langsmith-onboarding/
│   └── langsmith_getting_started.py    # Basic LangSmith + LangChain integration
├── 02-uselangsmit_without_langchain/
│   ├── use_langsmith_without_langchain.ipynb  # Standalone LangSmith tutorial
│   └── requirements.txt                # Additional dependencies
├── pyproject.toml                      # Project configuration
├── uv.lock                            # Lock file for uv package manager
└── README.md                          # This file
```

## 🛠️ Setup & Installation

### Prerequisites

- Python 3.11+
- LangSmith account and API key
- DashScope API key (for Tongyi/Qwen models)

### Installation

1. Clone the repository and navigate to the project directory:
```bash
cd 03-2-Langsmith
```

2. Install dependencies using uv (recommended):
```bash
uv install
```

Or using pip:
```bash
pip install -e .
```

### Environment Configuration

Create a `.env` file in the project root with your API keys:

```bash
# LangSmith Configuration
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=My-LangSmith-Project

# DashScope API (for Tongyi/Qwen models)
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

## 🎯 Usage Examples

### 1. Basic LangSmith + LangChain Integration

```python
# From 01-langsmith-onboarding/langsmith_getting_started.py
from langchain_community.llms import Tongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Define the translation chain
llm = Tongyi(model="qwen-plus-2025-04-28", api_key=os.getenv("DASHSCOPE_API_KEY"))
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
    ("human", "{text}")
])
chain = prompt | llm | StrOutputParser()

# Execute with automatic tracing
result = chain.invoke({
    "input_language": "English",
    "output_language": "Chinese",
    "text": "Hello, how are you?"
})
```

### 2. Standalone LangSmith Tracing

```python
from langsmith import traceable

@traceable(name="Text Processor")
def process_text(text: str) -> dict:
    """Process text with automatic tracing."""
    return {
        "original": text,
        "processed": text.upper(),
        "length": len(text),
        "word_count": len(text.split())
    }

# Function calls are automatically traced
result = process_text("Hello LangSmith!")
```

### 3. Advanced Features

The project demonstrates:

- **Nested Function Tracing**: Hierarchical execution flow
- **Custom Metadata**: Adding context to traces
- **Error Handling**: Automatic error capture
- **API Integration**: Tracing external service calls
- **Tags and Organization**: Filtering and categorization

## 🔍 Key Features

### LangSmith Integration Benefits

- **Automatic Tracing**: Monitor function execution without code changes
- **Error Tracking**: Capture and analyze failures
- **Performance Monitoring**: Track execution times and bottlenecks
- **Debugging Support**: Inspect inputs, outputs, and intermediate steps
- **Team Collaboration**: Share traces and insights

### Supported Models

- **Tongyi/Qwen**: Alibaba's language models via DashScope
- **OpenAI**: GPT models (requires OpenAI API key)
- **Custom Models**: Any model compatible with LangChain

## 📊 Monitoring & Observability

Once configured, your traces will appear in the LangSmith dashboard:

1. Visit [LangSmith Dashboard](https://smith.langchain.com)
2. Select your project
3. View traces, metrics, and performance data
4. Set up alerts and monitoring rules

## 🤝 Contributing

This is a tutorial project. Feel free to:

- Add new examples
- Improve existing code
- Fix bugs or documentation
- Share feedback and suggestions

## 📚 Resources

- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [LangChain Documentation](https://docs.langchain.com/)
- [DashScope API Reference](https://help.aliyun.com/zh/dashscope/)
- [LangSmith Python SDK](https://github.com/langchain-ai/langsmith-sdk)

## 🔧 Dependencies

Main dependencies (see `pyproject.toml` for complete list):

- `langchain-community`: LangChain integrations
- `langchain-core`: Core LangChain functionality
- `langchain-openai`: OpenAI integration
- `dashscope`: Alibaba DashScope API client
- `python-dotenv`: Environment variable management

## 📄 License

This project is for educational purposes. Please refer to the respective licenses of the used libraries and services.