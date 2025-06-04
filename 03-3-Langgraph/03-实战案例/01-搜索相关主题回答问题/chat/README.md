# DeerFlow Chat

A simplified chat system that performs web searches to provide up-to-date information in conversations.

## Features

- Web search integration with either Tavily or DuckDuckGo
- Conversational interface
- Handles search results to provide informative responses
- Simple command-line interface for testing

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install langchain langchain-openai langchain-community pydantic jinja2 pyyaml
```

3. (Optional) For Tavily search:

```bash
pip install tavily-python
```

4. (Optional) For API server:

```bash
pip install fastapi uvicorn
```

4. Set up environment variables in a `.env` file:

```
OPENAI_API_KEY=your_api_key
SEARCH_API=tavily  # or duckduckgo
TAVILY_API_KEY=your_tavily_api_key  # if using Tavily
```

## Usage

### Running the Interactive Chat

```bash
python -m chat.main
```

### Using the API Server

```bash
# Install FastAPI and uvicorn
pip install fastapi uvicorn

# Run the API server
python -m chat.examples.run_api_server
```

API documentation will be available at http://localhost:8000/docs

### Using in Your Code

```python
from chat import process_user_message, ChatState

# Initialize chat state
chat_state = ChatState()

# Process a user message
chat_state = process_user_message("Tell me about the latest news on AI", chat_state)

# Get the assistant's response
assistant_response = chat_state.messages[-1].content
print(f"Assistant: {assistant_response}")
```

### Examples

Check the `chat/examples` directory for example scripts:

- `basic_chat.py` - Basic chat with web search
- `no_search_chat.py` - Chat without web search
- `run_api_server.py` - Run the API server

## Configuration

The system can be configured using:

1. Environment variables
2. A `conf.yaml` file in the project root
3. Command-line arguments

See the example configuration in `conf.yaml.example`.

## Architecture

The chat system is built as a simplified version of the DeerFlow graph architecture, with two main components:

1. `search_node`: Searches the web for information
2. `chat_node`: Generates responses based on search results and conversation history

The system uses LangChain for most components, including LLMs and search tools.

### Project Structure

```
chat/
├── __init__.py          # Package exports
├── api.py               # FastAPI server
├── config.py            # Configuration
├── examples/            # Example scripts
├── llm.py               # LLM configuration
├── main.py              # CLI interface
├── nodes.py             # Core functionality
├── prompts.py           # Prompt templates
├── search.py            # Search tools
└── types.py             # Data types
```

## License

Same as parent DeerFlow project. 