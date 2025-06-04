import os
from datetime import datetime
from typing import Dict, List, Any, Optional

from langchain_core.messages import BaseMessage
from jinja2 import Environment, Template, select_autoescape


# System prompts for different roles
CHAT_SYSTEM_PROMPT = """You are a helpful assistant that searches the web to provide accurate and up-to-date information.
Current time: {{ current_time }}

Your goal is to have a conversation with the user, providing helpful and accurate responses based on:
1. The conversation history
2. Search results from the web (when available)

When search results are available, use them to provide information, but maintain a natural conversational style.
Cite sources when appropriate by mentioning the website name or using [1], [2], etc.

If search results are not available or don't contain relevant information, rely on your general knowledge but acknowledge limitations.
You should be helpful, polite, and concise in your responses.

User's locale: {{ locale }}
"""

SEARCH_SYSTEM_PROMPT = """You are examining search results to find relevant information for a user query.

User query: {{ query }}

Please analyze the following search results to find information relevant to the query.
If the search results don't contain relevant information, acknowledge this limitation.
"""


def render_template(template_string: str, variables: Dict[str, Any]) -> str:
    """Render a template string with the given variables.
    
    Args:
        template_string: The template string to render
        variables: Variables to use in rendering
        
    Returns:
        The rendered template
    """
    env = Environment(autoescape=select_autoescape())
    template = env.from_string(template_string)
    return template.render(**variables)


def create_chat_messages(state: Dict[str, Any]) -> List[Dict[str, str]]:
    """Create a list of messages for the chat model.
    
    Args:
        state: The current state
        
    Returns:
        A list of messages for the chat model
    """
    # Add system message
    system_variables = {
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "locale": state.get("locale", "en-US"),
    }
    
    system_message = {
        "role": "system", 
        "content": render_template(CHAT_SYSTEM_PROMPT, system_variables)
    }
    
    # Convert BaseMessage objects to dict format expected by most LLMs
    messages = [system_message]
    
    for msg in state.get("messages", []):
        if hasattr(msg, "type") and msg.type == "human":
            messages.append({"role": "user", "content": msg.content})
        elif hasattr(msg, "type") and msg.type == "ai":
            messages.append({"role": "assistant", "content": msg.content})
    
    # Add search results if available
    search_results = state.get("search_results")
    if search_results:
        latest_query = state["messages"][-1].content if state.get("messages") else ""
        search_content = "Search results for your query:\n\n"
        
        for i, result in enumerate(search_results):
            search_content += f"{i+1}. {result.get('title', 'No title')}\n"
            search_content += f"{result.get('content', 'No content')}\n\n"
        
        messages.append({"role": "user", "content": search_content})
    
    return messages 