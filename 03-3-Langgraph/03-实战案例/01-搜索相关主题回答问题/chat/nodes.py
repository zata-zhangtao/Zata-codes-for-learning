import json
import logging
from typing import Dict, List, Any, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from .config import ChatConfig
from .llm import get_chat_llm
from .prompts import create_chat_messages
from .search import search_web
from .models import ChatState

logger = logging.getLogger(__name__)


def search_node(state: ChatState, config: RunnableConfig) -> Dict[str, Any]:
    """Search node that retrieves information from the web.
    
    Args:
        state: The current chat state
        config: Configuration for the node
        
    Returns:
        Updated state with search results
    """
    logger.info("Search node is running")
    
    # Get configuration
    chat_config = ChatConfig.from_runnable_config(config)
    
    # Get the latest user message
    if not state.messages or not isinstance(state.messages[-1], HumanMessage):
        logger.warning("No user message found, skipping search")
        return {"search_results": None}
    
    query = state.messages[-1].content
    
    # Perform web search
    search_results = search_web(
        query=query,
        max_results=chat_config.max_search_results
    )
    
    logger.info(f"Found {len(search_results)} search results")
    
    return {"search_results": search_results}


def chat_node(state: ChatState, config: RunnableConfig) -> Dict[str, Any]:
    """Chat node that generates a response based on the conversation history and search results.
    
    Args:
        state: The current chat state
        config: Configuration for the node
        
    Returns:
        Updated state with the assistant's response
    """
    logger.info("Chat node is running")
    
    # Create messages for the LLM
    messages = create_chat_messages(vars(state))
    
    # Get the LLM
    llm = get_chat_llm()
    
    # Generate a response
    response = llm.invoke(messages)
    
    # Log the response
    logger.info(f"Chat response: {response.content}")
    
    # Create AI message
    ai_message = AIMessage(content=response.content)
    
    # Return updated state
    return {"messages": state.messages + [ai_message]}


def process_user_message(user_message: str, chat_state: Optional[ChatState] = None) -> ChatState:
    """Process a user message and return the updated state.
    
    This is the main entry point for the chat system.
    
    Args:
        user_message: The user's message
        chat_state: Optional existing chat state
        
    Returns:
        Updated chat state with the assistant's response
    """
    # Create or update chat state
    if chat_state is None:
        chat_state = ChatState()
    
    # Add user message to state
    chat_state.messages.append(HumanMessage(content=user_message))
    
    # Run search if enabled
    if chat_state.enable_search:
        search_results = search_node(chat_state, {})
        chat_state.search_results = search_results.get("search_results")
    
    # Run chat node
    result = chat_node(chat_state, {})
    
    # Update messages in state
    chat_state.messages = result.get("messages", chat_state.messages)
    
    return chat_state 