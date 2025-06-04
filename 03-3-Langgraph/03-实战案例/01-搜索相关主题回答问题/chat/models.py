from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class ChatState:
    """State for the chat system."""
    
    messages: List[BaseMessage] = []
    search_results: Optional[List[Dict[str, Any]]] = None
    locale: str = "en-US"
    enable_search: bool = True
    
    def __init__(self, **kwargs):
        """Initialize the state with optional overrides."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value) 