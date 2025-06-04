import os
import enum
from dataclasses import dataclass
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

# Load environment variables
load_dotenv()


class SearchEngine(enum.Enum):
    """Supported search engines."""
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"


# Default search engine selection from environment variable
SELECTED_SEARCH_ENGINE = os.getenv("SEARCH_API", SearchEngine.TAVILY.value)


@dataclass
class ChatConfig:
    """Configuration for the chat system."""
    
    max_search_results: int = 3  # Maximum number of search results to return
    
    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "ChatConfig":
        """Create a ChatConfig instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        
        return cls(
            max_search_results=int(os.getenv("MAX_SEARCH_RESULTS", 
                                           configurable.get("max_search_results", 3)))
        ) 