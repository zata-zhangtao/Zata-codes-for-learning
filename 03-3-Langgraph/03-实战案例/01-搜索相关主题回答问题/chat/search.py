import json
import logging
import os
from typing import List, Dict, Any, Optional

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import BaseTool

from .config import SELECTED_SEARCH_ENGINE, SearchEngine

logger = logging.getLogger(__name__)

try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    TAVILY_AVAILABLE = True
except ImportError:
    logger.warning("Tavily search not available. Install with: pip install tavily-python")
    TAVILY_AVAILABLE = False


def create_search_tool(max_search_results: int = 3) -> BaseTool:
    """Create a web search tool based on the selected search engine.
    
    Args:
        max_search_results: Maximum number of search results to return
        
    Returns:
        A search tool for the selected search engine
    """
    if SELECTED_SEARCH_ENGINE == SearchEngine.TAVILY.value:
        if not TAVILY_AVAILABLE:
            logger.warning("Tavily not available, falling back to DuckDuckGo")
            return DuckDuckGoSearchResults(max_results=max_search_results)
        
        return TavilySearchResults(
            max_results=max_search_results,
            include_raw_content=True,
        )
    else:
        return DuckDuckGoSearchResults(max_results=max_search_results)


def search_web(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """Search the web for the given query.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        List of search results, each with title and content
    """
    search_tool = create_search_tool(max_results)
    try:
        raw_results = search_tool.invoke(query)
        
        # Format results consistently regardless of search engine
        if SELECTED_SEARCH_ENGINE == SearchEngine.TAVILY.value and TAVILY_AVAILABLE:
            if isinstance(raw_results, list):
                return [
                    {"title": result.get("title", ""), "content": result.get("content", "")}
                    for result in raw_results
                ]
            else:
                logger.error(f"Unexpected Tavily search result format: {raw_results}")
                return []
        else:
            # Parse DuckDuckGo results
            try:
                results = []
                if isinstance(raw_results, str):
                    items = json.loads(raw_results)
                    for item in items:
                        results.append({
                            "title": item.get("title", ""),
                            "content": item.get("snippet", ""),
                            "url": item.get("link", "")
                        })
                return results
            except Exception as e:
                logger.error(f"Error parsing search results: {e}")
                return []
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return [] 