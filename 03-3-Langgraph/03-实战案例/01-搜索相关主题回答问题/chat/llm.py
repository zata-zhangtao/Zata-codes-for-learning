import os
import yaml
import logging
from typing import Dict, Any
from pathlib import Path

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# Cache for LLM instances
_llm_cache = {}


def load_config() -> Dict[str, Any]:
    """Load configuration from conf.yaml file."""
    config_path = Path(__file__).parent.parent / "conf.yaml"
    
    if not config_path.exists():
        logger.warning(f"Configuration file not found: {config_path}")
        # Return default configuration
        return {
            "CHAT_MODEL": {
                "model": "gpt-3.5-turbo",
                "temperature": 0.7,
            }
        }
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}


def get_chat_llm() -> ChatOpenAI:
    """Get the chat LLM instance, creating it if necessary."""
    from langchain_community.chat_models import ChatTongyi
    
    llm = ChatTongyi(
        model = "qwen-max"
    )

    return llm
    
    if "chat" in _llm_cache:
        return _llm_cache["chat"]
    
    config = load_config()
    
    # Get chat model configuration from config
    model_config = config.get("CHAT_MODEL", {})
    
    # Override with environment variables if present
    if api_key := os.getenv("OPENAI_API_KEY"):
        model_config["api_key"] = api_key
    
    if base_url := os.getenv("OPENAI_API_BASE"):
        model_config["base_url"] = base_url
    
    if model := os.getenv("OPENAI_MODEL"):
        model_config["model"] = model
    
    # Create LLM instance
    try:
        llm = ChatOpenAI(**model_config)
        _llm_cache["chat"] = llm
        return llm
    except Exception as e:
        logger.error(f"Error creating chat LLM: {e}")
        # Fallback to defaults
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        _llm_cache["chat"] = llm
        return llm 