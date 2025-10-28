"""
Configuration settings for Langchain application.
Uses Pydantic BaseSettings for environment variable management and validation.
"""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Base settings class for Langchain application configuration."""

    # Application Settings
    app_name: str = Field(default="Langchain App", description="Application name")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # LLM Provider API Keys
    DASHSCOPE_API_KEY: Optional[str] = Field(default=None, env="DASHSCOPE_API_KEY")

    # Model Configuration
    model_name: str = Field(default="qwen-max", description="Default LLM model name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: int = Field(default=2000, gt=0, description="Maximum tokens for generation")
    streaming: bool = Field(default=False, description="Enable streaming responses")

    # Embedding Configuration
    embedding_model: str = Field(default="text-embedding-ada-002", description="Embedding model name")
    embedding_dimension: int = Field(default=1536, description="Embedding vector dimension")

    # Vector Store Configuration
    vector_store_type: str = Field(default="faiss", description="Vector store type (faiss, chroma, pinecone, etc.)")
    vector_store_path: str = Field(default="./vector_store", description="Local vector store path")
    pinecone_api_key: Optional[str] = Field(default=None, description="Pinecone API key")
    pinecone_environment: Optional[str] = Field(default=None, description="Pinecone environment")
    pinecone_index_name: Optional[str] = Field(default=None, description="Pinecone index name")

    # Database Configuration
    database_url: Optional[str] = Field(default=None, description="Database connection URL")
    redis_url: Optional[str] = Field(default="redis://localhost:6379", description="Redis URL for caching")

    # Document Processing
    chunk_size: int = Field(default=1000, gt=0, description="Text chunk size for splitting")
    chunk_overlap: int = Field(default=200, ge=0, description="Overlap between chunks")

    # Search Configuration
    search_k: int = Field(default=4, gt=0, description="Number of documents to retrieve")
    search_type: str = Field(default="similarity", description="Search type (similarity, mmr, etc.)")

    # Agent Configuration
    max_iterations: int = Field(default=10, gt=0, description="Maximum agent iterations")
    agent_verbose: bool = Field(default=True, description="Enable agent verbose output")

    # Web Search & Tools
    serpapi_api_key: Optional[str] = Field(default=None, description="SerpAPI key for web search")
    wolfram_alpha_appid: Optional[str] = Field(default=None, description="Wolfram Alpha App ID")

    # Memory Configuration
    memory_type: str = Field(default="buffer", description="Memory type (buffer, summary, etc.)")
    memory_k: int = Field(default=5, gt=0, description="Number of messages to keep in memory")

    # Rate Limiting
    requests_per_minute: int = Field(default=60, gt=0, description="API requests per minute limit")

    # Timeout Configuration
    request_timeout: int = Field(default=60, gt=0, description="API request timeout in seconds")

    # Callbacks & Monitoring
    langsmith_api_key: Optional[str] = Field(default=None, description="LangSmith API key for tracing")
    langsmith_project: Optional[str] = Field(default=None, description="LangSmith project name")
    enable_tracing: bool = Field(default=False, description="Enable LangSmith tracing")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"
    )

    def get_openai_config(self) -> dict:
        """Get OpenAI configuration dictionary."""
        config = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.openai_api_key:
            config["api_key"] = self.openai_api_key
        if self.openai_api_base:
            config["base_url"] = self.openai_api_base
        return config

    def get_vector_store_config(self) -> dict:
        """Get vector store configuration dictionary."""
        return {
            "type": self.vector_store_type,
            "path": self.vector_store_path,
            "embedding_dimension": self.embedding_dimension,
        }

    def get_retriever_config(self) -> dict:
        """Get retriever configuration dictionary."""
        return {
            "search_type": self.search_type,
            "search_k": self.search_k,
        }

    def load_dashscope_llm(self):
        """
        Load DashScope (Qwen) LLM model.

        Returns:
            Tongyi LLM instance configured with settings

        Raises:
            ValueError: If DASHSCOPE_API_KEY is not set
        """
        if not self.DASHSCOPE_API_KEY:
            raise ValueError(
                "DASHSCOPE_API_KEY is not set. Please set it in your .env file or environment variables."
            )

        try:
            from langchain_community.llms import Tongyi
        except ImportError:
            raise ImportError(
                "langchain-community is required for DashScope. "
                "Install it with: uv add langchain-community"
            )

        return Tongyi(
            model_name=self.model_name,
            dashscope_api_key=self.DASHSCOPE_API_KEY,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            streaming=self.streaming,
        )

    def load_dashscope_chat_model(self):
        """
        Load DashScope (Qwen) Chat Model for conversational interactions.

        Returns:
            ChatTongyi instance configured with settings

        Raises:
            ValueError: If DASHSCOPE_API_KEY is not set
        """
        if not self.DASHSCOPE_API_KEY:
            raise ValueError(
                "DASHSCOPE_API_KEY is not set. Please set it in your .env file or environment variables."
            )

        try:
            from langchain_community.chat_models import ChatTongyi
        except ImportError:
            raise ImportError(
                "langchain-community is required for DashScope. "
                "Install it with: uv add langchain-community"
            )

        return ChatTongyi(
            model_name=self.model_name,
            dashscope_api_key=self.DASHSCOPE_API_KEY,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            streaming=self.streaming,
        )

    def get_dashscope_config(self) -> dict:
        """Get DashScope configuration dictionary."""
        return {
            "model_name": self.model_name,
            "dashscope_api_key": self.DASHSCOPE_API_KEY,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "streaming": self.streaming,
        }


# Global settings instance
settings = Settings()


# Convenience function to reload settings
def reload_settings() -> Settings:
    """Reload settings from environment."""
    global settings
    settings = Settings()
    return settings


if __name__ == "__main__":
    # Print current settings for debugging
    print("Current Settings:")
    print(f"App Name: {settings.app_name}")
    print(f"Model: {settings.model_name}")
    print(f"Temperature: {settings.temperature}")
    print(f"Vector Store: {settings.vector_store_type}")
    print(f"Debug Mode: {settings.debug}")
