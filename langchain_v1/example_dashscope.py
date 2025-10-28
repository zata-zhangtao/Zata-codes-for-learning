"""
Example usage of DashScope (Qwen) models with Langchain.
"""

from config import settings


def example_basic_llm():
    """Example: Basic LLM usage with DashScope."""
    print("=== Basic LLM Example ===")

    # Load the DashScope LLM
    llm = settings.load_dashscope_llm()

    # Simple text generation
    prompt = "What is artificial intelligence?"
    response = llm.invoke(prompt)

    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print()


def example_chat_model():
    """Example: Chat model usage with DashScope."""
    print("=== Chat Model Example ===")

    # Load the DashScope Chat Model
    chat_model = settings.load_dashscope_chat_model()

    # Chat interaction
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessage(content="Tell me a short joke about programming.")
    ]

    response = chat_model.invoke(messages)

    print(f"Messages: {messages}")
    print(f"Response: {response.content}")
    print()


def example_streaming():
    """Example: Streaming responses with DashScope."""
    print("=== Streaming Example ===")

    # Create a streaming config
    from config import Settings
    streaming_settings = Settings(streaming=True)

    # Load with streaming enabled
    llm = streaming_settings.load_dashscope_llm()

    prompt = "Count from 1 to 5 slowly."
    print(f"Prompt: {prompt}")
    print("Response (streaming): ", end="", flush=True)

    # Stream the response
    for chunk in llm.stream(prompt):
        print(chunk, end="", flush=True)

    print("\n")


def example_with_chain():
    """Example: Using DashScope model in a chain."""
    print("=== Chain Example ===")

    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    # Load the chat model
    chat_model = settings.load_dashscope_chat_model()

    # Create a simple chain
    prompt_template = PromptTemplate.from_template(
        "You are an expert in {topic}. Answer the following question: {question}"
    )

    chain = prompt_template | chat_model | StrOutputParser()

    # Invoke the chain
    response = chain.invoke({
        "topic": "machine learning",
        "question": "What is gradient descent?"
    })

    print(f"Response: {response}")
    print()


def example_config_info():
    """Example: Display current DashScope configuration."""
    print("=== Configuration Info ===")

    config = settings.get_dashscope_config()

    print("Current DashScope Configuration:")
    for key, value in config.items():
        if key == "dashscope_api_key" and value:
            # Mask the API key for security
            print(f"  {key}: {value[:8]}...{value[-4:]}")
        else:
            print(f"  {key}: {value}")
    print()


if __name__ == "__main__":
    print("DashScope (Qwen) Model Examples\n")

    # Check if API key is configured
    if not settings.DASHSCOPE_API_KEY:
        print("ERROR: DASHSCOPE_API_KEY is not set!")
        print("Please set it in your .env file:")
        print("  DASHSCOPE_API_KEY=your_api_key_here")
        exit(1)

    # Run examples
    try:
        # Display configuration
        example_config_info()

        # Basic examples
        example_basic_llm()
        example_chat_model()

        # Advanced examples
        # example_streaming()  # Uncomment to test streaming
        example_with_chain()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
