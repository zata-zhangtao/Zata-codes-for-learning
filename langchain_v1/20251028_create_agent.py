"""
Simple Agent Examples using LangChain and DashScope (Qwen).

Based on: https://docs.langchain.com/oss/python/langchain/agents

This module demonstrates how to create and use agents with different tools
and configurations using the DashScope platform.
"""

from config import settings
from langchain.agents import create_agent
from langchain_core.tools import tool


# ============================================================================
# Define Tools
# ============================================================================

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city.

    Args:
        city: The name of the city to get weather for.

    Returns:
        A string describing the weather.
    """
    # This is a mock function - in production, you'd call a real weather API
    weather_data = {
        "beijing": "Sunny, 25°C",
        "shanghai": "Cloudy, 22°C",
        "guangzhou": "Rainy, 28°C",
        "shenzhen": "Sunny, 30°C",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2", "10 * 5").

    Returns:
        The result of the calculation.
    """
    try:
        # Safe evaluation for simple math expressions
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result is: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"


@tool
def search_knowledge(query: str) -> str:
    """Search for information in the knowledge base.

    Args:
        query: The search query.

    Returns:
        Relevant information from the knowledge base.
    """
    # This is a mock function - in production, you'd search a real database or vector store
    knowledge = {
        "python": "Python is a high-level programming language known for its simplicity and readability.",
        "langchain": "LangChain is a framework for developing applications powered by language models.",
        "ai": "Artificial Intelligence is the simulation of human intelligence by machines.",
    }

    for key, value in knowledge.items():
        if key in query.lower():
            return value

    return f"No information found for: {query}"


# ============================================================================
# Example 1: Simple Agent
# ============================================================================

def example_simple_agent():
    """Simplest possible agent example."""
    print("=" * 80)
    print("Example 1: Simple Agent")
    print("=" * 80)

    # Load the DashScope chat model
    llm = settings.load_dashscope_chat_model()

    # Create agent with a single tool
    agent_graph = create_agent(
        model=llm,
        tools=[get_weather],
        system_prompt="You are a helpful weather assistant. Use the get_weather tool to answer questions."
    )

    # Ask a question
    question = "What's the weather like in Beijing?"
    print(f"\nQuestion: {question}")
    print("-" * 80)

    # Invoke the agent with messages format
    result = agent_graph.invoke({
        "messages": [{"role": "user", "content": question}]
    })

    # Get the last message from the result
    last_message = result["messages"][-1]
    print(f"\nAnswer: {last_message.content}")
    print("=" * 80)


# ============================================================================
# Example 2: Agent with Multiple Tools
# ============================================================================

def example_multi_tool_agent():
    """Create an agent with multiple tools."""
    print("\n" + "=" * 80)
    print("Example 2: Agent with Multiple Tools")
    print("=" * 80)

    # Load the DashScope chat model
    llm = settings.load_dashscope_chat_model()

    # Define the tools
    tools = [get_weather, calculate, search_knowledge]

    # Create the agent
    agent_graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt="You are a helpful assistant. Use the available tools to answer questions accurately."
    )

    # Test the agent with different questions
    questions = [
        "What's the weather in Shanghai?",
        "Calculate 123 * 456",
        "Tell me about Python",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 80)

        # Invoke the agent
        result = agent_graph.invoke({
            "messages": [{"role": "user", "content": question}]
        })

        # Get the answer
        last_message = result["messages"][-1]
        print(f"Answer: {last_message.content}")
        print("=" * 80)


# ============================================================================
# Example 3: Agent with Custom System Prompt
# ============================================================================

def example_custom_prompt_agent():
    """Create an agent with a custom system prompt."""
    print("\n" + "=" * 80)
    print("Example 3: Agent with Custom System Prompt")
    print("=" * 80)

    # Load the model
    llm = settings.load_dashscope_chat_model()

    # Define tools
    tools = [get_weather, calculate]

    # Custom system prompt that defines personality
    system_prompt = """You are a friendly and enthusiastic assistant named Qwen.
You love helping people and always respond with positivity and encouragement.
When you use tools, explain what you're doing and why.
Always end your response with a helpful tip or words of encouragement."""

    # Create agent with custom prompt
    agent_graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )

    # Test with a multi-step question
    question = "What's the weather in Shenzhen and what is 50 + 50?"
    print(f"\nQuestion: {question}")
    print("-" * 80)

    result = agent_graph.invoke({
        "messages": [{"role": "user", "content": question}]
    })

    last_message = result["messages"][-1]
    print(f"Answer: {last_message.content}")
    print("=" * 80)


# ============================================================================
# Example 4: Conversational Agent with Memory
# ============================================================================

def example_conversational_agent():
    """Create an agent with conversation memory."""
    print("\n" + "=" * 80)
    print("Example 4: Conversational Agent with Memory")
    print("=" * 80)

    # Load the model
    llm = settings.load_dashscope_chat_model()

    # Define tools
    tools = [get_weather, calculate]

    # Create agent
    agent_graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt="You are a helpful assistant. Remember the conversation context and refer back to previous messages when relevant."
    )

    # Simulate a multi-turn conversation
    conversations = [
        "What's the weather in Beijing?",
        "And what about Shanghai?",  # Uses context from previous question
        "Which city is warmer?",  # Uses context from both previous questions
    ]

    # Start with empty messages list
    messages = []

    for question in conversations:
        print(f"\nUser: {question}")
        print("-" * 80)

        # Add user message to conversation
        messages.append({"role": "user", "content": question})

        # Invoke the agent with full message history
        result = agent_graph.invoke({"messages": messages})

        # Get the agent's response
        agent_response = result["messages"][-1]
        print(f"Assistant: {agent_response.content}")

        # Update messages with the full conversation history from result
        messages = result["messages"]
        print("=" * 80)


# ============================================================================
# Example 5: Streaming Agent Responses
# ============================================================================

def example_streaming_agent():
    """Create an agent with streaming responses."""
    print("\n" + "=" * 80)
    print("Example 5: Streaming Agent Responses")
    print("=" * 80)

    # Load the model
    llm = settings.load_dashscope_chat_model()

    # Create agent
    agent_graph = create_agent(
        model=llm,
        tools=[get_weather, calculate],
        system_prompt="You are a helpful assistant."
    )

    # Ask a question
    question = "What's the weather in Guangzhou and calculate 100 * 200?"
    print(f"\nQuestion: {question}")
    print("-" * 80)
    print("\nStreaming response:")
    print("-" * 80)

    # Stream the agent's responses
    for chunk in agent_graph.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="updates"
    ):
        print(chunk)

    print("=" * 80)


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run all examples."""
    print("\nLangChain Agent Examples with DashScope (Qwen)\n")

    # Check API key
    if not settings.DASHSCOPE_API_KEY:
        print("ERROR: DASHSCOPE_API_KEY is not set!")
        print("Please set it in your .env file:")
        print("  DASHSCOPE_API_KEY=your_api_key_here")
        return

    print(f"Using model: {settings.model_name}")
    print(f"Temperature: {settings.temperature}")
    print(f"Max tokens: {settings.max_tokens}\n")

    try:
        # Run examples (uncomment the ones you want to run)
        example_simple_agent()
        example_multi_tool_agent()
        example_custom_prompt_agent()
        example_conversational_agent()
        example_streaming_agent()

        print("\n\nDone! Uncomment other examples in main() to try them.")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
