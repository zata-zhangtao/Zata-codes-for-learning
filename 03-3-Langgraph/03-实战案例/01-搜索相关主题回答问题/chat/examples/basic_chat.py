#!/usr/bin/env python3
"""
Basic example of using the chat system.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to allow importing chat module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chat import process_user_message, ChatState


def main():
    """Run a basic example of the chat system."""
    print("Basic Chat Example\n")
    
    # Initialize chat state
    chat_state = ChatState()
    
    # Example conversation
    questions = [
        "What is the current weather in New York?",
        "Tell me about the latest developments in AI",
        "What are some good books to read about history?",
    ]
    
    for i, question in enumerate(questions):
        print(f"User: {question}")
        
        # Process user message
        chat_state = process_user_message(question, chat_state)
        
        # Display assistant response
        if chat_state.messages and len(chat_state.messages) >= (i + 1) * 2:
            print(f"Assistant: {chat_state.messages[-1].content}")
            print()
        else:
            print("Assistant: I'm sorry, something went wrong.")
            print()


if __name__ == "__main__":
    main() 