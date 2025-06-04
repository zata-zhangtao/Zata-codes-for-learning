#!/usr/bin/env python3
"""
Example of using the chat system without web search.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to allow importing chat module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chat import process_user_message, ChatState


def main():
    """Run an example of the chat system without web search."""
    print("Chat Example Without Web Search\n")
    
    # Initialize chat state with search disabled
    chat_state = ChatState(enable_search=False)
    
    # Example conversation
    questions = [
        "Tell me a short story about a robot learning to paint",
        "What would be a good name for this robot?",
        "Can you describe one of its paintings?",
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