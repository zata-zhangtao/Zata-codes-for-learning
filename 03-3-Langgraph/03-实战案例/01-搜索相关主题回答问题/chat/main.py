import argparse
import logging
import os
import sys
from typing import List, Optional

from .nodes import process_user_message
from .models import ChatState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Set up environment variables from .env file if it exists."""
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value


def chat_loop():
    """Run an interactive chat loop."""
    print("Welcome to the DeerFlow Chat System!")
    print("Type 'exit', 'quit', or Ctrl+C to end the conversation.")
    print()
    
    # Initialize chat state
    chat_state = ChatState()
    
    try:
        while True:
            # Get user input
            user_input = input("> ")
            
            # Check for exit command
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            # Process user message
            chat_state = process_user_message(user_input, chat_state)
            
            # Display assistant response
            if chat_state.messages and len(chat_state.messages) >= 2:
                print(f"Assistant: {chat_state.messages[-1].content}")
            else:
                print("Assistant: I'm sorry, something went wrong. Please try again.")
    
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        logger.error(f"Error in chat loop: {e}")
        print("An error occurred. Please check the logs for details.")


def main():
    """Main entry point for the chat system."""
    # Set up environment variables
    setup_environment()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="DeerFlow Chat System")
    parser.add_argument("--no-search", action="store_true", help="Disable web search")
    args = parser.parse_args()
    
    # Start chat loop
    chat_loop()


if __name__ == "__main__":
    main() 