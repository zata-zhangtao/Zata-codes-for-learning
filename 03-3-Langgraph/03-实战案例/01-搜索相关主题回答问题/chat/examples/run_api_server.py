#!/usr/bin/env python3
"""
Script to run the API server.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to allow importing chat module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import uvicorn
from chat.api import app


def main():
    """Run the API server."""
    print("Starting DeerFlow Chat API server...")
    print("API documentation will be available at http://localhost:8000/docs")
    
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main() 