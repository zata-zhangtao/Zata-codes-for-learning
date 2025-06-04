"""
Simple API server for the chat system using FastAPI.
"""

import logging
import os
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .nodes import process_user_message
from .models import ChatState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="DeerFlow Chat API",
    description="API for the DeerFlow Chat system",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for chat sessions
chat_sessions: Dict[str, ChatState] = {}


class ChatMessage(BaseModel):
    """Chat message model."""
    
    role: str = Field(..., description="Role of the message sender (user or assistant)")
    content: str = Field(..., description="Content of the message")


class ChatRequest(BaseModel):
    """Chat request model."""
    
    session_id: Optional[str] = Field(None, description="Session ID (optional)")
    message: str = Field(..., description="User message")
    enable_search: Optional[bool] = Field(True, description="Whether to enable web search")


class ChatResponse(BaseModel):
    """Chat response model."""
    
    session_id: str = Field(..., description="Session ID")
    message: str = Field(..., description="Assistant message")
    messages: List[ChatMessage] = Field(..., description="Conversation history")
    search_results: Optional[List[Dict]] = Field(None, description="Search results if available")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest = Body(...)):
    """Process a chat message and return a response."""
    try:
        # Get or create session
        session_id = request.session_id or os.urandom(16).hex()
        chat_state = chat_sessions.get(session_id, ChatState(enable_search=request.enable_search))
        
        # Process message
        updated_state = process_user_message(request.message, chat_state)
        
        # Save updated state
        chat_sessions[session_id] = updated_state
        
        # Get assistant response
        if updated_state.messages and len(updated_state.messages) >= 2:
            assistant_message = updated_state.messages[-1].content
        else:
            assistant_message = "I'm sorry, something went wrong."
        
        # Convert messages to response format
        messages = []
        for msg in updated_state.messages:
            role = "user" if msg.type == "human" else "assistant"
            messages.append(ChatMessage(role=role, content=msg.content))
        
        # Create response
        return ChatResponse(
            session_id=session_id,
            message=assistant_message,
            messages=messages,
            search_results=updated_state.search_results,
        )
    
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 