from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import os

app = FastAPI(
    title="FastAPI Docker Demo",
    description="A simple FastAPI application for Docker containerization learning",
    version="1.0.0"
)

class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = None

items_db: Dict[int, Item] = {}

@app.get("/")
async def read_root():
    return {
        "message": "欢迎使用 FastAPI Docker 演示项目",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "fastapi-docker-demo"
    }

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item_id": item_id, "item": items_db[item_id]}

@app.post("/items/")
async def create_item(item: Item):
    item_id = len(items_db) + 1
    items_db[item_id] = item
    return {"item_id": item_id, "item": item, "message": "Item created successfully"}

@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    items_db[item_id] = item
    return {"item_id": item_id, "item": item, "message": "Item updated successfully"}

@app.delete("/items/{item_id}")
async def delete_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    deleted_item = items_db.pop(item_id)
    return {"item_id": item_id, "item": deleted_item, "message": "Item deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    workers = int(os.getenv("WORKERS", 1))
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers if workers > 1 else None,
        reload=False
    )