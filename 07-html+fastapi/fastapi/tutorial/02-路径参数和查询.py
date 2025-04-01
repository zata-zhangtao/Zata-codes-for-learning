from fastapi import FastAPI

app = FastAPI()



# 路径参数
@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}

# 查询参数
@app.get("/items/")
def read_item(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
