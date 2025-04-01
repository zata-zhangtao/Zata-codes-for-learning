from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
app = FastAPI()


@app.get("/async/")
async def read_async():
    await asyncio.sleep(1)  # 模拟异步操作
    return {"message": "This is async"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

