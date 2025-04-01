from fastapi import FastAPI
from fastapi import Depends
from pydantic import BaseModel
import asyncio
app = FastAPI()



# region 简单的依赖注入
async def common_parameters(q: str | None = None, skip: int = 0):
    return {"q": q, "skip": skip}
@app.get("/depends/") #  http://localhost:8000/items/?q=foo&skip=5&limit=20
async def read_depends(commons: dict = Depends(common_parameters)):
    return commons

# endregion

# region 异步依赖注入

async def get_db():
    db = "Database connection established"  # 模拟数据库连接
    try:
        yield db
    finally:
        print("Database connection closed")

@app.get("/users/") #  http://localhost:8000/users/
async def read_users(db: str = Depends(get_db)):
    return {"db": db}

# endregion


# region 类作为依赖
class CommonQueryParams:
    def __init__(self, q: str = None, skip: int = 0, limit: int = 10):
        self.q = q
        self.skip = skip
        self.limit = limit

@app.get("/items/") # http://localhost:8000/items/?q=test&skip=2
def read_items(commons: CommonQueryParams = Depends(CommonQueryParams)):
    return {"q": commons.q, "skip": commons.skip, "limit": commons.limit}
# endregion



# region 嵌套依赖注入
def query(q: str = None):
    return {"q": q}
def pagination(skip: int = 0, limit: int = 10, query_params: dict = Depends(query)):
    return {"skip": skip, "limit": limit, "query": query_params["q"]}
@app.get("/items/") #http://localhost:8000/items/?q=foo&skip=5
def read_items(params: dict = Depends(pagination)):
    return params
# endregion


# region 全局依赖注入

if False:  # 我这里把他关了
    app = FastAPI(dependencies=[Depends(common_parameters)]) #dependencies 参数：所有路由都会强制使用这个依赖。

    @app.get("/items/")
    def read_items():
        return {"message": "Items with global dependency"}

# endregion

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

