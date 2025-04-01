from fastapi import FastAPI

app = FastAPI()

## hello world
@app.get("/")
def read_root():
    return {"message": "Hello World"}

# 然后你就可以使用 uvicorn 01-simple_start:app  --reload  去开启服务了

# 当然你也可以直接运行，使用下面的方式去开启服务




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

