# apps_folder/main.py

from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from apps_folder.database.db import Database
from apps_folder.routers import evaluate

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    await Database.connect()

@app.on_event("shutdown")
async def shutdown_event():
    await Database.disconnect()

@app.get("/")
async def read_root():
    return {"message": "App Started"}

app.include_router(evaluate.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("apps_folder.main:app", host="0.0.0.0", port=8000, reload=True)