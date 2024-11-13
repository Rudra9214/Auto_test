# apps_folder/main.py

from fastapi import FastAPI
from .database.db import Database
from .routers import evaluate

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
