from fastapi import FastAPI
from .routers.hackrx import router as hackrx_router

app = FastAPI(title="HackRX Intelligent Query Retrieval")

app.include_router(hackrx_router, prefix="/api/v1")

@app.get("/")
def root():
    return {"status": "ok"}
