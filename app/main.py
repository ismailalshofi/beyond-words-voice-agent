from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.db import init_db
from app.routers import assistants, realtime

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()
    yield


app = FastAPI(title="Beyond Words", lifespan=lifespan)
app.include_router(assistants.router)
app.include_router(realtime.router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}
