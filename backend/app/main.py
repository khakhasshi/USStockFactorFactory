from contextlib import asynccontextmanager
from pathlib import Path

import asyncio

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api.routes import router
from .config import PORT
from .db import init_db
from .seed import seed_classics

FRONTEND = Path(__file__).resolve().parent.parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    asyncio.create_task(seed_classics())  # 首次启动播种经典因子 (后台, 幂等)
    yield


app = FastAPI(title="USStockFactorFactory", lifespan=lifespan)
app.include_router(router)
app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")


def main() -> None:
    uvicorn.run("app.main:app", host="0.0.0.0", port=PORT, log_level="info")


if __name__ == "__main__":
    main()
