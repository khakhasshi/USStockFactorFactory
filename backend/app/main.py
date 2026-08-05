from contextlib import asynccontextmanager
from pathlib import Path

import asyncio
import logging

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import router
from .config import (
    ALLOW_REMOTE_UNAUTHENTICATED,
    HOST,
    PORT,
    is_loopback_host,
)
from .db import init_db
from .observability import OBSERVABILITY
from .seed import seed_classics

FRONTEND = Path(__file__).resolve().parent.parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await OBSERVABILITY.start()
    seed_task = asyncio.create_task(
        seed_classics(),
        name="startup.seed_classics",
    )  # 首次启动播种经典因子 (后台, 幂等)
    OBSERVABILITY.track_task(seed_task)
    try:
        yield
    finally:
        if not seed_task.done():
            seed_task.cancel()
        await asyncio.gather(seed_task, return_exceptions=True)
        await OBSERVABILITY.stop()


app = FastAPI(
    title="FactorFactory Research Service",
    version="4.1",
    lifespan=lifespan,
)


def _route_template(request: Request) -> str:
    route = getattr(request.scope.get("route"), "path", None)
    if route is None:
        return "/__unmatched__"
    return route or "/__frontend__"


@app.middleware("http")
async def request_telemetry(request: Request, call_next):
    """Attach request IDs and collect bounded latency/error telemetry."""
    request_id, started = OBSERVABILITY.begin_request()
    try:
        response = await call_next(request)
    except Exception as exc:
        route = _route_template(request)
        duration_ms = OBSERVABILITY.finish_request(
            request_id=request_id,
            method=request.method,
            path=route,
            status=500,
            started=started,
            error=exc,
        )
        logging.getLogger("factorfactory.http").exception(
            "Unhandled request %s %s request_id=%s",
            request.method,
            route,
            request_id,
        )
        return JSONResponse(
            {
                "detail": "内部服务器错误",
                "request_id": request_id,
            },
            status_code=500,
            headers={
                "X-Request-ID": request_id,
                "Server-Timing": f"app;dur={duration_ms:.3f}",
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "Referrer-Policy": "no-referrer",
            },
        )
    route = _route_template(request)
    duration_ms = OBSERVABILITY.finish_request(
        request_id=request_id,
        method=request.method,
        path=route,
        status=response.status_code,
        started=started,
    )
    response.headers["X-Request-ID"] = request_id
    response.headers["Server-Timing"] = f"app;dur={duration_ms:.3f}"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    # Some browsers request this path even when the page uses an inline icon.
    return Response(status_code=204)


app.include_router(router)
app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")


def main() -> None:
    if not is_loopback_host(HOST) and not ALLOW_REMOTE_UNAUTHENTICATED:
        raise RuntimeError(
            "拒绝在无认证状态下监听非本机地址。若已确认处于隔离网络，"
            "请显式设置 FF_ALLOW_REMOTE_UNAUTHENTICATED=1。"
        )
    uvicorn.run("app.main:app", host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
