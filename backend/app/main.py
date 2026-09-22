from contextlib import asynccontextmanager
from pathlib import Path

import asyncio
import logging

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select

from .api.routes import invalidate_panel_dependents, router
from .config import (
    ALLOW_REMOTE_UNAUTHENTICATED,
    AUTOSTART_RESEARCH,
    HOST,
    PANEL_AUTO_RELOAD,
    PANEL_WATCH_SECONDS,
    PORT,
    SERVICE_ARCHITECTURE,
    SERVICE_INSTANCE,
    SERVICE_MARKET,
    is_loopback_host,
)
from .data.panel import PanelStore
from .db import SessionLocal, init_db
from .models import Experiment
from .observability import OBSERVABILITY
from .orchestrator import EngineManager
from .seed import seed_classics
from .trade_plans import router as trade_plan_router, scheduler as trade_plan_scheduler, initialize as initialize_trade_plans

FRONTEND = Path(__file__).resolve().parent.parent.parent / "frontend"


async def _watch_panel_sources() -> None:
    """Detect stable source changes and hot-swap loaded panel generations."""
    logger = logging.getLogger("factorfactory.panel_reload")
    while True:
        await asyncio.sleep(PANEL_WATCH_SECONDS)
        try:
            results = await asyncio.to_thread(
                PanelStore.reload_changed_instances
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - watcher must survive one bad poll
            logger.exception("Panel change detection poll failed")
            continue
        reloaded = [row for row in results if row.get("status") == "reloaded"]
        if reloaded:
            invalidation = invalidate_panel_dependents()
            for row in reloaded:
                logger.info(
                    "Panel hot reload completed market=%s generation=%s "
                    "date_max=%s duration_ms=%s cache_removed=%s",
                    row.get("market"),
                    row.get("generation"),
                    row.get("date_max"),
                    row.get("duration_ms"),
                    invalidation["screener_entries_removed"],
                )
        for row in results:
            if row.get("status") in {
                "reload_failed",
                "source_changed_during_reload",
            }:
                logger.warning(
                    "Panel hot reload retained old generation market=%s "
                    "status=%s generation=%s error=%s",
                    row.get("market"),
                    row.get("status"),
                    row.get("generation"),
                    row.get("error") or "source changed during reload",
                )


async def _supervise_bound_research() -> None:
    """Keep explicitly bound continuous workers alive for this service.

    Experiment rows remain the durable desired-state declaration.  A provider
    or worker failure therefore pauses with bounded exponential backoff and is
    resumed without losing already committed nodes, trials, or LLM audits.
    """
    logger = logging.getLogger("factorfactory.autostart")
    retry_attempts: dict[int, int] = {}
    next_retry_at: dict[int, float] = {}
    while True:
        try:
            async with SessionLocal() as session:
                rows = list((await session.scalars(
                    select(Experiment).where(Experiment.status == "open")
                )).all())
            desired = []
            for experiment in rows:
                config = dict(experiment.research_config or {})
                if not config.get("service_autostart"):
                    continue
                if str(config.get("service_instance") or "") != SERVICE_INSTANCE:
                    continue
                desired.append(experiment.id)

            manager = EngineManager.get()
            now = asyncio.get_running_loop().time()
            for experiment_id in desired:
                worker = manager.worker(experiment_id)
                if worker is not None and worker.running:
                    retry_attempts[experiment_id] = 0
                    next_retry_at[experiment_id] = 0.0
                    continue
                if now < next_retry_at.get(experiment_id, 0.0):
                    continue
                result = await manager.start("v2", experiment_id)
                if result.get("ok"):
                    logger.info(
                        "Autostarted continuous research experiment=%s service=%s",
                        experiment_id,
                        SERVICE_INSTANCE,
                    )
                    retry_attempts[experiment_id] = 0
                    next_retry_at[experiment_id] = now + 30.0
                else:
                    attempt = retry_attempts.get(experiment_id, 0) + 1
                    retry_attempts[experiment_id] = attempt
                    delay = min(900.0, 30.0 * (2 ** min(attempt - 1, 5)))
                    next_retry_at[experiment_id] = now + delay
                    logger.warning(
                        "Research autostart deferred experiment=%s delay=%ss reason=%s",
                        experiment_id,
                        int(delay),
                        result.get("msg"),
                    )
            await asyncio.sleep(15.0)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Continuous research supervisor poll failed")
            await asyncio.sleep(30.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await initialize_trade_plans()
    await OBSERVABILITY.start()
    trade_plan_task = asyncio.create_task(trade_plan_scheduler(), name="manual_trade_plans")
    seed_task = asyncio.create_task(
        seed_classics(),
        name="startup.seed_classics",
    )  # 首次启动播种经典因子 (后台, 幂等)
    OBSERVABILITY.track_task(seed_task)
    panel_watch_task = None
    research_supervisor_task = None
    if PANEL_AUTO_RELOAD:
        panel_watch_task = asyncio.create_task(
            _watch_panel_sources(),
            name="runtime.panel_hot_reload",
        )
        OBSERVABILITY.track_task(panel_watch_task)
    if AUTOSTART_RESEARCH:
        research_supervisor_task = asyncio.create_task(
            _supervise_bound_research(),
            name=f"runtime.research_autostart.{SERVICE_INSTANCE}",
        )
        OBSERVABILITY.track_task(research_supervisor_task)
    try:
        yield
    finally:
        trade_plan_task.cancel()
        await asyncio.gather(trade_plan_task, return_exceptions=True)
        if not seed_task.done():
            seed_task.cancel()
        if panel_watch_task is not None and not panel_watch_task.done():
            panel_watch_task.cancel()
        if (
            research_supervisor_task is not None
            and not research_supervisor_task.done()
        ):
            research_supervisor_task.cancel()
        await EngineManager.get().stop()
        await asyncio.gather(
            seed_task,
            *([panel_watch_task] if panel_watch_task is not None else []),
            *(
                [research_supervisor_task]
                if research_supervisor_task is not None
                else []
            ),
            return_exceptions=True,
        )
        await OBSERVABILITY.stop()


app = FastAPI(
    title="FactorFactory Research Service",
    version="4.1",
    lifespan=lifespan,
)


@app.get("/api/service/identity")
async def service_identity() -> dict:
    return {
        "service_instance": SERVICE_INSTANCE,
        "service_architecture": SERVICE_ARCHITECTURE or None,
        "port": PORT,
        "research_autostart": AUTOSTART_RESEARCH,
        "service_market": SERVICE_MARKET or None,
    }


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
    is_embedded_document = (
        request.url.path.startswith("/api/research-documents/")
        and request.url.path.endswith("/html")
    )
    is_embedded_html = is_embedded_document or (
        request.url.path.startswith("/api/leaderboards/")
        and "/files/" in request.url.path
    )
    response.headers["X-Frame-Options"] = (
        "SAMEORIGIN" if is_embedded_html else "DENY"
    )
    if is_embedded_html:
        response.headers["Content-Security-Policy"] = "frame-ancestors 'self'"
    response.headers["Referrer-Policy"] = "no-referrer"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    # Some browsers request this path even when the page uses an inline icon.
    return Response(status_code=204)


app.include_router(router)
app.include_router(trade_plan_router)
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
