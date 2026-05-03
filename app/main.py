from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import routes_chat, routes_sessions, routes_traces
from app.db.session import init_db
from app.obs.logging import configure_logging
from app.srop.errors import HelixError, helix_error_handler


from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    await init_db()
    yield


app = FastAPI(title="Helix SROP", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_sessions.router, prefix="/v1")
app.include_router(routes_chat.router, prefix="/v1")
app.include_router(routes_traces.router, prefix="/v1")


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


# TODO: register exception handlers for HelixError subclasses
app.add_exception_handler(HelixError, helix_error_handler)
