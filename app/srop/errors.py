from fastapi import Request
from fastapi.responses import JSONResponse


class HelixError(Exception):
    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

class SessionNotFoundError(HelixError):
    status_code = 404
    error_code = "SESSION_NOT_FOUND"

class UpstreamTimeoutError(HelixError):
    status_code = 504
    error_code = "UPSTREAM_TIMEOUT"

class TraceNotFoundError(HelixError):
    status_code = 404
    error_code = "TRACE_NOT_FOUND"

class GuardrailRefusalError(HelixError):
    status_code = 400
    error_code = "GUARDRAIL_REFUSAL"

async def helix_error_handler(request: Request, exc: HelixError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error_code": exc.error_code, "message": exc.message},
    )
