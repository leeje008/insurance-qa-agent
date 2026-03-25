"""api.security - API 키 인증 + Rate Limiting 미들웨어."""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from api.config import settings

# ---------------------------------------------------------------------------
# API 키 인증
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(_api_key_header)) -> str | None:
    """API 키 검증 의존성.

    settings.api_key가 빈 문자열이면 인증을 건너뛴다 (개발 모드).
    """
    if not settings.api_key:
        return None

    if not api_key or api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="유효하지 않은 API 키입니다")

    return api_key


# ---------------------------------------------------------------------------
# Rate Limiting 미들웨어
# ---------------------------------------------------------------------------

class RateLimitMiddleware(BaseHTTPMiddleware):
    """IP별 sliding window 기반 Rate Limiting.

    settings.rate_limit_rpm이 0이면 비활성화.
    """

    def __init__(self, app, rpm: int = 0) -> None:
        super().__init__(app)
        self.rpm = rpm or settings.rate_limit_rpm
        self._requests: dict[str, deque] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if self.rpm <= 0:
            return await call_next(request)

        # 헬스체크와 메트릭은 Rate Limit 제외
        if request.url.path in ("/health", "/metrics"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = self._requests[client_ip]

        # 1분 이전 요청 제거
        while window and window[0] < now - 60:
            window.popleft()

        if len(window) >= self.rpm:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"요청 한도 초과 (분당 {self.rpm}회)",
                    "retry_after_seconds": int(60 - (now - window[0])),
                },
            )

        window.append(now)
        return await call_next(request)
