"""api.monitoring - 요청 모니터링 미들웨어 및 메트릭."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


@dataclass
class EndpointMetrics:
    """엔드포인트별 메트릭."""

    request_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.request_count == 0:
            return 0.0
        return round(self.total_latency_ms / self.request_count, 2)

    def record(self, latency_ms: float, is_error: bool = False) -> None:
        self.request_count += 1
        self.total_latency_ms += latency_ms
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        if is_error:
            self.error_count += 1

    def to_dict(self) -> dict:
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "avg_latency_ms": self.avg_latency_ms,
            "min_latency_ms": (
                round(self.min_latency_ms, 2)
                if self.min_latency_ms != float("inf") else 0
            ),
            "max_latency_ms": round(self.max_latency_ms, 2),
        }


class MetricsStore:
    """글로벌 메트릭 저장소."""

    def __init__(self) -> None:
        self._endpoints: dict[str, EndpointMetrics] = defaultdict(EndpointMetrics)
        self._start_time = time.time()

    def record(
        self, method: str, path: str, latency_ms: float, status_code: int,
    ) -> None:
        key = f"{method} {path}"
        is_error = status_code >= 400
        self._endpoints[key].record(latency_ms, is_error)

    def summary(self) -> dict:
        uptime = time.time() - self._start_time
        total_requests = sum(m.request_count for m in self._endpoints.values())
        total_errors = sum(m.error_count for m in self._endpoints.values())

        return {
            "uptime_seconds": round(uptime, 1),
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": (
                round(total_errors / total_requests, 4)
                if total_requests > 0 else 0.0
            ),
            "endpoints": {
                k: v.to_dict() for k, v in self._endpoints.items()
            },
        }

    def reset(self) -> None:
        self._endpoints.clear()
        self._start_time = time.time()


# 글로벌 인스턴스
metrics_store = MetricsStore()


class MonitoringMiddleware(BaseHTTPMiddleware):
    """요청 응답 시간 및 상태 코드를 기록하는 미들웨어."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint,
    ) -> Response:
        start = time.time()
        response = await call_next(request)
        latency_ms = (time.time() - start) * 1000

        path = request.url.path
        # 헬스체크/정적 파일은 제외
        if path not in ("/health", "/favicon.ico"):
            metrics_store.record(
                request.method, path, latency_ms, response.status_code,
            )
            if latency_ms > 5000:
                logger.warning(
                    "느린 요청: %s %s — %.0fms",
                    request.method, path, latency_ms,
                )

        return response
