"""api.main - FastAPI 애플리케이션 팩토리."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):  # noqa: ARG001
    """애플리케이션 시작/종료 시 실행되는 lifespan 핸들러."""
    setup_logging()
    logger.info("Insurance QA Agent 서버를 시작합니다.")
    yield
    logger.info("Insurance QA Agent 서버를 종료합니다.")


def create_app() -> FastAPI:
    """FastAPI 앱 인스턴스를 생성합니다."""
    application = FastAPI(
        title="Insurance QA Agent",
        description="RAG 기반 보험약관 질의응답 API",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 모니터링 미들웨어
    from api.monitoring import MonitoringMiddleware, metrics_store

    application.add_middleware(MonitoringMiddleware)

    # 헬스 체크
    @application.get("/health")
    async def health_check():
        return {"status": "healthy"}

    # 메트릭 엔드포인트
    @application.get("/metrics")
    async def metrics():
        from api.cache import qa_cache
        return {
            "requests": metrics_store.summary(),
            "cache": qa_cache.stats,
        }

    # 라우터 등록
    from api.routers.admin import router as admin_router
    from api.routers.qa import router as qa_router

    application.include_router(qa_router, prefix="/qa", tags=["Q&A"])
    application.include_router(admin_router, prefix="/admin", tags=["Admin"])

    return application


app = create_app()
