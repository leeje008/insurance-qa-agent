# syntax=docker/dockerfile:1

# 1단계(build stage): 의존성 설치와 가상환경 생성
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /api

# Python 패키지 빌드에 필요한 시스템 의존성 설치
#  - libpq-dev: asyncpg / psycopg2 빌드용 PostgreSQL 헤더
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# uv 패키지 매니저 설치
RUN pip install --upgrade pip && pip install --no-cache-dir uv

# 의존성 정의 파일 복사 (lock 기반 재현성 확보)
COPY pyproject.toml uv.lock ./

# uv가 /opt/venv에 가상환경을 생성하고 프로덕션 의존성만 설치
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv sync --no-dev


# 2단계(runtime stage): 경량 프로덕션 이미지
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /api

# 런타임 최소 시스템 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 빌더 단계에서 만든 가상환경 복사
COPY --from=builder /opt/venv /opt/venv

# 애플리케이션 코드 복사 및 로그 디렉터리 생성
COPY . /api/
RUN mkdir -p /api/logs
RUN chmod +x /api/entrypoint.sh

# FastAPI 포트 노출
EXPOSE 8000

# 엔트리포인트: alembic 마이그레이션 후 Gunicorn 실행
ENTRYPOINT ["/api/entrypoint.sh"]
CMD ["gunicorn", "api.main:app", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:8000", "--timeout", "300", "--max-requests", "2000", "--max-requests-jitter", "200"]
