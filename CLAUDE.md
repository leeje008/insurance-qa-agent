# Insurance QA Agent

보험약관 Q&A 에이전트 — RAG 기반 보험약관 질의응답 시스템

## 프로젝트 구조

- `core/` — 공유 모듈 (상수, 예외, 로깅, 유틸리티)
- `api/` — FastAPI 서버 (라우터, 스키마, 서비스, DB, 검색, LLM, 에이전트)
- `app/` — Streamlit 프론트엔드
- `alembic/` — DB 마이그레이션
- `tests/` — pytest 테스트
- `data/` — PDF 약관 파일 (git 미추적)
- `docs/` — 프로젝트 문서

## 개발 명령어

```bash
uv sync                              # 의존성 설치
uv run ruff check api/ core/         # 린트
uv run pytest -v                     # 테스트
uv run uvicorn api.main:app --reload # 개발 서버
```

## 컨벤션

- Python 3.11+, ruff (line-length=100)
- 한국어 docstring (`__init__.py`)
- 스텁: docstring-only (pass 없음)
- DB: SQLAlchemy ORM + Alembic + PGVector
- LLM: Ollama (qwen2.5:14b) + nomic-embed-text
