# Insurance QA Agent

RAG 기반 보험약관 질의응답 시스템입니다.

## 주요 기능

- **약관 PDF 파싱**: 보험약관 PDF를 관/조/항/호 계층 구조로 자동 파싱
- **하이브리드 검색**: PGVector 시맨틱 검색 + 키워드 검색 + RRF 융합
- **LLM 답변 생성**: Ollama 로컬 LLM 기반 근거 조항 인용 답변
- **답변 검증**: 환각 감지 및 신뢰도 평가 (자동 재생성 루프)
- **용어 사전**: 보험 전문 용어 자동 해설

## 기술 스택

| 분류 | 기술 | 버전 |
|------|------|------|
| Language | Python | 3.11+ |
| Framework | FastAPI | 0.134.0 |
| Server | Uvicorn / Gunicorn | 0.41.0 / 25.1.0 |
| Database | PostgreSQL 16 (PGVector) | pgvector 0.4.2 |
| ORM | SQLAlchemy (async) | 2.0.47 |
| DB Driver | asyncpg | 0.31.0 |
| Migration | Alembic | 1.18.4 |
| LLM | LangChain + LangGraph | 1.2.10 / 1.0.10 |
| LLM Provider | Ollama (qwen2.5:14b) | langchain-ollama 1.0.1 |
| Embedding | nomic-embed-text (768 dim) | Ollama |
| PDF 파싱 | PyMuPDF + pdfplumber | 1.27.1 / 0.11.9 |
| HTTP Client | httpx | 0.28.1 |
| Validation | Pydantic | 2.12.5 |
| Frontend (MVP) | Streamlit | 1.54.0 |
| Package Manager | uv | - |
| Container | Docker + docker-compose | - |

## 프로젝트 구조

```
insurance-qa-agent/
├── api/                    # FastAPI 백엔드 (핵심)
│   ├── routers/            # API 엔드포인트 정의
│   ├── schemas/            # Pydantic Request/Response 모델
│   ├── services/           # 비즈니스 로직
│   ├── agents/             # LLM Agent (LangGraph 4노드 파이프라인)
│   ├── retrieval/          # 검색 모듈 (임베딩, 하이브리드 검색, PDF 파싱)
│   ├── llm/                # Ollama 클라이언트 + 프롬프트
│   └── db/                 # 데이터베이스 (ORM, Repository)
├── app/                    # Streamlit 프론트엔드
│   └── pages/              # Q&A 페이지
├── core/                   # 공유 모듈
│   ├── constants.py        # Enum, 상수
│   ├── exceptions.py       # 커스텀 예외 계층
│   └── utils/              # 텍스트 처리, 조항 번호 추출
├── data/                   # PDF 약관 파일 (git 미추적)
├── tests/                  # 테스트
├── docs/                   # 설계 문서
└── alembic/                # DB 마이그레이션
```

## 시작하기

### 환경 설정

```bash
# 의존성 설치
uv sync
```

### 환경 변수

`.env.example`을 참고하여 `.env` 파일을 생성합니다:

```bash
cp .env.example .env
```

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/insurance_qa

# Ollama (Local LLM)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:14b
EMBEDDING_MODEL=nomic-embed-text

# App
APP_ENV=development
LOG_LEVEL=DEBUG
```

### 인프라 실행

```bash
docker-compose up -d
```

PostgreSQL 16 (PGVector) + Ollama + API 서버가 실행됩니다.

### Ollama 모델 다운로드

```bash
# LLM 모델
ollama pull qwen2.5:14b

# 임베딩 모델
ollama pull nomic-embed-text
```

### DB 마이그레이션

```bash
alembic upgrade head
```

### API 서버 실행 (로컬 개발)

```bash
uv run uvicorn api.main:app --reload
```

서버가 `http://localhost:8000`에서 실행됩니다.

### Streamlit 앱 실행

```bash
uv run streamlit run app/streamlit_app.py
```

## 테스트

```bash
# 전체 테스트 실행
uv run pytest -v

# 커버리지 포함
uv run pytest --cov=api tests/
```

## 개발

### 코드 품질

```bash
# 린트 검사
uv run ruff check api/ core/

# 린트 자동 수정
uv run ruff check --fix api/ core/
```

### 의존성 관리

```bash
# 의존성 추가
uv add package-name

# 개발 의존성 추가
uv add --group dev package-name

# 동기화
uv sync
```

## Docker

```bash
# 빌드
docker build -t insurance-qa-agent .

# docker-compose (DB + Ollama + API)
docker-compose up -d
```

## RAG 파이프라인

```
사용자 질문
    ↓
Query Processor (질문 분석, 키워드/의도 추출)
    ↓
Retriever (Hybrid Search: Semantic + Keyword + RRF)
    ↓
Answer Generator (검색 결과 기반 답변 생성, 조항 인용)
    ↓
Answer Validator (환각 검증, 신뢰도 평가)
    ↓ 실패 시 Answer Generator로 재시도 (최대 2회)
최종 답변
```

## DB 스키마 (6 테이블)

| 테이블 | 설명 |
|--------|------|
| `insurance_products` | 보험 상품 (상품명, 보험사, 유형) |
| `policy_articles` | 약관 조항 (관/조/항/호 계층, 임베딩 벡터) |
| `article_references` | 조항 간 참조 관계 |
| `article_sub_chunks` | 긴 조항 분할 청크 (임베딩 벡터) |
| `insurance_glossary` | 보험 용어 사전 |
| `qa_logs` | Q&A 로그 (질문, 답변, 신뢰도, 피드백) |

## 라이선스

MIT
