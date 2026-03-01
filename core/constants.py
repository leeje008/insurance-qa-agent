"""core.constants - 프로젝트 전역 상수 정의."""

from enum import StrEnum

# ---------------------------------------------------------------------------
# 보험 유형
# ---------------------------------------------------------------------------

class InsuranceType(StrEnum):
    """보험 상품 유형."""

    LIFE = "life"
    HEALTH = "health"
    AUTO = "auto"
    FIRE = "fire"
    TRAVEL = "travel"
    LIABILITY = "liability"
    OTHER = "other"


# ---------------------------------------------------------------------------
# 약관 조항 계층 구조
# ---------------------------------------------------------------------------

class ArticleLevel(StrEnum):
    """약관 조항 계층 수준 (관 > 조 > 항 > 호)."""

    PART = "part"          # 관
    ARTICLE = "article"    # 조
    PARAGRAPH = "paragraph"  # 항
    ITEM = "item"          # 호


# ---------------------------------------------------------------------------
# Q&A 상태
# ---------------------------------------------------------------------------

class QAStatus(StrEnum):
    """질의응답 처리 상태."""

    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_CLARIFICATION = "needs_clarification"


# ---------------------------------------------------------------------------
# LangGraph 파이프라인 노드
# ---------------------------------------------------------------------------

class PipelineNode(StrEnum):
    """RAG 파이프라인 노드 이름."""

    QUERY_PROCESSOR = "query_processor"
    RETRIEVER = "retriever"
    ANSWER_GENERATOR = "answer_generator"
    ANSWER_VALIDATOR = "answer_validator"


# ---------------------------------------------------------------------------
# 검색 설정
# ---------------------------------------------------------------------------

# 시맨틱 검색 상위 K개 결과
TOP_K_RESULTS: int = 5

# 유사도 임계값 (이하 결과 제외)
SIMILARITY_THRESHOLD: float = 0.7

# RRF(Reciprocal Rank Fusion) 파라미터
RRF_K: int = 60

# 키워드 검색 가중치 vs 시맨틱 검색 가중치
KEYWORD_WEIGHT: float = 0.3
SEMANTIC_WEIGHT: float = 0.7


# ---------------------------------------------------------------------------
# 임베딩 설정
# ---------------------------------------------------------------------------

EMBEDDING_MODEL: str = "nomic-embed-text"
EMBEDDING_DIM: int = 768


# ---------------------------------------------------------------------------
# LLM 설정
# ---------------------------------------------------------------------------

DEFAULT_LLM_MODEL: str = "qwen2.5:14b"
FALLBACK_LLM_MODEL: str = "qwen2.5:7b"
LLM_TEMPERATURE: float = 0.1
LLM_MAX_TOKENS: int = 2048


# ---------------------------------------------------------------------------
# 청킹 설정
# ---------------------------------------------------------------------------

# 서브 청크 최대 토큰 수
SUB_CHUNK_MAX_TOKENS: int = 512

# 청크 간 오버랩 토큰 수
CHUNK_OVERLAP_TOKENS: int = 50


# ---------------------------------------------------------------------------
# 답변 검증
# ---------------------------------------------------------------------------

# 최대 재시도 횟수 (answer_validator → answer_generator 루프)
MAX_VALIDATION_RETRIES: int = 2

# 최소 신뢰도 점수
MIN_CONFIDENCE_SCORE: float = 0.6
