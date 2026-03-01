"""core.exceptions - 프로젝트 전역 예외 정의."""


# ---------------------------------------------------------------------------
# 최상위 예외
# ---------------------------------------------------------------------------

class InsuranceQAError(Exception):
    """Insurance QA Agent 최상위 예외.

    모든 도메인 예외가 이 클래스를 상속합니다.
    """

    def __init__(self, message: str = "", *, detail: str | None = None) -> None:
        self.message = message
        self.detail = detail
        super().__init__(message)


# ---------------------------------------------------------------------------
# PDF 파싱 관련
# ---------------------------------------------------------------------------

class PDFParsingError(InsuranceQAError):
    """PDF 약관 파싱 중 오류."""


# ---------------------------------------------------------------------------
# 임베딩 관련
# ---------------------------------------------------------------------------

class EmbeddingError(InsuranceQAError):
    """임베딩 생성/조회 중 오류."""


# ---------------------------------------------------------------------------
# 검색 관련
# ---------------------------------------------------------------------------

class RetrievalError(InsuranceQAError):
    """약관 검색 중 오류."""


class PolicyNotFoundError(InsuranceQAError):
    """요청한 약관/상품을 찾을 수 없음."""


# ---------------------------------------------------------------------------
# LLM 관련
# ---------------------------------------------------------------------------

class LLMResponseError(InsuranceQAError):
    """LLM 응답 오류 (Ollama 호출 실패, 파싱 불가 등)."""

    def __init__(
        self,
        message: str = "",
        *,
        service: str | None = None,
        status_code: int | None = None,
        detail: str | None = None,
    ) -> None:
        self.service = service
        self.status_code = status_code
        super().__init__(message, detail=detail)


# ---------------------------------------------------------------------------
# 검증 관련
# ---------------------------------------------------------------------------

class AnswerValidationError(InsuranceQAError):
    """답변 검증 실패 (근거 부족, 환각 감지 등)."""

    def __init__(
        self,
        message: str = "",
        *,
        issues: list[str] | None = None,
        detail: str | None = None,
    ) -> None:
        self.issues = issues or []
        super().__init__(message, detail=detail)


# ---------------------------------------------------------------------------
# 데이터베이스 관련
# ---------------------------------------------------------------------------

class DatabaseError(InsuranceQAError):
    """데이터베이스 접근 중 오류."""
