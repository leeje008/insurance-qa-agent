"""crawler.config - 크롤러 설정."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

# ---------------------------------------------------------------------------
# 프로젝트 루트 기준 기본 저장 경로
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = _PROJECT_ROOT / "data"


class CrawlerSettings(BaseSettings):
    """크롤러 전역 설정 (환경변수 / .env 오버라이드 가능)."""

    model_config = {"env_prefix": "CRAWLER_", "env_file": ".env", "extra": "ignore"}

    # 저장 경로
    data_dir: Path = Field(default=DEFAULT_DATA_DIR, description="약관 PDF 저장 루트 디렉토리")

    # HTTP 설정
    user_agent: str = Field(
        default="InsuranceQABot/1.0 (+https://github.com/insurance-qa-agent)",
        description="크롤러 User-Agent 헤더",
    )
    request_delay_min: float = Field(default=5.0, description="요청 간 최소 대기 시간(초)")
    request_delay_max: float = Field(default=10.0, description="요청 간 최대 대기 시간(초)")
    request_timeout: float = Field(default=30.0, description="HTTP 요청 타임아웃(초)")
    max_retries: int = Field(default=3, description="실패 시 최대 재시도 횟수")

    # Playwright 설정
    headless: bool = Field(default=True, description="Playwright 헤드리스 모드")
    browser_type: str = Field(default="chromium", description="Playwright 브라우저 유형")


def get_crawler_settings() -> CrawlerSettings:
    """크롤러 설정 싱글톤 반환."""
    return CrawlerSettings()
