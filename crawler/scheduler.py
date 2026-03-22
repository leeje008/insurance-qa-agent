"""crawler.scheduler - ETL 갱신 주기 관리."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from crawler.config import CrawlerSettings, get_crawler_settings

logger = logging.getLogger(__name__)

SCHEDULE_FILENAME = "schedule_state.json"

# ---------------------------------------------------------------------------
# 갱신 주기 (일 단위)
# ---------------------------------------------------------------------------

DEFAULT_INTERVALS: dict[str, int] = {
    "fss_standard": 90,     # 금감원 표준약관: 분기별
    "knia_auto": 90,        # 손보협회 자동차보험: 분기별
    "samsungfire": 30,      # 개별 보험사: 월 1회
    "hyundai": 30,
    "meritz": 30,
    "dbins": 30,
    "kbinsure": 30,
    "samsunglife": 30,
    "hanwhalife": 30,
    "kyobo": 30,
}


class ScheduleState:
    """크롤러 실행 이력 관리 — 소스별 마지막 실행 시각 추적."""

    def __init__(self, settings: CrawlerSettings | None = None) -> None:
        self._settings = settings or get_crawler_settings()
        self._state_path = self._settings.data_dir / SCHEDULE_FILENAME
        self._state: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self._state_path.exists():
            with open(self._state_path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save(self) -> None:
        self._settings.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self._state_path, "w", encoding="utf-8") as f:
            json.dump(self._state, f, ensure_ascii=False, indent=2, default=str)

    def get_last_run(self, source: str) -> datetime | None:
        """소스의 마지막 실행 시각 반환."""
        ts = self._state.get(source, {}).get("last_run")
        if ts:
            return datetime.fromisoformat(ts)
        return None

    def record_run(self, source: str, count: int = 0) -> None:
        """소스 실행 결과 기록."""
        self._state[source] = {
            "last_run": datetime.now(UTC).isoformat(),
            "last_count": count,
        }
        self._save()
        logger.info("스케줄 기록: %s — %d건 수집", source, count)

    def needs_update(self, source: str) -> bool:
        """소스가 갱신 주기를 초과했는지 확인."""
        interval_days = DEFAULT_INTERVALS.get(source, 30)
        last_run = self.get_last_run(source)

        if last_run is None:
            return True

        elapsed = (datetime.now(UTC) - last_run).days
        needs = elapsed >= interval_days
        logger.debug(
            "스케줄 체크: %s — 경과 %d일 / 주기 %d일 → %s",
            source, elapsed, interval_days, "갱신 필요" if needs else "최신",
        )
        return needs

    def list_due_sources(self) -> list[str]:
        """갱신이 필요한 소스 목록 반환."""
        return [source for source in DEFAULT_INTERVALS if self.needs_update(source)]

    def summary(self) -> dict[str, Any]:
        """전체 스케줄 현황 반환."""
        result: dict[str, Any] = {}
        for source, interval in DEFAULT_INTERVALS.items():
            last_run = self.get_last_run(source)
            result[source] = {
                "interval_days": interval,
                "last_run": last_run.isoformat() if last_run else None,
                "needs_update": self.needs_update(source),
            }
        return result
