"""crawler.run - 크롤러 CLI 엔트리포인트.

Usage:
    uv run python -m crawler.run --source all          # 갱신 필요한 전체 소스 실행
    uv run python -m crawler.run --source fss_standard  # 금감원 표준약관만
    uv run python -m crawler.run --source knia_auto     # 손보협회 자동차보험만
    uv run python -m crawler.run --source all --dry-run # 상품 목록만 조회
    uv run python -m crawler.run --status               # 스케줄 현황 조회
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from core.logging_config import setup_logging
from crawler.base import BaseCrawler
from crawler.scheduler import ScheduleState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 소스 레지스트리
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[BaseCrawler]] = {}


def _register_sources() -> None:
    """사용 가능한 크롤러 소스를 등록한다.

    임포트 에러가 발생하면 해당 소스를 건너뛴다.
    """
    try:
        from crawler.sources.fss_standard import FSSStandardCrawler
        _REGISTRY["fss_standard"] = FSSStandardCrawler
    except ImportError:
        pass

    try:
        from crawler.sources.knia_auto import KNIAAutoCrawler
        _REGISTRY["knia_auto"] = KNIAAutoCrawler
    except ImportError:
        pass

    try:
        from crawler.sources.samsungfire import SamsungFireCrawler
        _REGISTRY["samsungfire"] = SamsungFireCrawler
    except ImportError:
        pass

    try:
        from crawler.sources.hyundai import HyundaiCrawler
        _REGISTRY["hyundai"] = HyundaiCrawler
    except ImportError:
        pass

    try:
        from crawler.sources.meritz import MeritzCrawler
        _REGISTRY["meritz"] = MeritzCrawler
    except ImportError:
        pass

    try:
        from crawler.sources.dbins import DBInsCrawler
        _REGISTRY["dbins"] = DBInsCrawler
    except ImportError:
        pass

    try:
        from crawler.sources.kbinsure import KBInsureCrawler
        _REGISTRY["kbinsure"] = KBInsureCrawler
    except ImportError:
        pass

    try:
        from crawler.sources.samsunglife import SamsungLifeCrawler
        _REGISTRY["samsunglife"] = SamsungLifeCrawler
    except ImportError:
        pass

    try:
        from crawler.sources.hanwhalife import HanwhaLifeCrawler
        _REGISTRY["hanwhalife"] = HanwhaLifeCrawler
    except ImportError:
        pass

    try:
        from crawler.sources.kyobo import KyoboCrawler
        _REGISTRY["kyobo"] = KyoboCrawler
    except ImportError:
        pass


def _get_registry() -> dict[str, type[BaseCrawler]]:
    if not _REGISTRY:
        _register_sources()
    return _REGISTRY


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

async def run_source(source_name: str, *, dry_run: bool = False) -> int:
    """단일 소스 크롤러 실행. 저장 건수 반환."""
    registry = _get_registry()

    if source_name not in registry:
        logger.error("알 수 없는 소스: %s (사용 가능: %s)", source_name, list(registry.keys()))
        return 0

    crawler_cls = registry[source_name]
    crawler = crawler_cls()

    try:
        results = await crawler.run(dry_run=dry_run)
        count = len(results)

        if not dry_run:
            schedule = ScheduleState()
            schedule.record_run(source_name, count)

        return count
    finally:
        await crawler.close()


async def run_all(*, dry_run: bool = False, force: bool = False) -> dict[str, int]:
    """갱신 필요한 모든 소스 실행.

    Args:
        dry_run: 상품 목록만 조회.
        force: 스케줄 무시하고 전체 실행.

    Returns:
        소스별 저장 건수 dict.
    """
    registry = _get_registry()
    schedule = ScheduleState()

    if force:
        sources = list(registry.keys())
    else:
        due = schedule.list_due_sources()
        sources = [s for s in due if s in registry]

    if not sources:
        logger.info("갱신 필요한 소스 없음")
        return {}

    logger.info("실행 대상 소스: %s", sources)
    results: dict[str, int] = {}

    for source_name in sources:
        count = await run_source(source_name, dry_run=dry_run)
        results[source_name] = count

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI 메인 함수."""
    setup_logging()

    parser = argparse.ArgumentParser(description="보험약관 크롤러")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="수집할 소스 (all | fss_standard | knia_auto | samsungfire | ...)",
    )
    parser.add_argument("--dry-run", action="store_true", help="상품 목록만 조회 (다운로드 안 함)")
    parser.add_argument("--force", action="store_true", help="스케줄 무시하고 강제 실행")
    parser.add_argument("--status", action="store_true", help="스케줄 현황 조회")
    parser.add_argument("--list-sources", action="store_true", help="사용 가능한 소스 목록")

    args = parser.parse_args()

    if args.list_sources:
        registry = _get_registry()
        print("사용 가능한 소스:")
        for name in sorted(registry.keys()):
            print(f"  - {name}")
        return

    if args.status:
        schedule = ScheduleState()
        print(json.dumps(schedule.summary(), ensure_ascii=False, indent=2, default=str))
        return

    if args.source is None:
        parser.print_help()
        sys.exit(1)

    if args.source == "all":
        results = asyncio.run(run_all(dry_run=args.dry_run, force=args.force))
    else:
        count = asyncio.run(run_source(args.source, dry_run=args.dry_run))
        results = {args.source: count}

    print(f"\n수집 결과: {json.dumps(results, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
