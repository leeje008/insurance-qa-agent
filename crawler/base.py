"""crawler.base - 크롤러 추상 베이스 클래스."""

from __future__ import annotations

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from crawler.config import CrawlerSettings, get_crawler_settings
from crawler.storage import PolicyStorage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 상품 메타데이터
# ---------------------------------------------------------------------------

@dataclass
class ProductMeta:
    """수집 대상 보험 상품 메타데이터."""

    insurer: str
    product_name: str
    version: str
    effective_date: str | None = None
    download_url: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BaseCrawler
# ---------------------------------------------------------------------------

class BaseCrawler(ABC):
    """보험약관 크롤러 추상 베이스 클래스.

    서브클래스는 source_name, base_url, fetch_product_list, download_pdf를 구현해야 한다.
    """

    source_name: str = ""
    base_url: str = ""

    def __init__(self, settings: CrawlerSettings | None = None) -> None:
        self._settings = settings or get_crawler_settings()
        self._storage = PolicyStorage(self._settings)
        self._client: httpx.AsyncClient | None = None
        self._robots_parser: RobotFileParser | None = None

    # ------------------------------------------------------------------
    # HTTP 클라이언트
    # ------------------------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        """공유 httpx 비동기 클라이언트 반환."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers={"User-Agent": self._settings.user_agent},
                timeout=self._settings.request_timeout,
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        """리소스 정리."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Rate Limiting
    # ------------------------------------------------------------------

    async def _delay(self) -> None:
        """설정된 범위 내에서 랜덤 대기."""
        delay = random.uniform(
            self._settings.request_delay_min,
            self._settings.request_delay_max,
        )
        logger.debug("Rate limit 대기: %.1f초", delay)
        await asyncio.sleep(delay)

    # ------------------------------------------------------------------
    # robots.txt 검증
    # ------------------------------------------------------------------

    async def check_robots_txt(self, url: str) -> bool:
        """주어진 URL이 robots.txt에 의해 허용되는지 확인.

        robots.txt를 가져올 수 없으면 기본 허용(True)으로 처리한다.
        """
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

        if self._robots_parser is None:
            self._robots_parser = RobotFileParser()
            try:
                client = await self._get_client()
                resp = await client.get(robots_url)
                if resp.status_code == 200:
                    self._robots_parser.parse(resp.text.splitlines())
                    logger.info("robots.txt 로드 완료: %s", robots_url)
                else:
                    logger.info(
                        "robots.txt 없음 (status=%d): %s — 기본 허용",
                        resp.status_code, robots_url,
                    )
                    return True
            except httpx.HTTPError:
                logger.warning("robots.txt 접근 실패: %s — 기본 허용", robots_url)
                return True

        allowed = self._robots_parser.can_fetch(self._settings.user_agent, url)
        if not allowed:
            logger.warning("robots.txt 차단: %s", url)
        return allowed

    # ------------------------------------------------------------------
    # HTTP 헬퍼
    # ------------------------------------------------------------------

    async def _fetch(self, url: str, *, method: str = "GET", **kwargs: Any) -> httpx.Response:
        """HTTP 요청 + rate limiting + 재시도.

        Raises:
            httpx.HTTPStatusError: 최대 재시도 후에도 실패 시.
        """
        client = await self._get_client()

        for attempt in range(1, self._settings.max_retries + 1):
            try:
                await self._delay()
                resp = await client.request(method, url, **kwargs)
                resp.raise_for_status()
                logger.info(
                    "[%d/%d] %s %s → %d",
                    attempt, self._settings.max_retries, method, url, resp.status_code,
                )
                return resp
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                logger.warning(
                    "[%d/%d] %s %s 실패: %s",
                    attempt, self._settings.max_retries, method, url, exc,
                )
                if attempt == self._settings.max_retries:
                    raise

        raise RuntimeError("Unreachable")  # pragma: no cover

    async def _fetch_bytes(self, url: str) -> bytes:
        """URL에서 바이트 데이터 가져오기."""
        resp = await self._fetch(url)
        return resp.content

    # ------------------------------------------------------------------
    # 추상 메서드 (서브클래스 구현)
    # ------------------------------------------------------------------

    @abstractmethod
    async def fetch_product_list(self) -> list[ProductMeta]:
        """수집 대상 상품 목록 조회.

        Returns:
            ProductMeta 리스트.
        """

    @abstractmethod
    async def download_pdf(self, product: ProductMeta) -> bytes | None:
        """상품의 약관 PDF 다운로드.

        Returns:
            PDF 바이트 데이터, 또는 다운로드 불가 시 None.
        """

    # ------------------------------------------------------------------
    # 수집 실행
    # ------------------------------------------------------------------

    async def run(self, *, dry_run: bool = False) -> list[dict[str, Any]]:
        """전체 수집 프로세스 실행.

        Args:
            dry_run: True이면 상품 목록만 조회하고 다운로드하지 않음.

        Returns:
            저장된 manifest 항목 리스트.
        """
        logger.info("=== 크롤러 시작: %s ===", self.source_name)

        # robots.txt 사전 검증
        if self.base_url:
            allowed = await self.check_robots_txt(self.base_url)
            if not allowed:
                logger.error("robots.txt에 의해 차단됨: %s — 수집 중단", self.base_url)
                return []

        products = await self.fetch_product_list()
        logger.info("수집 대상 상품: %d건", len(products))

        if dry_run:
            for p in products:
                logger.info("[DRY-RUN] %s / %s / %s", p.insurer, p.product_name, p.version)
            return []

        saved: list[dict[str, Any]] = []
        for i, product in enumerate(products, 1):
            logger.info("[%d/%d] %s / %s", i, len(products), product.insurer, product.product_name)

            # 개별 URL에 대한 robots.txt 검증
            if product.download_url:
                allowed = await self.check_robots_txt(product.download_url)
                if not allowed:
                    logger.warning("robots.txt 차단으로 건너뜀: %s", product.download_url)
                    continue

            try:
                pdf_data = await self.download_pdf(product)
            except Exception:
                logger.exception(
                    "PDF 다운로드 실패: %s / %s", product.insurer, product.product_name,
                )
                continue

            if pdf_data is None:
                logger.warning("PDF 데이터 없음: %s / %s", product.insurer, product.product_name)
                continue

            entry = self._storage.save_pdf(
                pdf_data,
                source=self.source_name,
                insurer=product.insurer,
                product_name=product.product_name,
                version=product.version,
                effective_date=product.effective_date,
                source_url=product.download_url,
                extra_metadata=product.extra if product.extra else None,
            )
            if entry:
                saved.append(entry)

        logger.info("=== 크롤러 완료: %s — 신규 저장 %d건 ===", self.source_name, len(saved))
        await self.close()
        return saved

    # ------------------------------------------------------------------
    # 변경 감지
    # ------------------------------------------------------------------

    async def detect_updates(self) -> list[ProductMeta]:
        """기존 manifest 대비 신규/변경 상품 감지.

        Returns:
            신규 또는 변경된 ProductMeta 리스트.
        """
        products = await self.fetch_product_list()
        new_products: list[ProductMeta] = []

        for product in products:
            existing = self._storage.find_by_source(
                self.source_name, product.insurer, product.product_name, product.version
            )
            if existing is None:
                new_products.append(product)

        logger.info(
            "변경 감지 결과: 전체 %d건 중 신규 %d건",
            len(products), len(new_products),
        )
        return new_products
