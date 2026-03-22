"""crawler.sources.hyundai - 현대해상 약관 크롤러.

현대해상 공시실에서 보험약관 PDF를 수집한다.
robots.txt: 부분 허용 (공시실 접근 가능)

URL: https://www.hi.co.kr
"""

from __future__ import annotations

import logging
import re

from crawler.base import BaseCrawler, ProductMeta

logger = logging.getLogger(__name__)

BASE_URL = "https://www.hi.co.kr"


class HyundaiCrawler(BaseCrawler):
    """현대해상 약관 크롤러."""

    source_name = "hyundai"
    base_url = BASE_URL

    async def fetch_product_list(self) -> list[ProductMeta]:
        """현대해상 공시실에서 약관 목록 수집."""
        from playwright.async_api import async_playwright

        products: list[ProductMeta] = []

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=self._settings.headless)
            page = await browser.new_page(user_agent=self._settings.user_agent)

            # 공시실 페이지 탐색 — 정확한 URL은 런타임에 확인
            disclosure_urls = [
                f"{BASE_URL}/service/disclosure/clause.do",
                f"{BASE_URL}/direct/service/disclosure/clause.do",
                f"{BASE_URL}/khd/disclosure.do",
            ]

            loaded = False
            for url in disclosure_urls:
                try:
                    logger.info("현대해상 공시실 시도: %s", url)
                    resp = await page.goto(url, wait_until="networkidle", timeout=15000)
                    if resp and resp.status == 200:
                        loaded = True
                        break
                except Exception:
                    continue

            if not loaded:
                # 메인 페이지에서 공시실 링크 탐색
                await page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
                disclosure_link = await page.query_selector(
                    "a[href*='disclosure'], a[href*='clause'], a:has-text('공시실')"
                )
                if disclosure_link:
                    await disclosure_link.click()
                    await page.wait_for_load_state("networkidle", timeout=15000)
                    loaded = True

            if not loaded:
                logger.error("현대해상 공시실 접근 실패")
                await browser.close()
                return products

            # 카테고리/상품 선택 및 약관 목록 추출
            products.extend(await self._extract_products(page))
            await browser.close()

        logger.info("현대해상 약관 총 %d건 목록 수집 완료", len(products))
        return products

    async def _extract_products(self, page) -> list[ProductMeta]:
        """페이지에서 약관 목록 추출."""
        products: list[ProductMeta] = []

        # 카테고리 탐색
        selects = await page.query_selector_all(
            "select[name*='category'], select[name*='insType'], "
            "select[name*='prodType'], select#category"
        )

        if selects:
            select = selects[0]
            options = await select.query_selector_all("option")

            for opt in options:
                value = await opt.get_attribute("value") or ""
                text = (await opt.inner_text()).strip()
                if not value or text in ("선택", "전체", "--"):
                    continue

                await self._delay()
                selector_id = (
                    await select.get_attribute("id")
                    or await select.get_attribute("name")
                )
                sel = f"#{selector_id}" if selector_id else "select"
                await page.select_option(sel, value=value)
                await page.wait_for_timeout(2000)

                products.extend(await self._extract_from_table(page, category=text))
        else:
            products.extend(await self._extract_from_table(page))

        return products

    async def _extract_from_table(self, page, category: str = "") -> list[ProductMeta]:
        """테이블/리스트에서 약관 항목 추출."""
        products: list[ProductMeta] = []

        rows = await page.query_selector_all("table tbody tr, div.list-item, li.item")

        for row in rows:
            try:
                link = await row.query_selector("a[href]")
                if not link:
                    continue

                title = (await link.inner_text()).strip()
                href = await link.get_attribute("href") or ""
                if not title or len(title) < 2:
                    continue

                download_url = None
                if ".pdf" in href.lower():
                    download_url = href if href.startswith("http") else f"{BASE_URL}{href}"
                elif href.startswith("/"):
                    download_url = f"{BASE_URL}{href}"

                # 날짜 추출
                date_text = ""
                date_el = await row.query_selector("td:nth-child(3), span.date")
                if date_el:
                    date_text = (await date_el.inner_text()).strip()

                product_name = f"{category} - {title}" if category else title
                products.append(ProductMeta(
                    insurer="현대해상",
                    product_name=product_name,
                    version=date_text or "latest",
                    effective_date=date_text or None,
                    download_url=download_url,
                    extra={"category": category},
                ))

            except Exception:
                logger.debug("행 파싱 실패", exc_info=True)

        # PDF 직접 링크 탐색
        if not products:
            pdf_links = await page.query_selector_all("a[href*='.pdf']")
            for link in pdf_links:
                try:
                    href = await link.get_attribute("href") or ""
                    text = (await link.inner_text()).strip()
                    download_url = href if href.startswith("http") else f"{BASE_URL}{href}"
                    name_match = re.search(r"/([^/]+)\.pdf", href, re.IGNORECASE)
                    name = text or (name_match.group(1) if name_match else "약관")

                    products.append(ProductMeta(
                        insurer="현대해상",
                        product_name=name,
                        version="latest",
                        download_url=download_url,
                    ))
                except Exception:
                    pass

        return products

    async def download_pdf(self, product: ProductMeta) -> bytes | None:
        """약관 PDF 다운로드."""
        if not product.download_url:
            return None

        try:
            data = await self._fetch_bytes(product.download_url)
            if data[:4] == b"%PDF":
                return data
            logger.warning("PDF가 아닌 응답: %s", product.download_url)
            return None
        except Exception:
            logger.exception("PDF 다운로드 실패: %s / %s", product.insurer, product.product_name)
            return None
