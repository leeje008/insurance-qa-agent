"""crawler.sources.hanwhalife - 한화생명 약관 크롤러.

한화생명 공시실에서 생명보험 약관 PDF를 수집한다.
.do 엔드포인트 기반.

URL: https://www.hanwhalife.com
"""

from __future__ import annotations

import logging

from crawler.base import BaseCrawler, ProductMeta

logger = logging.getLogger(__name__)

BASE_URL = "https://www.hanwhalife.com"


class HanwhaLifeCrawler(BaseCrawler):
    """한화생명 약관 크롤러."""

    source_name = "hanwhalife"
    base_url = BASE_URL

    async def fetch_product_list(self) -> list[ProductMeta]:
        """한화생명 공시실에서 약관 목록 수집."""
        from playwright.async_api import async_playwright

        products: list[ProductMeta] = []

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=self._settings.headless)
            page = await browser.new_page(user_agent=self._settings.user_agent)

            disclosure_urls = [
                f"{BASE_URL}/main/disclosure/clause.do",
                f"{BASE_URL}/disclosure/clause.do",
                f"{BASE_URL}/company/disclosure/main.do",
            ]

            loaded = False
            for url in disclosure_urls:
                try:
                    resp = await page.goto(
                        url, wait_until="networkidle", timeout=15000,
                    )
                    if resp and resp.status == 200:
                        loaded = True
                        break
                except Exception:
                    continue

            if not loaded:
                await page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
                link = await page.query_selector(
                    "a[href*='disclosure'], a:has-text('공시실')"
                )
                if link:
                    await link.click()
                    await page.wait_for_load_state("networkidle", timeout=15000)
                    loaded = True

            if not loaded:
                logger.error("한화생명 공시실 접근 실패")
                await browser.close()
                return products

            products.extend(await self._extract_products(page))
            await browser.close()

        logger.info("한화생명 약관 총 %d건 수집 완료", len(products))
        return products

    async def _extract_products(self, page) -> list[ProductMeta]:
        """페이지에서 약관 목록 추출."""
        products: list[ProductMeta] = []

        selects = await page.query_selector_all("select")
        if selects:
            first_select = selects[0]
            options = await first_select.query_selector_all("option")
            for opt in options:
                value = await opt.get_attribute("value") or ""
                text = (await opt.inner_text()).strip()
                if not value or text in ("선택", "전체", "--", ""):
                    continue
                await self._delay()
                sel_id = (
                    await first_select.get_attribute("id")
                    or await first_select.get_attribute("name")
                )
                try:
                    sel = f"#{sel_id}" if sel_id else "select"
                    await page.select_option(sel, value=value)
                    await page.wait_for_timeout(2000)
                except Exception:
                    continue
                products.extend(
                    await self._extract_links(page, category=text)
                )
        else:
            products.extend(await self._extract_links(page))

        return products

    async def _extract_links(
        self, page, category: str = "",
    ) -> list[ProductMeta]:
        """PDF 링크 추출."""
        products: list[ProductMeta] = []
        rows = await page.query_selector_all("table tbody tr, div.list-item")

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
                if href and not href.startswith("javascript"):
                    download_url = (
                        href if href.startswith("http") else f"{BASE_URL}{href}"
                    )

                name = f"{category} - {title}" if category else title
                products.append(ProductMeta(
                    insurer="한화생명",
                    product_name=name,
                    version="latest",
                    download_url=download_url,
                ))
            except Exception:
                logger.debug("행 파싱 실패", exc_info=True)

        return products

    async def download_pdf(self, product: ProductMeta) -> bytes | None:
        """약관 PDF 다운로드."""
        if not product.download_url:
            return None
        try:
            data = await self._fetch_bytes(product.download_url)
            if data[:4] == b"%PDF":
                return data
            return None
        except Exception:
            logger.exception(
                "PDF 다운로드 실패: %s / %s",
                product.insurer, product.product_name,
            )
            return None
