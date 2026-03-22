"""crawler.sources.dbins - DB손해보험 약관 크롤러.

DB손해보험 약관 다운로드 페이지에서 보험약관 PDF를 수집한다.
robots.txt: 매우 제한적 — 사전 협의 권장.

URL: https://www.idbins.com/FWMAIV1534.do
"""

from __future__ import annotations

import logging
import re

from crawler.base import BaseCrawler, ProductMeta

logger = logging.getLogger(__name__)

BASE_URL = "https://www.idbins.com"
DISCLOSURE_URL = f"{BASE_URL}/FWMAIV1534.do"


class DBInsCrawler(BaseCrawler):
    """DB손해보험 약관 크롤러.

    robots.txt가 매우 제한적이므로 check_robots_txt() 결과에 따라
    자동으로 크롤링을 중단할 수 있다.
    """

    source_name = "dbins"
    base_url = DISCLOSURE_URL

    async def fetch_product_list(self) -> list[ProductMeta]:
        """DB손보 약관 목록 수집."""
        from playwright.async_api import async_playwright

        products: list[ProductMeta] = []

        # robots.txt 사전 검증
        allowed = await self.check_robots_txt(DISCLOSURE_URL)
        if not allowed:
            logger.warning("DB손보 robots.txt 차단 — 크롤링 중단")
            return products

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=self._settings.headless)
            page = await browser.new_page(user_agent=self._settings.user_agent)

            logger.info("DB손보 약관 페이지 로드: %s", DISCLOSURE_URL)
            await page.goto(DISCLOSURE_URL, wait_until="networkidle", timeout=30000)

            # 카테고리/상품 선택 드롭다운 탐색
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

            await browser.close()

        logger.info("DB손보 약관 총 %d건 목록 수집 완료", len(products))
        return products

    async def _extract_links(
        self, page, category: str = "",
    ) -> list[ProductMeta]:
        """페이지에서 약관 PDF 링크 추출."""
        products: list[ProductMeta] = []

        rows = await page.query_selector_all("table tbody tr")
        for row in rows:
            try:
                cells = await row.query_selector_all("td")
                if len(cells) < 2:
                    continue
                title = (await cells[0].inner_text()).strip()
                if not title or len(title) < 2:
                    continue

                date_text = ""
                if len(cells) >= 3:
                    date_text = (await cells[-1].inner_text()).strip()

                link = await row.query_selector("a[href]")
                download_url = None
                if link:
                    href = await link.get_attribute("href") or ""
                    if href and not href.startswith("javascript"):
                        download_url = (
                            href if href.startswith("http")
                            else f"{BASE_URL}{href}"
                        )

                name = f"{category} - {title}" if category else title
                products.append(ProductMeta(
                    insurer="DB손해보험",
                    product_name=name,
                    version=date_text or "latest",
                    effective_date=date_text or None,
                    download_url=download_url,
                    extra={"category": category},
                ))
            except Exception:
                logger.debug("행 파싱 실패", exc_info=True)

        if not products:
            pdf_links = await page.query_selector_all("a[href*='.pdf']")
            for link in pdf_links:
                try:
                    href = await link.get_attribute("href") or ""
                    text = (await link.inner_text()).strip()
                    download_url = (
                        href if href.startswith("http")
                        else f"{BASE_URL}{href}"
                    )
                    match = re.search(r"/([^/]+)\.pdf", href, re.IGNORECASE)
                    name = text or (match.group(1) if match else "약관")
                    products.append(ProductMeta(
                        insurer="DB손해보험",
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
            logger.exception(
                "PDF 다운로드 실패: %s / %s",
                product.insurer, product.product_name,
            )
            return None
