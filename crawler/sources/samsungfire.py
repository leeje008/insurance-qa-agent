"""crawler.sources.samsungfire - 삼성화재 약관 크롤러.

삼성화재 공시실에서 보험약관 PDF를 수집한다.
robots.txt: 전면 허용 (Allow: /)

URL: https://www.samsungfire.com/vh/page/VH.REIF0012.do
"""

from __future__ import annotations

import logging
import re

from crawler.base import BaseCrawler, ProductMeta

logger = logging.getLogger(__name__)

BASE_URL = "https://www.samsungfire.com"
DISCLOSURE_URL = f"{BASE_URL}/vh/page/VH.REIF0012.do"


class SamsungFireCrawler(BaseCrawler):
    """삼성화재 약관 크롤러.

    JS SPA 기반이므로 Playwright를 사용한다.
    카테고리 → 상품 → 판매 기간 → 약관 다운로드 4단계 선택 구조.
    """

    source_name = "samsungfire"
    base_url = DISCLOSURE_URL

    async def fetch_product_list(self) -> list[ProductMeta]:
        """삼성화재 공시실에서 약관 목록 수집."""
        from playwright.async_api import async_playwright

        products: list[ProductMeta] = []

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=self._settings.headless)
            page = await browser.new_page(user_agent=self._settings.user_agent)

            logger.info("삼성화재 공시실 로드: %s", DISCLOSURE_URL)
            await page.goto(DISCLOSURE_URL, wait_until="networkidle", timeout=30000)

            # 보험 카테고리 탭/드롭다운 탐색
            categories = await page.query_selector_all(
                "select#category option, "
                "ul.tab-list li a, "
                "div.category-list a, "
                "select[name*='category'] option, "
                "select[name*='insType'] option"
            )

            if not categories:
                # 전체 상품 목록이 한 페이지에 있는 경우
                products.extend(await self._extract_from_page(page))
                await browser.close()
                return products

            for cat in categories:
                tag = await cat.evaluate("el => el.tagName")
                value = await cat.get_attribute("value") or ""
                text = (await cat.inner_text()).strip()

                if not text or value == "" or text in ("선택", "전체", "--"):
                    continue

                logger.info("카테고리 선택: %s", text)
                await self._delay()

                try:
                    if tag.upper() == "OPTION":
                        parent = await cat.evaluate(
                            "el => el.parentElement.id || el.parentElement.name"
                        )
                        await page.select_option(f"#{parent}" if parent else "select", value=value)
                    else:
                        await cat.click()

                    await page.wait_for_timeout(2000)
                    await page.wait_for_load_state("networkidle", timeout=10000)

                    # 상품 드롭다운 탐색
                    product_options = await page.query_selector_all(
                        "select#product option, "
                        "select[name*='product'] option, "
                        "select[name*='goods'] option"
                    )

                    if product_options:
                        for prod_opt in product_options:
                            prod_value = await prod_opt.get_attribute("value") or ""
                            prod_text = (await prod_opt.inner_text()).strip()
                            if not prod_text or prod_value == "" or prod_text in ("선택", "전체"):
                                continue

                            await self._delay()
                            parent_id = await prod_opt.evaluate(
                                "el => el.parentElement.id || el.parentElement.name"
                            )
                            selector = f"#{parent_id}" if parent_id else "select:nth-of-type(2)"
                            await page.select_option(selector, value=prod_value)
                            await page.wait_for_timeout(2000)

                            products.extend(await self._extract_from_page(
                                page, category=text, product_hint=prod_text,
                            ))
                    else:
                        products.extend(await self._extract_from_page(page, category=text))

                except Exception:
                    logger.exception("카테고리 처리 실패: %s", text)

            await browser.close()

        logger.info("삼성화재 약관 총 %d건 목록 수집 완료", len(products))
        return products

    async def _extract_from_page(
        self, page, category: str = "", product_hint: str = ""
    ) -> list[ProductMeta]:
        """현재 페이지에서 약관 다운로드 링크 추출."""
        products: list[ProductMeta] = []

        # 약관 다운로드 링크 탐색
        links = await page.query_selector_all(
            "a[href*='.pdf'], "
            "a[href*='download'], "
            "a[href*='leaflet'], "
            "a[onclick*='download'], "
            "button.download, "
            "a.btn-download"
        )

        # 테이블 행에서 추출
        rows = await page.query_selector_all("table tbody tr, div.list-item")

        if rows:
            for row in rows:
                try:
                    cells = await row.query_selector_all("td")
                    if len(cells) < 2:
                        continue

                    title = (await cells[0].inner_text()).strip()
                    date_text = ""
                    if len(cells) >= 3:
                        date_text = (await cells[-1].inner_text()).strip()

                    # 다운로드 링크
                    link = await row.query_selector("a[href], a[onclick]")
                    download_url = None
                    if link:
                        href = await link.get_attribute("href") or ""
                        if href and not href.startswith("javascript"):
                            download_url = href if href.startswith("http") else f"{BASE_URL}{href}"

                    product_name = f"{category} - {title}" if category else title
                    version = date_text or "latest"

                    products.append(ProductMeta(
                        insurer="삼성화재",
                        product_name=product_name,
                        version=version,
                        effective_date=date_text or None,
                        download_url=download_url,
                        extra={"category": category, "product_hint": product_hint},
                    ))
                except Exception:
                    logger.debug("행 파싱 실패", exc_info=True)

        elif links:
            for link in links:
                try:
                    href = await link.get_attribute("href") or ""
                    text = (await link.inner_text()).strip()

                    if href.startswith("http"):
                        download_url = href
                    elif href.startswith("/"):
                        download_url = f"{BASE_URL}{href}"
                    else:
                        continue

                    name_match = re.search(r"/([^/]+)\.pdf", href, re.IGNORECASE)
                    fallback = name_match.group(1) if name_match else f"{category} 약관"
                    product_name = text or fallback

                    products.append(ProductMeta(
                        insurer="삼성화재",
                        product_name=product_name,
                        version="latest",
                        download_url=download_url,
                        extra={"category": category},
                    ))
                except Exception:
                    logger.debug("링크 파싱 실패", exc_info=True)

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
