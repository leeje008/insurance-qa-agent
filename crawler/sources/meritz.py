"""crawler.sources.meritz - 메리츠화재 약관 크롤러.

메리츠화재 공시실에서 보험약관 PDF를 수집한다.
robots.txt: 미제공 (기본 허용)

URL: https://www.meritzfire.com
PDF URL 패턴: cmdown.meritzfire.com/manager/cm/document/[file].pdf
"""

from __future__ import annotations

import logging
import re

from crawler.base import BaseCrawler, ProductMeta

logger = logging.getLogger(__name__)

BASE_URL = "https://www.meritzfire.com"


class MeritzCrawler(BaseCrawler):
    """메리츠화재 약관 크롤러."""

    source_name = "meritz"
    base_url = BASE_URL

    async def fetch_product_list(self) -> list[ProductMeta]:
        """메리츠화재 공시실에서 약관 목록 수집."""
        from playwright.async_api import async_playwright

        products: list[ProductMeta] = []

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=self._settings.headless)
            page = await browser.new_page(user_agent=self._settings.user_agent)

            # 공시실 페이지 탐색
            disclosure_urls = [
                f"{BASE_URL}/disclosure/clause",
                f"{BASE_URL}/static/disclosure/clause.html",
                f"{BASE_URL}/company/disclosure/clause.do",
            ]

            loaded = False
            for url in disclosure_urls:
                try:
                    logger.info("메리츠화재 공시실 시도: %s", url)
                    resp = await page.goto(url, wait_until="networkidle", timeout=15000)
                    if resp and resp.status == 200:
                        loaded = True
                        break
                except Exception:
                    continue

            if not loaded:
                await page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
                link = await page.query_selector(
                    "a[href*='disclosure'], a[href*='clause'], a:has-text('공시실')"
                )
                if link:
                    await link.click()
                    await page.wait_for_load_state("networkidle", timeout=15000)
                    loaded = True

            if not loaded:
                logger.error("메리츠화재 공시실 접근 실패")
                await browser.close()
                return products

            # 약관 목록 추출
            products.extend(await self._extract_products(page))
            await browser.close()

        logger.info("메리츠화재 약관 총 %d건 목록 수집 완료", len(products))
        return products

    async def _extract_products(self, page) -> list[ProductMeta]:
        """페이지에서 약관 항목 추출."""
        products: list[ProductMeta] = []

        # 카테고리 드롭다운 탐색
        selects = await page.query_selector_all("select")
        if selects:
            for select in selects:
                options = await select.query_selector_all("option")
                for opt in options:
                    value = await opt.get_attribute("value") or ""
                    text = (await opt.inner_text()).strip()
                    if not value or text in ("선택", "전체", "--"):
                        continue

                    await self._delay()
                    sel_id = await select.get_attribute("id") or await select.get_attribute("name")
                    try:
                        await page.select_option(f"#{sel_id}" if sel_id else "select", value=value)
                        await page.wait_for_timeout(2000)
                    except Exception:
                        continue

                    products.extend(await self._extract_links(page, category=text))
                break  # 첫 번째 select만
        else:
            products.extend(await self._extract_links(page))

        return products

    async def _extract_links(self, page, category: str = "") -> list[ProductMeta]:
        """현재 페이지에서 PDF 링크 추출."""
        products: list[ProductMeta] = []

        # 테이블 행 탐색
        rows = await page.query_selector_all("table tbody tr")
        for row in rows:
            try:
                cells = await row.query_selector_all("td")
                if len(cells) < 2:
                    continue

                title = (await cells[0].inner_text()).strip()
                date_text = ""
                if len(cells) >= 3:
                    date_text = (await cells[-1].inner_text()).strip()

                link = await row.query_selector("a[href]")
                download_url = None
                if link:
                    href = await link.get_attribute("href") or ""
                    if href and not href.startswith("javascript"):
                        if href.startswith("http"):
                            download_url = href
                        elif href.startswith("/"):
                            download_url = f"{BASE_URL}{href}"
                        elif "cmdown" in href:
                            download_url = f"https://{href}"

                product_name = f"{category} - {title}" if category else title
                products.append(ProductMeta(
                    insurer="메리츠화재",
                    product_name=product_name,
                    version=date_text or "latest",
                    effective_date=date_text or None,
                    download_url=download_url,
                    extra={"category": category},
                ))
            except Exception:
                logger.debug("행 파싱 실패", exc_info=True)

        # PDF 직접 링크
        if not products:
            pdf_links = await page.query_selector_all(
                "a[href*='.pdf'], a[href*='cmdown']"
            )
            for link in pdf_links:
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
                    name = text or (name_match.group(1) if name_match else "약관")

                    products.append(ProductMeta(
                        insurer="메리츠화재",
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
