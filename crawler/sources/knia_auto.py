"""crawler.sources.knia_auto - 손해보험협회 자동차보험 약관 크롤러.

손해보험협회 자동차보험 종합포털에서 12개 손보사의 자동차보험 약관을 수집한다.
동적 페이지이므로 Playwright를 사용한다.

URL: https://carinfo.knia.or.kr/lmxsrv/law/icnyLawList.do
리스크: 낮음 (중앙 소스, robots.txt 미설정)
"""

from __future__ import annotations

import logging
import re

from crawler.base import BaseCrawler, ProductMeta

logger = logging.getLogger(__name__)

BASE_URL = "https://carinfo.knia.or.kr"
LIST_URL = f"{BASE_URL}/lmxsrv/law/icnyLawList.do"

# 12개 손보사 목록
INSURERS = [
    "메리츠화재", "한화손해보험", "롯데손해보험", "MG손해보험",
    "흥국화재", "삼성화재", "현대해상", "KB손해보험",
    "DB손해보험", "AXA손해보험", "하나손해보험", "캐롯손해보험",
]


class KNIAAutoCrawler(BaseCrawler):
    """손해보험협회 자동차보험 약관 크롤러.

    Playwright로 동적 페이지를 렌더링하여 보험사별 약관을 수집한다.
    """

    source_name = "knia_auto"
    base_url = LIST_URL

    async def fetch_product_list(self) -> list[ProductMeta]:
        """12개 손보사의 자동차보험 약관 목록 수집."""
        from playwright.async_api import async_playwright

        products: list[ProductMeta] = []

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=self._settings.headless)
            page = await browser.new_page(user_agent=self._settings.user_agent)

            logger.info("약관 목록 페이지 로드: %s", LIST_URL)
            await page.goto(LIST_URL, wait_until="networkidle", timeout=30000)

            # 보험사 카드/링크 요소 탐색
            insurer_links = await page.query_selector_all(
                "div.company-list a, "
                "ul.insurance-list li a, "
                "div.card a, "
                "div.insurer-item a, "
                "a[href*='icnyLaw'], "
                "a[href*='lawDetail'], "
                "div.list-wrap a"
            )

            if not insurer_links:
                # 보험사명 텍스트가 포함된 모든 클릭 가능 요소 탐색
                insurer_links = await page.query_selector_all("a, div[onclick], li[onclick]")

            # 각 보험사 카드를 클릭하여 상세 페이지 진입
            for insurer_name in INSURERS:
                await self._delay()

                try:
                    # 보험사명으로 요소 찾기
                    el = await page.query_selector(f"text={insurer_name}")
                    if not el:
                        # h4, span 등에서 찾기
                        el = await page.query_selector(
                            f"h4:has-text('{insurer_name}'), "
                            f"span:has-text('{insurer_name}'), "
                            f"a:has-text('{insurer_name}')"
                        )

                    if not el:
                        logger.warning("보험사 요소 미발견: %s", insurer_name)
                        continue

                    # 클릭하여 상세 페이지로 이동
                    logger.info("보험사 클릭: %s", insurer_name)

                    # 새 페이지 열림 감지
                    async with page.context.expect_page(timeout=10000) as new_page_info:
                        await el.click()
                    detail_page = await new_page_info.value
                    await detail_page.wait_for_load_state("networkidle", timeout=15000)

                    # 상세 페이지에서 약관 PDF 링크 추출
                    detail_products = await self._extract_products_from_detail(
                        detail_page, insurer_name
                    )
                    products.extend(detail_products)

                    await detail_page.close()

                except Exception:
                    # 새 페이지가 열리지 않는 경우 — 같은 페이지에서 콘텐츠 변경
                    try:
                        await page.wait_for_timeout(2000)
                        detail_products = await self._extract_products_from_detail(
                            page, insurer_name
                        )
                        products.extend(detail_products)
                        # 목록으로 복귀
                        await page.goto(LIST_URL, wait_until="networkidle", timeout=30000)
                    except Exception:
                        logger.exception("보험사 처리 실패: %s", insurer_name)
                        await page.goto(LIST_URL, wait_until="networkidle", timeout=30000)

            await browser.close()

        logger.info("손보협회 자동차보험 약관 총 %d건 목록 수집 완료", len(products))
        return products

    async def _extract_products_from_detail(self, page, insurer_name: str) -> list[ProductMeta]:
        """상세 페이지에서 약관 PDF 링크 추출."""
        products: list[ProductMeta] = []

        # PDF 링크 탐색 — 다양한 패턴
        pdf_links = await page.query_selector_all(
            "a[href$='.pdf'], "
            "a[href*='download'], "
            "a[href*='fileDown'], "
            "a[onclick*='download'], "
            "a[onclick*='fileDown']"
        )

        # 테이블 행에서 약관 정보 추출
        rows = await page.query_selector_all(
            "table tbody tr, div.list-item, li.law-item"
        )

        if rows:
            for row in rows:
                try:
                    # 제목 추출
                    title_el = await row.query_selector("td:first-child, a, span.title")
                    if not title_el:
                        continue
                    title = (await title_el.inner_text()).strip()
                    if not title or len(title) < 2:
                        continue

                    # 날짜 추출
                    date_text = ""
                    date_el = await row.query_selector("td:nth-child(2), span.date")
                    if date_el:
                        date_text = (await date_el.inner_text()).strip()

                    # 다운로드 링크
                    link_el = await row.query_selector("a[href]")
                    href = ""
                    if link_el:
                        href = await link_el.get_attribute("href") or ""

                    download_url = None
                    if href and not href.startswith("javascript"):
                        if href.startswith("/"):
                            download_url = f"{BASE_URL}{href}"
                        elif href.startswith("http"):
                            download_url = href

                    version = date_text or "latest"

                    products.append(ProductMeta(
                        insurer=insurer_name,
                        product_name=title,
                        version=version,
                        effective_date=date_text or None,
                        download_url=download_url,
                    ))

                except Exception:
                    logger.debug("행 파싱 실패", exc_info=True)
                    continue

        elif pdf_links:
            for link in pdf_links:
                try:
                    href = await link.get_attribute("href") or ""
                    text = (await link.inner_text()).strip()

                    if href.startswith("/"):
                        download_url = f"{BASE_URL}{href}"
                    elif href.startswith("http"):
                        download_url = href
                    else:
                        continue

                    # 파일명에서 상품명 추출
                    name_match = re.search(r"/([^/]+)\.pdf", href, re.IGNORECASE)
                    product_name = text or (name_match.group(1) if name_match else "자동차보험약관")

                    products.append(ProductMeta(
                        insurer=insurer_name,
                        product_name=product_name,
                        version="latest",
                        download_url=download_url,
                    ))
                except Exception:
                    logger.debug("PDF 링크 파싱 실패", exc_info=True)
                    continue

        if not products:
            # 전체 페이지에서 PDF 다운로드 가능한 요소 탐색
            all_links = await page.query_selector_all("a[href]")
            for link in all_links:
                href = await link.get_attribute("href") or ""
                if ".pdf" in href.lower():
                    if href.startswith("/"):
                        download_url = f"{BASE_URL}{href}"
                    elif href.startswith("http"):
                        download_url = href
                    else:
                        continue

                    text = (await link.inner_text()).strip() or "자동차보험약관"
                    products.append(ProductMeta(
                        insurer=insurer_name,
                        product_name=text,
                        version="latest",
                        download_url=download_url,
                    ))

        logger.info("%s: %d건 약관 발견", insurer_name, len(products))
        return products

    async def download_pdf(self, product: ProductMeta) -> bytes | None:
        """약관 PDF 다운로드."""
        if not product.download_url:
            logger.warning("다운로드 URL 없음: %s / %s", product.insurer, product.product_name)
            return None

        # Playwright 다운로드 이벤트가 필요한 경우
        if "javascript" in product.download_url or "onclick" in str(product.extra):
            return await self._download_via_playwright(product)

        # 직접 HTTP 다운로드
        logger.info("PDF 다운로드: %s", product.download_url)
        try:
            data = await self._fetch_bytes(product.download_url)
            # PDF 매직 바이트 검증
            if data[:4] == b"%PDF":
                return data
            logger.warning("PDF가 아닌 응답: %s", product.download_url)
            return None
        except Exception:
            logger.exception("PDF 다운로드 실패: %s", product.download_url)
            return None

    async def _download_via_playwright(self, product: ProductMeta) -> bytes | None:
        """Playwright를 통한 PDF 다운로드 (동적 링크용)."""
        from playwright.async_api import async_playwright

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=self._settings.headless)
            page = await browser.new_page(user_agent=self._settings.user_agent)

            try:
                await page.goto(
                    product.download_url, wait_until="networkidle", timeout=30000
                )

                # 다운로드 버튼 클릭
                download_btn = await page.query_selector(
                    "a[href$='.pdf'], a[onclick*='download'], button.download"
                )
                if download_btn:
                    async with page.expect_download(timeout=15000) as download_info:
                        await download_btn.click()
                    download = await download_info.value
                    file_path = await download.path()
                    if file_path:
                        with open(file_path, "rb") as f:
                            return f.read()
            except Exception:
                logger.exception("Playwright 다운로드 실패: %s", product.product_name)
            finally:
                await browser.close()

        return None
