"""crawler.sources.fss_standard - 금융감독원 표준약관 크롤러.

금감원 '금융상품 표준약관 > 보험약관' 게시판에서 표준약관 PDF를 수집한다.
금감원은 JS 렌더링 게시판이므로 Playwright를 사용한다.

URL: https://www.fss.or.kr/fss/bbs/B0000115/list.do?menuNo=200504
리스크: 최저 (규제기관 직접 제공 공식 자료)
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urljoin

from crawler.base import BaseCrawler, ProductMeta

logger = logging.getLogger(__name__)

BASE_URL = "https://www.fss.or.kr"
LIST_URL = f"{BASE_URL}/fss/bbs/B0000115/list.do?menuNo=200504"
# 게시판 페이지 URL 패턴
LIST_PAGE_URL = f"{BASE_URL}/fss/bbs/B0000115/list.do?menuNo=200504&pageIndex={{page}}"


class FSSStandardCrawler(BaseCrawler):
    """금융감독원 표준약관 크롤러.

    Playwright로 게시판 페이지를 렌더링하여 게시글 목록을 추출하고,
    각 게시글의 첨부파일(PDF)을 다운로드한다.
    """

    source_name = "fss_standard"
    base_url = LIST_URL

    async def fetch_product_list(self) -> list[ProductMeta]:
        """게시판에서 표준약관 게시글 목록 수집.

        Playwright로 JS 렌더링된 페이지를 파싱하여 게시글 제목, 날짜,
        상세 페이지 URL을 추출한다.
        """
        from playwright.async_api import async_playwright

        products: list[ProductMeta] = []

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=self._settings.headless)
            page = await browser.new_page(
                user_agent=self._settings.user_agent,
            )

            page_idx = 1
            max_pages = 20  # 안전 상한

            while page_idx <= max_pages:
                url = LIST_PAGE_URL.format(page=page_idx)
                logger.info("게시판 페이지 로드: %s", url)

                await page.goto(url, wait_until="networkidle", timeout=30000)

                # 게시글 행 추출 — 금감원 게시판은 <table> 기반
                rows = await page.query_selector_all(
                    "table.boardList tbody tr, "
                    "div.board-list table tbody tr, "
                    "div.bbs_list table tbody tr, "
                    "table.bbs-list tbody tr"
                )

                if not rows:
                    # 대체 선택자: 리스트형 게시판
                    rows = await page.query_selector_all(
                        "ul.board-list li, div.list-group a, div.bbs-list li"
                    )

                if not rows:
                    logger.info("게시글 없음 — 페이지 %d에서 종료", page_idx)
                    break

                found_in_page = 0
                for row in rows:
                    try:
                        # 제목과 링크 추출
                        link_el = await row.query_selector("a[href]")
                        if not link_el:
                            continue

                        title = (await link_el.inner_text()).strip()
                        href = await link_el.get_attribute("href")

                        if not title or not href:
                            continue

                        # 날짜 추출
                        date_text = ""
                        date_el = await row.query_selector("td:nth-child(4), span.date, td.date")
                        if date_el:
                            date_text = (await date_el.inner_text()).strip()

                        # 상세 페이지 URL 구성
                        if href.startswith("/"):
                            detail_url = urljoin(BASE_URL, href)
                        elif href.startswith("javascript"):
                            # javascript:fn_detail('nttId') 패턴에서 ID 추출
                            match = re.search(r"'(\d+)'", href)
                            if match:
                                ntt_id = match.group(1)
                                detail_url = (
                                    f"{BASE_URL}/fss/bbs/B0000115/view.do"
                                    f"?menuNo=200504&nttId={ntt_id}"
                                )
                            else:
                                continue
                        else:
                            detail_url = href

                        # 버전: 날짜 또는 제목에서 추출
                        version = date_text or "unknown"

                        products.append(ProductMeta(
                            insurer="금융감독원",
                            product_name=title,
                            version=version,
                            effective_date=date_text or None,
                            download_url=detail_url,
                            extra={"detail_url": detail_url},
                        ))
                        found_in_page += 1

                    except Exception:
                        logger.exception("게시글 파싱 실패")
                        continue

                logger.info("페이지 %d: %d건 추출", page_idx, found_in_page)

                if found_in_page == 0:
                    break

                # 다음 페이지 존재 여부 확인
                next_btn = await page.query_selector(
                    "a.next, a[title='다음'], a.page-next, "
                    "a[onclick*='pageIndex']:last-child"
                )
                if not next_btn:
                    break

                page_idx += 1
                await self._delay()

            await browser.close()

        logger.info("금감원 표준약관 총 %d건 목록 수집 완료", len(products))
        return products

    async def download_pdf(self, product: ProductMeta) -> bytes | None:
        """게시글 상세 페이지에서 첨부 PDF 다운로드.

        상세 페이지에 진입하여 첨부파일 링크를 찾고 PDF를 다운로드한다.
        """
        detail_url = product.extra.get("detail_url") or product.download_url
        if not detail_url:
            logger.warning("상세 URL 없음: %s", product.product_name)
            return None

        from playwright.async_api import async_playwright

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=self._settings.headless)
            page = await browser.new_page(user_agent=self._settings.user_agent)

            logger.info("상세 페이지 로드: %s", detail_url)
            await page.goto(detail_url, wait_until="networkidle", timeout=30000)

            # 첨부파일 링크 탐색 — 다양한 금감원 게시판 패턴
            file_links = await page.query_selector_all(
                "a[href*='.pdf'], "
                "a[href*='download'], "
                "a[href*='fileDown'], "
                "a[href*='atchFileDown'], "
                "div.file-area a, "
                "div.file_area a, "
                "dd.file a"
            )

            pdf_url: str | None = None
            for link in file_links:
                href = await link.get_attribute("href")
                text = await link.inner_text()
                if href and (".pdf" in href.lower() or ".hwp" not in text.lower()):
                    if href.startswith("/"):
                        pdf_url = urljoin(BASE_URL, href)
                    else:
                        pdf_url = href
                    break

            if not pdf_url:
                # 다운로드 이벤트 방식 시도 (일부 금감원 페이지)
                logger.warning(
                    "PDF 링크 미발견: %s — 클릭 시도", product.product_name,
                )
                download_btns = await page.query_selector_all(
                    "a[onclick*='download'], button[onclick*='download'], "
                    "a.file-download, span.file-name a"
                )
                if download_btns:
                    try:
                        async with page.expect_download(timeout=15000) as download_info:
                            await download_btns[0].click()
                        download = await download_info.value
                        file_path = await download.path()
                        if file_path:
                            with open(file_path, "rb") as f:
                                pdf_data = f.read()
                            await browser.close()
                            return pdf_data
                    except Exception:
                        logger.exception("다운로드 클릭 실패: %s", product.product_name)

                await browser.close()
                return None

            await browser.close()

        # PDF URL을 httpx로 다운로드
        logger.info("PDF 다운로드: %s", pdf_url)
        return await self._fetch_bytes(pdf_url)
