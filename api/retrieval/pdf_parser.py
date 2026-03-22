"""api.retrieval.pdf_parser - 보험약관 PDF 파싱 (관/조/항/호 구조 추출)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber

from core.constants import ArticleLevel
from core.utils.text import extract_article_numbers, normalize_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 조항 계층 패턴 정규식
# ---------------------------------------------------------------------------

# 관 (Part): "제1관", "제 1 관"
RE_PART = re.compile(r"^제\s*(\d+)\s*관\s*[.\s]*(.*)")
# 조 (Article): "제1조", "제 1 조 (제목)"
RE_ARTICLE = re.compile(r"^제\s*(\d+)\s*조\s*[\(（]?\s*(.*?)[\)）]?\s*$")
# 항 (Paragraph): "① ②" 또는 "1. 2."
RE_PARAGRAPH = re.compile(r"^([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])\s*(.*)")
RE_PARAGRAPH_NUM = re.compile(r"^(\d{1,2})\.\s+(.*)")
# 호 (Item): "1. 2." (항 아래 세부 항목) 또는 "가. 나. 다."
RE_ITEM = re.compile(r"^([가나다라마바사아자차카타파하])\.\s+(.*)")

# 별표/부표 (용어 사전 영역)
RE_GLOSSARY_HEADER = re.compile(r"[【\[<]?\s*별\s*표\s*[】\]>]?|용어\s*(?:의\s*)?정의|용어\s*설명")
# 용어 정의 패턴: "「용어」: 정의" 또는 "'용어'란 ..."
RE_TERM_DEF = re.compile(
    r"[「\"](.+?)[」\"]\s*[:：]\s*(.+)|"
    r"[''](.+?)[''](?:이)?란\s+(.+)|"
    r"(\d+)\.\s*[\"「](.+?)[\"」]\s*[:：]\s*(.+)"
)

# 조항 간 참조 패턴
RE_CROSS_REF = re.compile(r"제\s*\d+\s*조(?:\s*제\s*\d+\s*항)?")


# ---------------------------------------------------------------------------
# 파싱 결과 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class ParsedArticle:
    """파싱된 약관 조항."""

    level: ArticleLevel
    number: str
    title: str
    content: str
    parent_number: str | None = None
    references: list[str] = field(default_factory=list)


@dataclass
class ParsedGlossary:
    """파싱된 용어 사전 항목."""

    term: str
    definition: str


@dataclass
class ParsedPolicy:
    """PDF 파싱 결과 전체."""

    product_name: str
    insurer: str
    articles: list[ParsedArticle] = field(default_factory=list)
    glossary: list[ParsedGlossary] = field(default_factory=list)
    raw_text: str = ""


# ---------------------------------------------------------------------------
# PDF 텍스트 추출
# ---------------------------------------------------------------------------

def extract_text_pymupdf(pdf_path: Path) -> str:
    """PyMuPDF로 PDF 텍스트 추출."""
    doc = fitz.open(str(pdf_path))
    pages: list[str] = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return "\n".join(pages)


def extract_text_pdfplumber(pdf_path: Path) -> str:
    """pdfplumber로 PDF 텍스트 추출 (테이블 구조 보존)."""
    pages: list[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n".join(pages)


def extract_text(pdf_path: Path) -> str:
    """PDF에서 텍스트 추출. PyMuPDF 우선, 실패 시 pdfplumber 폴백."""
    try:
        text = extract_text_pymupdf(pdf_path)
        if text.strip():
            return text
    except Exception:
        logger.debug("PyMuPDF 추출 실패, pdfplumber 시도", exc_info=True)

    return extract_text_pdfplumber(pdf_path)


# ---------------------------------------------------------------------------
# 조항 계층 구조 파싱
# ---------------------------------------------------------------------------

def _detect_level(line: str) -> tuple[ArticleLevel, str, str] | None:
    """줄에서 조항 수준/번호/제목을 감지.

    Returns:
        (level, number, title) 또는 매칭 실패 시 None.
    """
    line = line.strip()

    m = RE_PART.match(line)
    if m:
        return ArticleLevel.PART, f"제{m.group(1)}관", m.group(2).strip()

    m = RE_ARTICLE.match(line)
    if m:
        return ArticleLevel.ARTICLE, f"제{m.group(1)}조", m.group(2).strip()

    m = RE_PARAGRAPH.match(line)
    if m:
        circled = m.group(1)
        idx = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳".index(circled) + 1
        return ArticleLevel.PARAGRAPH, f"제{idx}항", m.group(2).strip()

    m = RE_ITEM.match(line)
    if m:
        return ArticleLevel.ITEM, f"{m.group(1)}호", m.group(2).strip()

    return None


def parse_articles(text: str) -> list[ParsedArticle]:
    """텍스트에서 관/조/항/호 계층 구조를 추출."""
    articles: list[ParsedArticle] = []
    lines = text.split("\n")

    current_part: str | None = None
    current_article: str | None = None
    current_paragraph: str | None = None
    current: ParsedArticle | None = None

    for line in lines:
        line = line.rstrip()
        if not line.strip():
            if current:
                current.content += "\n"
            continue

        detected = _detect_level(line)

        if detected:
            level, number, title = detected

            if current:
                current.content = normalize_text(current.content)
                refs = extract_article_numbers(current.content)
                current.references = [
                    r for r in refs if r != current.number
                ]
                articles.append(current)

            parent = None
            if level == ArticleLevel.PART:
                current_part = number
                current_article = None
                current_paragraph = None
            elif level == ArticleLevel.ARTICLE:
                current_article = number
                current_paragraph = None
                parent = current_part
            elif level == ArticleLevel.PARAGRAPH:
                current_paragraph = number
                parent = current_article
            elif level == ArticleLevel.ITEM:
                parent = current_paragraph or current_article

            current = ParsedArticle(
                level=level,
                number=number,
                title=title,
                content=line,
                parent_number=parent,
            )
        elif current:
            current.content += "\n" + line

    # 마지막 조항
    if current:
        current.content = normalize_text(current.content)
        refs = extract_article_numbers(current.content)
        current.references = [r for r in refs if r != current.number]
        articles.append(current)

    return articles


# ---------------------------------------------------------------------------
# 용어 사전 추출
# ---------------------------------------------------------------------------

def parse_glossary(text: str) -> list[ParsedGlossary]:
    """텍스트에서 용어 사전(별표) 영역을 추출."""
    glossary: list[ParsedGlossary] = []
    lines = text.split("\n")
    in_glossary = False

    for line in lines:
        line = line.strip()

        if RE_GLOSSARY_HEADER.search(line):
            in_glossary = True
            continue

        if not in_glossary:
            continue

        # 새로운 조(제N조) 시작이면 용어 사전 영역 종료
        if RE_ARTICLE.match(line):
            in_glossary = False
            continue

        m = RE_TERM_DEF.match(line)
        if m:
            groups = m.groups()
            if groups[0] and groups[1]:
                term, definition = groups[0], groups[1]
            elif groups[2] and groups[3]:
                term, definition = groups[2], groups[3]
            elif groups[4] and groups[5] and groups[6]:
                term, definition = groups[5], groups[6]
            else:
                continue

            glossary.append(ParsedGlossary(
                term=normalize_text(term),
                definition=normalize_text(definition),
            ))

    return glossary


# ---------------------------------------------------------------------------
# 메인 파서
# ---------------------------------------------------------------------------

def parse_policy_pdf(
    pdf_path: Path,
    *,
    product_name: str = "",
    insurer: str = "",
) -> ParsedPolicy:
    """보험약관 PDF를 파싱하여 구조화된 결과를 반환.

    Args:
        pdf_path: PDF 파일 경로.
        product_name: 보험 상품명.
        insurer: 보험사명.

    Returns:
        ParsedPolicy 객체 (조항 목록, 용어 사전, 원문 텍스트).
    """
    logger.info("PDF 파싱 시작: %s", pdf_path)

    raw_text = extract_text(pdf_path)
    if not raw_text.strip():
        logger.warning("PDF 텍스트 없음: %s", pdf_path)
        return ParsedPolicy(
            product_name=product_name, insurer=insurer, raw_text=""
        )

    articles = parse_articles(raw_text)
    glossary = parse_glossary(raw_text)

    logger.info(
        "PDF 파싱 완료: %s — 조항 %d건, 용어 %d건",
        pdf_path, len(articles), len(glossary),
    )

    return ParsedPolicy(
        product_name=product_name or pdf_path.stem,
        insurer=insurer,
        articles=articles,
        glossary=glossary,
        raw_text=raw_text,
    )
