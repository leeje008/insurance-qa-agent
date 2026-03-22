"""api.ingest - PDF 인제스트 파이프라인 CLI.

약관 PDF를 파싱하여 임베딩을 생성하고 DB에 저장한다.

Usage:
    uv run python -m api.ingest --pdf data/samsungfire/화재보험.pdf --insurer 삼성화재
    uv run python -m api.ingest --dir data/samsungfire/ --insurer 삼성화재
    uv run python -m api.ingest --manifest data/manifest.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from core.logging_config import setup_logging

logger = logging.getLogger(__name__)


async def ingest_pdf(
    pdf_path: Path,
    *,
    insurer: str,
    product_name: str = "",
    insurance_type: str = "other",
    skip_embedding: bool = False,
) -> dict:
    """단일 PDF 인제스트.

    1. PDF 파싱 (조항/용어 추출)
    2. 임베딩 생성 (skip_embedding=False인 경우)
    3. DB 저장

    Returns:
        인제스트 결과 요약 dict.
    """
    from api.db.database import async_session
    from api.db.repositories import policy_repo
    from api.retrieval.embedder import EmbeddingClient
    from api.retrieval.pdf_parser import parse_policy_pdf

    parsed = parse_policy_pdf(
        pdf_path,
        product_name=product_name or pdf_path.stem,
        insurer=insurer,
    )

    if not parsed.articles:
        logger.warning("조항 없음: %s", pdf_path)
        return {"file": str(pdf_path), "articles": 0, "glossary": 0}

    embedding_client = EmbeddingClient() if not skip_embedding else None

    async with async_session() as session:
        async with session.begin():
            # 1. 상품 생성/조회
            existing = await policy_repo.get_product_by_name(
                session, parsed.product_name, parsed.insurer,
            )
            if existing:
                product = existing
                logger.info(
                    "기존 상품 사용: id=%d, name=%s",
                    product.id, product.name,
                )
            else:
                product = await policy_repo.create_product(
                    session,
                    name=parsed.product_name,
                    insurance_type=insurance_type,
                    insurer=parsed.insurer,
                )

            # 2. 조항 저장 + 임베딩
            article_map: dict[str, int] = {}  # number → article.id
            saved_articles = 0

            for art in parsed.articles:
                # 부모 ID 매핑
                parent_id = None
                if art.parent_number and art.parent_number in article_map:
                    parent_id = article_map[art.parent_number]

                # 임베딩 생성
                embedding = None
                sub_chunks = []
                if embedding_client:
                    try:
                        embedding, sub_chunks = (
                            await embedding_client.embed_with_chunks(art.content)
                        )
                    except Exception:
                        logger.warning(
                            "임베딩 실패: %s — 임베딩 없이 저장",
                            art.number, exc_info=True,
                        )

                article = await policy_repo.create_article(
                    session,
                    product_id=product.id,
                    level=art.level,
                    number=art.number,
                    title=art.title or None,
                    content=art.content,
                    parent_id=parent_id,
                    embedding=embedding,
                )
                article_map[art.number] = article.id
                saved_articles += 1

                # 서브 청크 저장
                for chunk_idx, chunk_text, chunk_embedding in sub_chunks:
                    await policy_repo.create_sub_chunk(
                        session,
                        article_id=article.id,
                        chunk_index=chunk_idx,
                        content=chunk_text,
                        embedding=chunk_embedding,
                    )

            # 3. 조항 간 참조 관계 저장
            ref_count = 0
            for art in parsed.articles:
                if not art.references:
                    continue
                source_id = article_map.get(art.number)
                if not source_id:
                    continue
                for ref_num in art.references:
                    target_id = article_map.get(ref_num)
                    if target_id:
                        await policy_repo.create_reference(
                            session,
                            source_id=source_id,
                            target_id=target_id,
                        )
                        ref_count += 1

            # 4. 용어 사전 저장
            for gloss in parsed.glossary:
                await policy_repo.create_glossary_term(
                    session,
                    term=gloss.term,
                    definition=gloss.definition,
                    product_id=product.id,
                )

    if embedding_client:
        await embedding_client.close()

    result = {
        "file": str(pdf_path),
        "product_id": product.id,
        "articles": saved_articles,
        "references": ref_count,
        "glossary": len(parsed.glossary),
    }
    logger.info("인제스트 완료: %s", json.dumps(result, ensure_ascii=False))
    return result


async def ingest_directory(
    dir_path: Path,
    *,
    insurer: str,
    insurance_type: str = "other",
    skip_embedding: bool = False,
) -> list[dict]:
    """디렉토리 내 모든 PDF 인제스트."""
    pdf_files = sorted(dir_path.glob("*.pdf"))
    if not pdf_files:
        logger.warning("PDF 파일 없음: %s", dir_path)
        return []

    logger.info("인제스트 대상: %d개 PDF in %s", len(pdf_files), dir_path)
    results = []
    for pdf in pdf_files:
        result = await ingest_pdf(
            pdf,
            insurer=insurer,
            insurance_type=insurance_type,
            skip_embedding=skip_embedding,
        )
        results.append(result)

    return results


async def ingest_from_manifest(
    manifest_path: Path,
    *,
    skip_embedding: bool = False,
) -> list[dict]:
    """manifest.json 기반 인제스트."""
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    data_dir = manifest_path.parent
    results = []

    for entry in manifest:
        pdf_path = data_dir / entry["file_path"]
        if not pdf_path.exists():
            logger.warning("PDF 없음: %s", pdf_path)
            continue

        result = await ingest_pdf(
            pdf_path,
            insurer=entry.get("insurer", ""),
            product_name=entry.get("product_name", ""),
            skip_embedding=skip_embedding,
        )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI 메인."""
    setup_logging()

    parser = argparse.ArgumentParser(description="보험약관 PDF 인제스트")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf", type=Path, help="단일 PDF 파일")
    group.add_argument("--dir", type=Path, help="PDF 디렉토리")
    group.add_argument("--manifest", type=Path, help="manifest.json 경로")

    parser.add_argument("--insurer", type=str, default="", help="보험사명")
    parser.add_argument("--type", type=str, default="other", help="보험 유형")
    parser.add_argument(
        "--skip-embedding", action="store_true",
        help="임베딩 생성 건너뛰기 (DB/Ollama 없이 테스트용)",
    )

    args = parser.parse_args()

    if args.pdf:
        if not args.insurer:
            parser.error("--pdf 사용 시 --insurer 필수")
        results = asyncio.run(ingest_pdf(
            args.pdf,
            insurer=args.insurer,
            insurance_type=args.type,
            skip_embedding=args.skip_embedding,
        ))
        results = [results]
    elif args.dir:
        if not args.insurer:
            parser.error("--dir 사용 시 --insurer 필수")
        results = asyncio.run(ingest_directory(
            args.dir,
            insurer=args.insurer,
            insurance_type=args.type,
            skip_embedding=args.skip_embedding,
        ))
    else:
        results = asyncio.run(ingest_from_manifest(
            args.manifest,
            skip_embedding=args.skip_embedding,
        ))

    total_articles = sum(r.get("articles", 0) for r in results)
    total_glossary = sum(r.get("glossary", 0) for r in results)
    print(
        f"\n인제스트 완료: PDF {len(results)}건, "
        f"조항 {total_articles}건, 용어 {total_glossary}건"
    )


if __name__ == "__main__":
    main()
