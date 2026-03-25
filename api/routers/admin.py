"""api.routers.admin - 약관 업로드/관리 엔드포인트."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_db
from api.schemas.admin_schema import (
    IngestRequest,
    IngestResponse,
    ProductListResponse,
    StatsResponse,
)
from api.security import verify_api_key
from api.services import admin_service

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.get("/products", response_model=ProductListResponse)
async def list_products(
    db: AsyncSession = Depends(get_db),
) -> ProductListResponse:
    """보험 상품 목록 조회."""
    return await admin_service.list_products(db)


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest) -> IngestResponse:
    """PDF 약관 인제스트."""
    try:
        return await admin_service.ingest_policy(
            pdf_path=req.pdf_path,
            insurer=req.insurer,
            product_name=req.product_name,
            insurance_type=req.insurance_type,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/stats", response_model=StatsResponse)
async def stats(
    db: AsyncSession = Depends(get_db),
) -> StatsResponse:
    """시스템 통계 조회."""
    return await admin_service.get_stats(db)


@router.post("/eval/reranker-benchmark")
async def reranker_benchmark(
    k: int = 5,
    strategies: list[str] | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """리랭커 전략별 MRR/NDCG 벤치마크 (golden set 기반)."""
    from api.evaluation.dataset import build_from_feedback, load_golden_set
    from api.evaluation.runner import benchmark_rerankers

    # golden set 로드 (파일 우선, 없으면 피드백에서 생성)
    golden_set = load_golden_set()
    if not golden_set:
        golden_set = await build_from_feedback(db)

    if not golden_set:
        return {"error": "golden set이 비어 있습니다. 피드백 데이터가 필요합니다."}

    results = await benchmark_rerankers(db, golden_set, strategies=strategies, k=k)
    return {"k": k, "num_queries": len(golden_set), "results": results}


@router.get("/eval/annotation-queue")
async def annotation_queue(
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Active Learning 기반 어노테이션 우선순위 큐."""
    from api.evaluation.active_learning import build_annotation_queue

    queue = await build_annotation_queue(db, limit=limit)
    return {
        "total_unlabeled": queue.total_unlabeled,
        "queue_size": len(queue.candidates),
        "strategy_counts": queue.strategy_counts,
        "candidates": [
            {
                "qa_log_id": c.qa_log_id,
                "question": c.question[:100],
                "priority_score": round(c.priority_score, 3),
                "strategy": c.strategy,
                "reason": c.reason,
            }
            for c in queue.candidates
        ],
    }


@router.post("/eval/failure-analysis")
async def failure_analysis(
    max_feedback_score: int = 2,
    limit: int = 200,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Retrieval 실패 분석 — 6가지 에러 택소노미 기반."""
    from api.evaluation.failure_analysis import analyze_failures

    report = await analyze_failures(db, max_feedback_score=max_feedback_score, limit=limit)
    return {
        "total_failures": report.total_failures,
        "mode_counts": report.mode_counts,
        "mode_percentages": report.mode_percentages,
        "recommendations": report.recommendations,
        "sample_cases": [
            {
                "qa_log_id": c.qa_log_id,
                "question": c.question[:100],
                "feedback_score": c.feedback_score,
                "confidence": c.confidence,
                "failure_modes": [str(m) for m in c.failure_modes],
            }
            for c in report.cases[:20]
        ],
    }


@router.post("/rerank/compare")
async def compare_reranking(
    question: str,
    product_id: int | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """동일 질문에 대해 모든 리랭커 전략 결과를 비교 반환."""
    from api.retrieval.embedder import EmbeddingClient
    from api.retrieval.hybrid_search import hybrid_search
    from api.retrieval.reranker import compare_rerankers, list_rerankers

    # 1. 임베딩 생성
    embedder = EmbeddingClient()
    try:
        query_embedding = await embedder.embed(question)
    finally:
        await embedder.close()

    # 2. 하이브리드 검색 (over-fetch)
    results = await hybrid_search(
        db, query_embedding, question.split()[:5], product_id=product_id,
    )

    documents = [
        {
            "article_id": r.article_id, "product_id": r.product_id,
            "number": r.number, "title": r.title,
            "content": r.content, "level": r.level, "score": r.score,
        }
        for r in results
    ]

    # 3. 모든 리랭커로 비교
    comparison = await compare_rerankers(question, documents)

    return {
        "question": question,
        "strategies": list_rerankers(),
        "initial_count": len(documents),
        "results": {
            name: [
                {
                    "rank": r.rank, "score": r.score,
                    "number": r.document.get("number", ""),
                    "title": r.document.get("title", ""),
                }
                for r in rerank_results
            ]
            for name, rerank_results in comparison.items()
        },
    }
