"""api.routers.qa - Q&A 엔드포인트."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.repositories import qa_log_repo
from api.dependencies import get_db
from api.schemas.qa_schema import (
    AnswerResponse,
    FeedbackRequest,
    QALogResponse,
    QuestionRequest,
)
from api.services import qa_service

router = APIRouter()


@router.post("/ask", response_model=AnswerResponse)
async def ask(
    req: QuestionRequest,
    db: AsyncSession = Depends(get_db),
) -> AnswerResponse:
    """보험약관 질의응답."""
    try:
        return await qa_service.ask_question(
            req.question, product_id=req.product_id, session=db,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/history", response_model=list[QALogResponse])
async def history(
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
) -> list[QALogResponse]:
    """Q&A 히스토리 조회."""
    logs = await qa_log_repo.list_qa_logs(db, limit=limit, offset=offset)
    return [QALogResponse.model_validate(log) for log in logs]


@router.post("/{log_id}/feedback")
async def feedback(
    log_id: int,
    req: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Q&A 피드백 등록."""
    updated = await qa_log_repo.update_feedback(db, log_id, req.score)
    if not updated:
        raise HTTPException(status_code=404, detail="Q&A 로그를 찾을 수 없습니다")
    await db.commit()
    return {"status": "ok", "log_id": log_id, "score": req.score}


@router.post("/ask/stream")
async def ask_stream(
    req: QuestionRequest,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """SSE 스트리밍 질의응답.

    각 파이프라인 단계 진행 상황을 실시간으로 전달한다.
    """

    async def event_generator():
        yield _sse("status", {"stage": "query_processing", "message": "질문 분석 중..."})

        try:
            result = await qa_service.ask_question(
                req.question, product_id=req.product_id, session=db,
            )

            yield _sse("status", {"stage": "completed", "message": "답변 생성 완료"})
            yield _sse("answer", {
                "answer": result.answer,
                "confidence": result.confidence,
                "sources": [s.model_dump() for s in result.sources],
                "log_id": result.log_id,
            })
        except Exception as exc:
            yield _sse("error", {"message": str(exc)})

        yield _sse("done", {})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: str, data: dict) -> str:
    """SSE 이벤트 포맷."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
