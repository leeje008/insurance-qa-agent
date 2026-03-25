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
from api.security import verify_api_key
from api.services import qa_service

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.post("/ask", response_model=AnswerResponse)
async def ask(
    req: QuestionRequest,
    db: AsyncSession = Depends(get_db),
) -> AnswerResponse:
    """보험약관 질의응답."""
    try:
        history = [{"role": m.role, "content": m.content} for m in req.history]
        return await qa_service.ask_question(
            req.question, product_id=req.product_id, history=history, session=db,
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

    각 파이프라인 노드 완료 시 실시간으로 진행 상태를 전달한다.
    """

    async def event_generator():
        try:
            history = [{"role": m.role, "content": m.content} for m in req.history]
            async for chunk in qa_service.ask_question_streaming(
                req.question, product_id=req.product_id, history=history, session=db,
            ):
                yield _sse(chunk["event"], chunk["data"])
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
