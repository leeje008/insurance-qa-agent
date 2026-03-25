"""api.agents.graph - LangGraph 4노드 RAG 파이프라인 (조건부 재시도 포함)."""

from __future__ import annotations

import json
import logging
from typing import Any

from langgraph.graph import END, StateGraph

from api.agents.state import PipelineState
from core.constants import MAX_VALIDATION_RETRIES, MIN_CONFIDENCE_SCORE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 노드 1: Query Processor
# ---------------------------------------------------------------------------

async def query_processor(state: PipelineState) -> dict[str, Any]:
    """질문 분석 — 키워드/의도 추출 + 질문 임베딩."""
    from api.llm.ollama_client import invoke_with_fallback
    from api.llm.prompts import CONTEXTUAL_QUERY_PROMPT, HYDE_PROMPT, QUERY_ANALYSIS_PROMPT
    from api.retrieval.embedder import EmbeddingClient

    question = state["question"]
    history = state.get("conversation_history", [])
    logger.info("Query Processor: %s", question[:80])

    # 멀티턴 대화 맥락이 있으면 질문 재작성
    if history:
        history_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in history[-5:]
        )
        rewrite_prompt = CONTEXTUAL_QUERY_PROMPT.format(
            history=history_text, question=question,
        )
        try:
            question = (await invoke_with_fallback(rewrite_prompt)).strip()
            logger.info("맥락 반영 재작성: %s", question[:80])
        except Exception:
            logger.warning("질문 재작성 실패, 원본 사용", exc_info=True)

    # LLM으로 질문 분석
    prompt = QUERY_ANALYSIS_PROMPT.format(question=question)
    try:
        raw = await invoke_with_fallback(prompt)
        # JSON 파싱
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        analysis = json.loads(raw)
    except (json.JSONDecodeError, Exception):
        logger.warning("질문 분석 JSON 파싱 실패, 기본값 사용", exc_info=True)
        analysis = {
            "keywords": question.split()[:5],
            "intent": "general",
            "article_refs": [],
            "rewritten_query": question,
        }

    # 질문 임베딩 + HyDE 임베딩
    embedder = EmbeddingClient()
    try:
        rewritten = analysis.get("rewritten_query", question)
        query_embedding = await embedder.embed(rewritten)

        # HyDE: 가상 약관 조항 생성 → 임베딩
        hyde_embedding: list[float] = []
        try:
            hyde_text = (
                await invoke_with_fallback(HYDE_PROMPT.format(question=question))
            ).strip()
            if hyde_text:
                hyde_embedding = await embedder.embed(hyde_text)
                logger.info("HyDE 임베딩 생성 완료 (%d chars)", len(hyde_text))
        except Exception:
            logger.warning("HyDE 생성 실패, query embedding만 사용", exc_info=True)
    finally:
        await embedder.close()

    return {
        "keywords": analysis.get("keywords", []),
        "intent": analysis.get("intent", "general"),
        "article_refs": analysis.get("article_refs", []),
        "rewritten_query": analysis.get("rewritten_query", question),
        "query_embedding": query_embedding,
        "hyde_embedding": hyde_embedding,
    }


# ---------------------------------------------------------------------------
# 노드 2: Retriever
# ---------------------------------------------------------------------------

async def retriever(state: PipelineState) -> dict[str, Any]:
    """하이브리드 검색 실행 (query + HyDE dual embedding)."""
    from api.db.database import async_session
    from api.retrieval.hybrid_search import hybrid_search, rrf_fusion

    query_embedding = state.get("query_embedding", [])
    hyde_embedding = state.get("hyde_embedding", [])
    keywords = state.get("keywords", [])
    product_id = state.get("product_id")

    logger.info(
        "Retriever: keywords=%s, product_id=%s, hyde=%s",
        keywords[:3], product_id, bool(hyde_embedding),
    )

    async with async_session() as session:
        results = await hybrid_search(
            session, query_embedding, keywords, product_id=product_id,
        )

        # HyDE 임베딩이 있으면 추가 시맨틱 검색 후 RRF 병합
        if hyde_embedding:
            from api.retrieval.hybrid_search import semantic_search

            hyde_results = await semantic_search(
                session, hyde_embedding, product_id=product_id,
            )
            if hyde_results:
                results = rrf_fusion(results, hyde_results, top_k=len(results))
                logger.info("HyDE 검색 결과 %d건 병합", len(hyde_results))

        # PRF (Pseudo-Relevance Feedback): 상위 문서에서 확장 키워드 추출 후 재검색
        if results and keywords:
            from api.retrieval.hybrid_search import keyword_search
            from api.retrieval.query_expansion import expand_keywords

            initial_docs = [
                {"content": r.content, "title": r.title} for r in results[:3]
            ]
            expanded_kw = expand_keywords(keywords, initial_docs, max_expansion=3)
            if len(expanded_kw) > len(keywords):
                prf_results = await keyword_search(
                    session, expanded_kw, product_id=product_id,
                )
                if prf_results:
                    results = rrf_fusion(results, prf_results, top_k=len(results))
                    logger.info(
                        "PRF 확장 키워드 %d개 → %d건 병합",
                        len(expanded_kw), len(prf_results),
                    )

    documents = [
        {
            "article_id": r.article_id,
            "product_id": r.product_id,
            "number": r.number,
            "title": r.title,
            "content": r.content,
            "level": r.level,
            "score": r.score,
        }
        for r in results
    ]

    logger.info("Retriever: %d건 검색 완료", len(documents))
    return {"documents": documents}


# ---------------------------------------------------------------------------
# 노드 2.5: Reranker
# ---------------------------------------------------------------------------

async def reranker(state: PipelineState) -> dict[str, Any]:
    """리랭커를 사용하여 검색 결과 재순위화."""
    from api.retrieval.reranker import get_reranker

    documents = state.get("documents", [])
    question = state["question"]

    if len(documents) <= 2:
        logger.info("Reranker: 문서 %d건 (리랭킹 생략)", len(documents))
        return {"documents": documents}

    reranker_instance = get_reranker()
    logger.info("Reranker: %s 전략으로 %d건 리랭킹", reranker_instance.name, len(documents))

    results = await reranker_instance.rerank(question, documents)

    reranked_docs = [r.document for r in results]
    logger.info("Reranker: %d건 → %d건", len(documents), len(reranked_docs))
    return {"documents": reranked_docs}


# ---------------------------------------------------------------------------
# 노드 3: Answer Generator
# ---------------------------------------------------------------------------

async def answer_generator(state: PipelineState) -> dict[str, Any]:
    """검색 결과 기반 답변 생성."""
    from api.llm.ollama_client import invoke_with_fallback
    from api.llm.prompts import ANSWER_GENERATION_PROMPT, format_context

    question = state["question"]
    documents = state.get("documents", [])

    if not documents:
        return {
            "answer": "죄송합니다. 관련 약관 조항을 찾지 못했습니다. "
                      "질문을 다시 확인해 주세요.",
            "sources_json": "[]",
        }

    context = format_context(documents)
    prompt = ANSWER_GENERATION_PROMPT.format(
        context=context, question=question,
    )

    logger.info("Answer Generator: %d개 조항 기반 답변 생성", len(documents))
    answer = await invoke_with_fallback(prompt)

    sources = [
        {"number": d["number"], "title": d.get("title", "")}
        for d in documents
    ]

    return {
        "answer": answer.strip(),
        "sources_json": json.dumps(sources, ensure_ascii=False),
    }


# ---------------------------------------------------------------------------
# 노드 4: Answer Validator
# ---------------------------------------------------------------------------

async def answer_validator(state: PipelineState) -> dict[str, Any]:
    """답변 환각 검증 + 신뢰도 평가."""
    from api.llm.ollama_client import invoke_with_fallback
    from api.llm.prompts import ANSWER_VALIDATION_PROMPT, format_context

    question = state["question"]
    answer = state.get("answer", "")
    documents = state.get("documents", [])
    retry_count = state.get("retry_count", 0)

    if not documents or not answer:
        return {
            "is_valid": True,
            "confidence": 0.0,
            "issues": [],
            "retry_count": retry_count,
        }

    context = format_context(documents)
    prompt = ANSWER_VALIDATION_PROMPT.format(
        context=context, question=question, answer=answer,
    )

    logger.info("Answer Validator: retry_count=%d", retry_count)

    try:
        raw = await invoke_with_fallback(prompt)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        validation = json.loads(raw)
    except (json.JSONDecodeError, Exception):
        logger.warning("검증 JSON 파싱 실패, 유효로 처리", exc_info=True)
        return {
            "is_valid": True,
            "confidence": 0.5,
            "issues": ["검증 결과 파싱 실패"],
            "retry_count": retry_count,
        }

    return {
        "is_valid": validation.get("is_valid", True),
        "confidence": float(validation.get("confidence", 0.5)),
        "issues": validation.get("issues", []),
        "retry_count": retry_count + 1,
    }


# ---------------------------------------------------------------------------
# 조건부 라우팅: 재시도 여부 판단
# ---------------------------------------------------------------------------

def should_retry(state: PipelineState) -> str:
    """검증 실패 시 재생성 루프 진입 여부."""
    is_valid = state.get("is_valid", True)
    confidence = state.get("confidence", 1.0)
    retry_count = state.get("retry_count", 0)

    if is_valid and confidence >= MIN_CONFIDENCE_SCORE:
        return "end"

    if retry_count >= MAX_VALIDATION_RETRIES:
        logger.warning(
            "최대 재시도 도달 (%d회), 현재 답변 사용", retry_count,
        )
        return "end"

    logger.info(
        "답변 재생성: valid=%s, confidence=%.2f, retry=%d",
        is_valid, confidence, retry_count,
    )
    return "retry"


# ---------------------------------------------------------------------------
# 그래프 빌드
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """RAG 파이프라인 StateGraph 생성."""
    graph = StateGraph(PipelineState)

    graph.add_node("query_processor", query_processor)
    graph.add_node("retriever", retriever)
    graph.add_node("reranker", reranker)
    graph.add_node("answer_generator", answer_generator)
    graph.add_node("answer_validator", answer_validator)

    graph.set_entry_point("query_processor")
    graph.add_edge("query_processor", "retriever")
    graph.add_edge("retriever", "reranker")
    graph.add_edge("reranker", "answer_generator")
    graph.add_edge("answer_generator", "answer_validator")

    graph.add_conditional_edges(
        "answer_validator",
        should_retry,
        {"end": END, "retry": "answer_generator"},
    )

    return graph


def get_pipeline():
    """컴파일된 RAG 파이프라인 반환."""
    graph = build_graph()
    return graph.compile()
