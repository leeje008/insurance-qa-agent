"""api.evaluation.ltr - Learning-to-Rank (LightGBM LambdaRank).

retrieval_logs의 피처를 비선형 결합하여 문서 관련성을 학습.
기존 정적 RRF 가중치 + 블랙박스 리랭커를 대체 가능.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import PolicyArticle, QALog, RetrievalLog

logger = logging.getLogger(__name__)

LTR_MODEL_PATH = Path(__file__).parent / "ltr_model.txt"


@dataclass
class LTRFeatures:
    """LTR 피처 벡터."""

    semantic_score: float
    keyword_score: float
    rrf_score: float
    rerank_score: float
    doc_length: int
    article_level: int  # part=0, article=1, paragraph=2, item=3
    title_match: float  # 0.0 or 1.0
    query_length: int
    num_keywords: int
    score_gap: float  # top-1과 현재 문서 점수 차이

    def to_array(self) -> list[float]:
        return [
            self.semantic_score,
            self.keyword_score,
            self.rrf_score,
            self.rerank_score,
            self.doc_length,
            self.article_level,
            self.title_match,
            self.query_length,
            self.num_keywords,
            self.score_gap,
        ]

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "semantic_score", "keyword_score", "rrf_score", "rerank_score",
            "doc_length", "article_level", "title_match",
            "query_length", "num_keywords", "score_gap",
        ]


_LEVEL_MAP = {"part": 0, "article": 1, "paragraph": 2, "item": 3}


def extract_features(
    question: str,
    documents: list[dict],
    keywords: list[str] | None = None,
) -> list[list[float]]:
    """쿼리-문서 쌍에서 LTR 피처를 추출."""
    query_len = len(question)
    num_kw = len(keywords) if keywords else 0
    max_score = max((d.get("score", 0) for d in documents), default=0)
    query_terms = set(question.lower().split())

    features = []
    for doc in documents:
        title = (doc.get("title") or "").lower()
        title_match = 1.0 if any(t in title for t in query_terms) else 0.0
        level = _LEVEL_MAP.get(doc.get("level", "article"), 1)

        f = LTRFeatures(
            semantic_score=doc.get("semantic_score", doc.get("score", 0)),
            keyword_score=doc.get("keyword_score", 0),
            rrf_score=doc.get("rrf_score", doc.get("score", 0)),
            rerank_score=doc.get("rerank_score", 0),
            doc_length=len(doc.get("content", "")),
            article_level=level,
            title_match=title_match,
            query_length=query_len,
            num_keywords=num_kw,
            score_gap=max_score - doc.get("score", 0),
        )
        features.append(f.to_array())

    return features


async def build_training_data(
    session: AsyncSession,
    min_feedback: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """retrieval_logs + qa_logs에서 LTR 학습 데이터 구축.

    Returns:
        (X, y, groups) — 피처 행렬, 라벨, 쿼리 그룹 크기
    """
    # qa_logs with feedback 조인 retrieval_logs
    stmt = (
        select(
            RetrievalLog.qa_log_id,
            RetrievalLog.semantic_score,
            RetrievalLog.keyword_score,
            RetrievalLog.rrf_score,
            RetrievalLog.rerank_score,
            RetrievalLog.was_used_in_answer,
            RetrievalLog.article_id,
            QALog.question,
            QALog.feedback_score,
            PolicyArticle.content,
            PolicyArticle.level,
            PolicyArticle.title,
        )
        .join(QALog, RetrievalLog.qa_log_id == QALog.id)
        .join(PolicyArticle, RetrievalLog.article_id == PolicyArticle.id)
        .where(QALog.feedback_score >= min_feedback)
        .order_by(RetrievalLog.qa_log_id, RetrievalLog.rank_before_rerank)
    )

    result = await session.execute(stmt)
    rows = result.all()

    if not rows:
        return np.array([]), np.array([]), np.array([])

    features_list = []
    labels = []
    groups = []
    current_qa_id = None
    group_size = 0

    for row in rows:
        if current_qa_id != row.qa_log_id:
            if current_qa_id is not None:
                groups.append(group_size)
            current_qa_id = row.qa_log_id
            group_size = 0

        level = _LEVEL_MAP.get(row.level or "article", 1)
        query_terms = set((row.question or "").lower().split())
        title_match = 1.0 if any(
            t in (row.title or "").lower() for t in query_terms
        ) else 0.0

        features_list.append([
            row.semantic_score or 0,
            row.keyword_score or 0,
            row.rrf_score or 0,
            row.rerank_score or 0,
            len(row.content or ""),
            level,
            title_match,
            len(row.question or ""),
            len(query_terms),
            0,  # score_gap은 그룹 내 후처리로 계산
        ])

        # 라벨: used_in_answer + feedback_score 기반
        if row.was_used_in_answer and row.feedback_score:
            labels.append(row.feedback_score)
        else:
            labels.append(0)

        group_size += 1

    if current_qa_id is not None:
        groups.append(group_size)

    x_data = np.array(features_list, dtype=np.float32)
    y_data = np.array(labels, dtype=np.float32)
    group_arr = np.array(groups, dtype=np.int32)

    # score_gap 후처리: 각 그룹 내 max_score - current_score
    idx = 0
    for g in group_arr:
        if g > 0:
            group_scores = x_data[idx:idx + g, 2]  # rrf_score
            max_score = group_scores.max()
            x_data[idx:idx + g, 9] = max_score - group_scores
        idx += g

    logger.info(
        "LTR 학습 데이터: %d 페어, %d 쿼리 그룹",
        len(y_data), len(group_arr),
    )

    return x_data, y_data, group_arr


class LTRModel:
    """LightGBM LambdaRank 래퍼."""

    def __init__(self) -> None:
        self._model = None

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def train(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        groups: np.ndarray,
        *,
        num_leaves: int = 31,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
    ) -> dict[str, float]:
        """LambdaRank 모델 학습."""
        import lightgbm as lgb

        train_data = lgb.Dataset(
            x_data, label=y_data, group=groups.tolist(),
            feature_name=LTRFeatures.feature_names(),
        )

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [3, 5],
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "verbose": -1,
        }

        self._model = lgb.train(
            params,
            train_data,
            num_boost_round=n_estimators,
        )

        # 피처 중요도
        importance = dict(zip(
            LTRFeatures.feature_names(),
            self._model.feature_importance(importance_type="gain").tolist(),
        ))

        logger.info("LTR 모델 학습 완료 — 피처 중요도: %s", importance)
        return {"feature_importance": importance, "n_estimators": n_estimators}

    def predict(self, x_data: np.ndarray) -> np.ndarray:
        """관련성 점수 예측."""
        if self._model is None:
            raise RuntimeError("모델이 학습되지 않았습니다")
        return self._model.predict(x_data)

    def save(self, path: Path = LTR_MODEL_PATH) -> None:
        """모델 저장."""
        if self._model is None:
            raise RuntimeError("저장할 모델 없음")
        self._model.save_model(str(path))
        logger.info("LTR 모델 저장: %s", path)

    def load(self, path: Path = LTR_MODEL_PATH) -> bool:
        """모델 로드. 성공 시 True."""
        if not path.exists():
            return False
        import lightgbm as lgb

        self._model = lgb.Booster(model_file=str(path))
        logger.info("LTR 모델 로드: %s", path)
        return True
