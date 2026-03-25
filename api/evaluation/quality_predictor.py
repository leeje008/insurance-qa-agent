"""api.evaluation.quality_predictor - Answer Quality 예측 모델.

GradientBoosting 회귀로 feedback_score를 예측.
예측값 ≥ threshold이면 LLM 기반 answer_validator를 스킵하여 레이턴시 절감.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import QALog, RetrievalLog

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "quality_model.pkl"

# 스킵 임계값: 예측된 품질이 이 값 이상이면 LLM 검증 생략
QUALITY_SKIP_THRESHOLD = 4.0


@dataclass
class QualityFeatures:
    """답변 품질 예측 피처."""

    confidence: float
    num_sources: int
    answer_length: int
    question_length: int
    max_retrieval_score: float
    mean_retrieval_score: float
    score_gap: float  # top-1과 top-2 점수 차이
    num_retrieved: int
    has_reranking: int  # 0 or 1

    def to_array(self) -> list[float]:
        return [
            self.confidence,
            self.num_sources,
            self.answer_length,
            self.question_length,
            self.max_retrieval_score,
            self.mean_retrieval_score,
            self.score_gap,
            self.num_retrieved,
            self.has_reranking,
        ]

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "confidence", "num_sources", "answer_length", "question_length",
            "max_retrieval_score", "mean_retrieval_score", "score_gap",
            "num_retrieved", "has_reranking",
        ]


class QualityPredictor:
    """GradientBoosting 기반 답변 품질 예측기."""

    def __init__(self) -> None:
        self._model = None
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, float]:
        """모델 학습.

        Args:
            features: 피처 행렬 (n_samples x n_features).
            labels: feedback_score (1-5).

        Returns:
            학습 결과 (mae, r2, n_samples).
        """
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import cross_val_score

        if len(labels) < 30:
            return {"error": "최소 30건의 학습 데이터 필요", "n_samples": len(labels)}

        self._model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )
        self._model.fit(features, labels)
        self._is_trained = True

        # 교차검증
        mae = 0.0
        r2 = 0.0
        if len(labels) >= 50:
            mae_scores = cross_val_score(
                self._model, features, labels, cv=5, scoring="neg_mean_absolute_error",
            )
            mae = float(-mae_scores.mean())
            r2_scores = cross_val_score(
                self._model, features, labels, cv=5, scoring="r2",
            )
            r2 = float(r2_scores.mean())

        # 피처 중요도
        importance = dict(zip(
            QualityFeatures.feature_names(),
            self._model.feature_importances_.tolist(),
        ))

        logger.info(
            "Quality 예측 모델 학습 완료: MAE=%.3f, R²=%.3f, importance=%s",
            mae, r2, importance,
        )

        return {
            "n_samples": len(labels),
            "mae": mae,
            "r2": r2,
            "feature_importance": importance,
        }

    def predict(self, features: np.ndarray) -> float:
        """품질 점수 예측 (1-5 범위)."""
        if not self._is_trained:
            return 0.0
        pred = float(self._model.predict(features.reshape(1, -1))[0])
        return max(1.0, min(5.0, pred))

    def should_skip_validation(self, features: np.ndarray) -> bool:
        """예측 품질이 높으면 LLM 검증 스킵 권장."""
        if not self._is_trained:
            return False
        return self.predict(features) >= QUALITY_SKIP_THRESHOLD

    def save(self, path: Path = MODEL_PATH) -> None:
        """모델 저장."""
        import joblib

        if not self._is_trained:
            raise RuntimeError("저장할 모델 없음")
        joblib.dump(self._model, path)
        logger.info("Quality 예측 모델 저장: %s", path)

    def load(self, path: Path = MODEL_PATH) -> bool:
        """모델 로드. 성공 시 True."""
        if not path.exists():
            return False
        import joblib

        self._model = joblib.load(path)
        self._is_trained = True
        logger.info("Quality 예측 모델 로드: %s", path)
        return True


async def build_training_data(
    session: AsyncSession,
) -> tuple[np.ndarray, np.ndarray]:
    """qa_logs + retrieval_logs에서 학습 데이터 구축.

    Returns:
        (features, labels) — 피처 행렬, feedback_score 라벨
    """
    # feedback이 있는 qa_logs
    stmt = (
        select(
            QALog.id,
            QALog.question,
            QALog.answer,
            QALog.confidence,
            QALog.sources_json,
            QALog.feedback_score,
        )
        .where(QALog.feedback_score.is_not(None))
    )
    result = await session.execute(stmt)
    logs = result.all()

    if not logs:
        return np.array([]), np.array([])

    features_list = []
    labels = []

    for log in logs:
        qa_log_id, question, answer, confidence, sources_json, feedback = log

        # retrieval_logs에서 점수 집계
        rl_stmt = (
            select(
                func.count(RetrievalLog.id).label("num_retrieved"),
                func.max(RetrievalLog.rrf_score).label("max_score"),
                func.avg(RetrievalLog.rrf_score).label("avg_score"),
                func.count(
                    RetrievalLog.rerank_score
                ).label("has_rerank"),
            )
            .where(RetrievalLog.qa_log_id == qa_log_id)
        )
        rl_result = await session.execute(rl_stmt)
        rl_row = rl_result.one_or_none()

        num_retrieved = rl_row.num_retrieved if rl_row else 0
        max_score = float(rl_row.max_score or 0) if rl_row else 0
        avg_score = float(rl_row.avg_score or 0) if rl_row else 0
        has_rerank = 1 if (rl_row and rl_row.has_rerank > 0) else 0

        # 소스 수 계산
        import json
        try:
            num_sources = len(json.loads(sources_json)) if sources_json else 0
        except (json.JSONDecodeError, TypeError):
            num_sources = 0

        feat = QualityFeatures(
            confidence=confidence or 0,
            num_sources=num_sources,
            answer_length=len(answer or ""),
            question_length=len(question or ""),
            max_retrieval_score=max_score,
            mean_retrieval_score=avg_score,
            score_gap=max_score - avg_score,
            num_retrieved=num_retrieved,
            has_reranking=has_rerank,
        )
        features_list.append(feat.to_array())
        labels.append(feedback)

    logger.info("Quality 학습 데이터: %d건", len(labels))
    return np.array(features_list, dtype=np.float32), np.array(labels, dtype=np.float32)
