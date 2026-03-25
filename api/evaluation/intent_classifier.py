"""api.evaluation.intent_classifier - 경량 Query Intent 분류 모델.

TF-IDF + LinearSVC 기반 — LLM 호출 없이 <50ms 추론.
6 클래스: coverage, claim, exclusion, definition, comparison, general.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import QALog

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "intent_model"
INTENT_CLASSES = ["coverage", "claim", "exclusion", "definition", "comparison", "general"]


class IntentClassifier:
    """TF-IDF + LinearSVC 기반 Intent 분류기."""

    def __init__(self) -> None:
        self._vectorizer = None
        self._model = None
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(
        self,
        questions: list[str],
        intents: list[str],
    ) -> dict[str, float]:
        """분류기 학습.

        Args:
            questions: 질문 텍스트 리스트.
            intents: 대응하는 intent 라벨 리스트.

        Returns:
            학습 결과 (accuracy, n_samples, class_distribution).
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import cross_val_score
        from sklearn.svm import LinearSVC

        if len(questions) < 30:
            return {"error": "최소 30건의 학습 데이터 필요", "n_samples": len(questions)}

        # TF-IDF (character n-gram 3-5, 한국어에 효과적)
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=5000,
            sublinear_tf=True,
        )
        x_data = self._vectorizer.fit_transform(questions)

        # LinearSVC
        self._model = LinearSVC(max_iter=2000, class_weight="balanced")
        self._model.fit(x_data, intents)
        self._is_trained = True

        # 교차검증 (데이터 충분 시)
        accuracy = 0.0
        if len(questions) >= 50:
            cv_folds = min(5, len(set(intents)))
            if cv_folds >= 2:
                scores = cross_val_score(self._model, x_data, intents, cv=cv_folds)
                accuracy = float(scores.mean())

        # 클래스 분포
        from collections import Counter
        dist = dict(Counter(intents))

        logger.info(
            "Intent 분류기 학습 완료: %d건, accuracy=%.2f, classes=%s",
            len(questions), accuracy, dist,
        )

        return {
            "n_samples": len(questions),
            "accuracy": accuracy,
            "class_distribution": dist,
        }

    def predict(self, question: str) -> str:
        """질문의 intent 예측."""
        if not self._is_trained:
            return "general"

        x_data = self._vectorizer.transform([question])
        return self._model.predict(x_data)[0]

    def predict_with_confidence(self, question: str) -> tuple[str, float]:
        """intent 예측 + decision function 기반 신뢰도."""
        if not self._is_trained:
            return "general", 0.0

        x_data = self._vectorizer.transform([question])
        intent = self._model.predict(x_data)[0]

        # decision_function으로 신뢰도 추정
        decisions = self._model.decision_function(x_data)[0]
        if hasattr(decisions, '__len__'):
            # 다클래스: softmax-like normalization
            import numpy as np
            exp_d = np.exp(decisions - decisions.max())
            confidence = float(exp_d.max() / exp_d.sum())
        else:
            confidence = min(abs(float(decisions)), 1.0)

        return intent, confidence

    def save(self, model_dir: Path = MODEL_DIR) -> None:
        """모델 저장."""
        import joblib

        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._vectorizer, model_dir / "vectorizer.pkl")
        joblib.dump(self._model, model_dir / "model.pkl")
        logger.info("Intent 분류기 저장: %s", model_dir)

    def load(self, model_dir: Path = MODEL_DIR) -> bool:
        """모델 로드. 성공 시 True."""
        vec_path = model_dir / "vectorizer.pkl"
        model_path = model_dir / "model.pkl"

        if not vec_path.exists() or not model_path.exists():
            return False

        import joblib

        self._vectorizer = joblib.load(vec_path)
        self._model = joblib.load(model_path)
        self._is_trained = True
        logger.info("Intent 분류기 로드: %s", model_dir)
        return True


async def bootstrap_from_llm(
    session: AsyncSession,
    limit: int = 500,
) -> list[tuple[str, str]]:
    """qa_logs의 질문을 LLM으로 일괄 라벨링하여 학습 데이터 생성.

    Returns:
        [(question, intent), ...] 리스트
    """
    from api.llm.ollama_client import invoke_with_fallback
    from api.llm.prompts import QUERY_ANALYSIS_PROMPT

    stmt = (
        select(QALog.question)
        .order_by(QALog.created_at.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    questions = [row[0] for row in result.all()]

    labeled: list[tuple[str, str]] = []

    for question in questions:
        try:
            prompt = QUERY_ANALYSIS_PROMPT.format(question=question)
            raw = (await invoke_with_fallback(prompt)).strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            analysis = json.loads(raw)
            intent = analysis.get("intent", "general")
            if intent in INTENT_CLASSES:
                labeled.append((question, intent))
        except Exception:
            continue

    logger.info("LLM 부트스트래핑: %d/%d건 라벨링 성공", len(labeled), len(questions))
    return labeled
