"""api.evaluation.calibration - 신뢰도 캘리브레이션 (Platt scaling)."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import QALog

logger = logging.getLogger(__name__)

CALIBRATION_MODEL_PATH = Path(__file__).parent / "calibration_model.json"


class ConfidenceCalibrator:
    """Platt scaling으로 raw confidence → calibrated confidence 변환.

    Logistic regression: P(satisfied) = 1 / (1 + exp(-(w * x + b)))
    여기서 satisfied = feedback_score >= 4.
    """

    def __init__(self) -> None:
        self.weight: float = 1.0
        self.bias: float = 0.0
        self._fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        raw_confidences: list[float],
        feedback_scores: list[int],
        learning_rate: float = 0.1,
        epochs: int = 100,
    ) -> dict[str, float]:
        """qa_logs 데이터로 캘리브레이션 모델 학습.

        Args:
            raw_confidences: validator 출력 confidence 리스트.
            feedback_scores: 사용자 피드백 점수 (1-5) 리스트.

        Returns:
            학습 결과 (weight, bias, loss).
        """
        if len(raw_confidences) < 10:
            logger.warning("캘리브레이션 데이터 부족: %d건", len(raw_confidences))
            return {"error": "최소 10건의 데이터 필요"}

        # feedback_score >= 4 → positive (1), < 4 → negative (0)
        labels = [1.0 if s >= 4 else 0.0 for s in feedback_scores]

        w, b = 1.0, 0.0

        # Gradient descent for logistic regression
        for _ in range(epochs):
            dw, db = 0.0, 0.0
            for x, y in zip(raw_confidences, labels):
                z = w * x + b
                pred = _sigmoid(z)
                err = pred - y
                dw += err * x
                db += err

            n = len(raw_confidences)
            w -= learning_rate * dw / n
            b -= learning_rate * db / n

        self.weight = w
        self.bias = b
        self._fitted = True

        # 로그 손실 계산
        loss = 0.0
        for x, y in zip(raw_confidences, labels):
            pred = max(min(_sigmoid(w * x + b), 1 - 1e-7), 1e-7)
            loss -= y * math.log(pred) + (1 - y) * math.log(1 - pred)
        loss /= len(raw_confidences)

        logger.info("캘리브레이션 학습 완료: w=%.4f, b=%.4f, loss=%.4f", w, b, loss)
        return {"weight": w, "bias": b, "log_loss": loss, "n_samples": len(raw_confidences)}

    def calibrate(self, raw_confidence: float) -> float:
        """보정된 신뢰도 반환."""
        if not self._fitted:
            return raw_confidence
        return _sigmoid(self.weight * raw_confidence + self.bias)

    def save(self, path: Path = CALIBRATION_MODEL_PATH) -> None:
        """모델 파라미터 저장."""
        data = {"weight": self.weight, "bias": self.bias, "fitted": self._fitted}
        with open(path, "w") as f:
            json.dump(data, f)
        logger.info("캘리브레이션 모델 저장: %s", path)

    def load(self, path: Path = CALIBRATION_MODEL_PATH) -> bool:
        """모델 파라미터 로드. 성공 시 True."""
        if not path.exists():
            return False
        with open(path) as f:
            data = json.load(f)
        self.weight = data["weight"]
        self.bias = data["bias"]
        self._fitted = data.get("fitted", True)
        logger.info("캘리브레이션 모델 로드: w=%.4f, b=%.4f", self.weight, self.bias)
        return True


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


async def fit_from_db(session: AsyncSession) -> dict[str, float]:
    """DB의 qa_logs에서 캘리브레이션 학습 데이터를 수집하고 모델 학습."""
    stmt = (
        select(QALog.confidence, QALog.feedback_score)
        .where(QALog.confidence.is_not(None))
        .where(QALog.feedback_score.is_not(None))
    )
    result = await session.execute(stmt)
    rows = result.all()

    if not rows:
        return {"error": "캘리브레이션 데이터 없음"}

    confidences = [float(r[0]) for r in rows]
    scores = [int(r[1]) for r in rows]

    calibrator = ConfidenceCalibrator()
    fit_result = calibrator.fit(confidences, scores)

    if "error" not in fit_result:
        calibrator.save()

    return fit_result
