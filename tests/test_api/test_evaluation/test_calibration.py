"""tests.test_api.test_evaluation.test_calibration - 신뢰도 캘리브레이션 테스트."""

from api.evaluation.calibration import ConfidenceCalibrator, _sigmoid


class TestSigmoid:
    """Sigmoid 함수 테스트."""

    def test_zero(self):
        assert abs(_sigmoid(0.0) - 0.5) < 1e-6

    def test_large_positive(self):
        assert _sigmoid(10.0) > 0.99

    def test_large_negative(self):
        assert _sigmoid(-10.0) < 0.01

    def test_range(self):
        for x in [-5, -1, 0, 1, 5]:
            result = _sigmoid(x)
            assert 0.0 <= result <= 1.0


class TestConfidenceCalibrator:
    """ConfidenceCalibrator 테스트."""

    def test_not_fitted_returns_raw(self):
        cal = ConfidenceCalibrator()
        assert not cal.is_fitted
        assert cal.calibrate(0.7) == 0.7

    def test_fit_basic(self):
        cal = ConfidenceCalibrator()
        # 높은 confidence → 높은 만족도, 낮은 confidence → 낮은 만족도
        confidences = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
        feedback = [5, 5, 4, 4, 3, 2, 2, 1, 1, 1]
        result = cal.fit(confidences, feedback)
        assert "error" not in result
        assert cal.is_fitted
        assert result["n_samples"] == 10

    def test_fit_insufficient_data(self):
        cal = ConfidenceCalibrator()
        result = cal.fit([0.5], [3])
        assert "error" in result

    def test_calibrate_after_fit(self):
        cal = ConfidenceCalibrator()
        confidences = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
        feedback = [5, 5, 4, 4, 3, 2, 2, 1, 1, 1]
        cal.fit(confidences, feedback)

        # 보정된 결과는 0-1 범위
        result = cal.calibrate(0.8)
        assert 0.0 <= result <= 1.0

    def test_save_load(self, tmp_path):
        cal = ConfidenceCalibrator()
        confidences = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
        feedback = [5, 5, 4, 4, 3, 2, 2, 1, 1, 1]
        cal.fit(confidences, feedback)

        path = tmp_path / "model.json"
        cal.save(path)

        cal2 = ConfidenceCalibrator()
        assert cal2.load(path)
        assert cal2.is_fitted
        assert abs(cal.weight - cal2.weight) < 1e-6

    def test_load_nonexistent(self, tmp_path):
        cal = ConfidenceCalibrator()
        assert not cal.load(tmp_path / "nonexistent.json")
