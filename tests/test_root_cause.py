"""Tests for root cause analysis module."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.analysis.root_cause import RootCauseAnalyzer


@pytest.fixture()
def analyzer() -> RootCauseAnalyzer:
    return RootCauseAnalyzer(sensor_names=["temp_00", "vib_00", "pres_00"])


@pytest.fixture()
def sample_contributions() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    return {
        "temp_00": rng.uniform(0, 1, 100),
        "vib_00": rng.uniform(0, 0.5, 100),
        "pres_00": rng.uniform(0, 0.3, 100),
    }


@pytest.fixture()
def sample_scores() -> np.ndarray:
    scores = np.zeros(100)
    scores[10] = 0.9
    scores[50] = 0.7
    scores[90] = 0.3
    return scores


@pytest.fixture()
def sample_timestamps() -> np.ndarray:
    base = datetime(2024, 1, 1)
    return np.array([base + timedelta(hours=i) for i in range(100)])


class TestAnalyze:
    def test_returns_results_only_for_anomalous_points(
        self,
        analyzer: RootCauseAnalyzer,
        sample_contributions: dict[str, np.ndarray],
        sample_scores: np.ndarray,
        sample_timestamps: np.ndarray,
    ) -> None:
        results = analyzer.analyze(sample_contributions, sample_scores, sample_timestamps)
        # Only scores > 0.5 should be returned
        assert len(results) == 2  # index 10 (0.9) and 50 (0.7)

    def test_top_contributors_sorted_descending(
        self,
        analyzer: RootCauseAnalyzer,
        sample_contributions: dict[str, np.ndarray],
        sample_scores: np.ndarray,
        sample_timestamps: np.ndarray,
    ) -> None:
        results = analyzer.analyze(sample_contributions, sample_scores, sample_timestamps)
        for result in results:
            scores_list = [s for _, s in result.top_contributors]
            assert scores_list == sorted(scores_list, reverse=True)

    def test_top_k_limits_contributors(
        self,
        analyzer: RootCauseAnalyzer,
        sample_contributions: dict[str, np.ndarray],
        sample_scores: np.ndarray,
        sample_timestamps: np.ndarray,
    ) -> None:
        results = analyzer.analyze(sample_contributions, sample_scores, sample_timestamps, top_k=2)
        for result in results:
            assert len(result.top_contributors) <= 2

    def test_explanation_not_empty(
        self,
        analyzer: RootCauseAnalyzer,
        sample_contributions: dict[str, np.ndarray],
        sample_scores: np.ndarray,
        sample_timestamps: np.ndarray,
    ) -> None:
        results = analyzer.analyze(sample_contributions, sample_scores, sample_timestamps)
        for result in results:
            assert len(result.explanation) > 0

    def test_handles_none_timestamps(
        self,
        analyzer: RootCauseAnalyzer,
        sample_contributions: dict[str, np.ndarray],
        sample_scores: np.ndarray,
    ) -> None:
        results = analyzer.analyze(sample_contributions, sample_scores, None)
        assert len(results) == 2
        assert results[0].timestamp == 10  # Index used as fallback


class TestExplanation:
    def test_critical_severity(self, analyzer: RootCauseAnalyzer) -> None:
        explanation = analyzer._generate_explanation([("temp_00", 0.9)], 0.9)
        assert "Critical" in explanation

    def test_moderate_severity(self, analyzer: RootCauseAnalyzer) -> None:
        explanation = analyzer._generate_explanation([("temp_00", 0.6)], 0.6)
        assert "Moderate" in explanation

    def test_empty_contributors(self, analyzer: RootCauseAnalyzer) -> None:
        explanation = analyzer._generate_explanation([], 0.9)
        assert "No significant" in explanation


class TestSensorAnomalyRates:
    def test_rates_between_0_and_1(
        self,
        analyzer: RootCauseAnalyzer,
        sample_contributions: dict[str, np.ndarray],
    ) -> None:
        rates = analyzer.get_sensor_anomaly_rates(sample_contributions, threshold=0.5)
        for rate in rates.values():
            assert 0 <= rate <= 1

    def test_returns_all_sensors(
        self,
        analyzer: RootCauseAnalyzer,
        sample_contributions: dict[str, np.ndarray],
    ) -> None:
        rates = analyzer.get_sensor_anomaly_rates(sample_contributions)
        assert set(rates.keys()) == set(sample_contributions.keys())


class TestTemporalPatterns:
    def test_patterns_by_hour(
        self,
        analyzer: RootCauseAnalyzer,
        sample_contributions: dict[str, np.ndarray],
        sample_timestamps: np.ndarray,
    ) -> None:
        patterns = analyzer.get_temporal_patterns(sample_contributions, sample_timestamps)
        assert set(patterns.keys()) == set(sample_contributions.keys())
        for hourly in patterns.values():
            for hour, score in hourly.items():
                assert 0 <= hour < 24
                assert isinstance(score, float)
