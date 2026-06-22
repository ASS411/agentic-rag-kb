"""Tests for the RAGAS evaluation service and API.

Covers:
- EvaluationService initialization
- Metric descriptions
- EvaluationReport model
- EvaluationRequest/Response models
- API endpoint validation
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from app.models.evaluation import (
    EvaluationReport,
    EvaluationSample,
    EvaluationRequest,
    EvaluationResponse,
    EvaluationDataset,
    EvaluationHistoryItem,
    MetricScore,
    EvalTestsetRequest,
)
from app.core.evaluator import (
    EvaluationService,
    METRIC_DESCRIPTIONS,
    quick_evaluate,
)


# ---------------------------------------------------------------------------
# EvaluationService tests
# ---------------------------------------------------------------------------


class TestEvaluationServiceInit:
    """Test EvaluationService initialization."""

    def test_init_with_defaults(self):
        """Service initializes with lazy-loaded components."""
        service = EvaluationService()
        assert service._llm_client is None
        assert service._embedder is None
        assert service._retriever is None

    def test_init_with_components(self):
        """Service accepts custom components."""
        mock_llm = MagicMock()
        mock_embedder = MagicMock()
        mock_retriever = MagicMock()

        service = EvaluationService(
            llm_client=mock_llm,
            embedder=mock_embedder,
            retriever=mock_retriever,
        )

        assert service._llm_client is mock_llm
        assert service._embedder is mock_embedder
        assert service._retriever is mock_retriever


class TestMetricDescriptions:
    """Test metric descriptions."""

    def test_metric_descriptions_exist(self):
        """All expected metrics have descriptions."""
        expected_metrics = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "noise_sensitivity",
            "response_conciseness",
        ]
        for metric in expected_metrics:
            assert metric in METRIC_DESCRIPTIONS
            assert "name" in METRIC_DESCRIPTIONS[metric]
            assert "description" in METRIC_DESCRIPTIONS[metric]
            assert "higher_is_better" in METRIC_DESCRIPTIONS[metric]

    def test_get_available_metrics(self):
        """get_available_metrics returns all descriptions."""
        service = EvaluationService()
        metrics = service.get_available_metrics()
        assert len(metrics) == len(METRIC_DESCRIPTIONS)


class TestEvaluationReport:
    """Test EvaluationReport model."""

    def test_report_creation(self):
        """Report can be created with required fields."""
        report = EvaluationReport(
            id="test-123",
            created_at=datetime.utcnow(),
            total_samples=10,
        )
        assert report.id == "test-123"
        assert report.total_samples == 10
        assert report.faithfulness_avg == 0.0

    def test_overall_score_with_metrics(self):
        """overall_score calculates correctly."""
        report = EvaluationReport(
            id="test-456",
            total_samples=5,
            faithfulness_avg=0.8,
            answer_relevancy_avg=0.7,
            context_precision_avg=0.6,
            context_recall_avg=0.9,
        )
        expected = (0.8 + 0.7 + 0.6 + 0.9) / 4
        assert report.overall_score == expected

    def test_overall_score_empty_metrics(self):
        """overall_score returns 0.0 when no metrics."""
        report = EvaluationReport(id="test-789", total_samples=0)
        assert report.overall_score == 0.0

    def test_summary_property(self):
        """summary returns expected structure."""
        report = EvaluationReport(
            id="sum-test",
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            dataset_name="test_ds",
            total_samples=20,
            faithfulness_avg=0.85,
            answer_relevancy_avg=0.90,
            context_precision_avg=0.75,
            context_recall_avg=0.80,
        )
        summary = report.summary

        assert summary["id"] == "sum-test"
        assert summary["total_samples"] == 20
        assert "overall_score" in summary
        assert "metrics" in summary
        assert summary["metrics"]["faithfulness"] == 0.85


# ---------------------------------------------------------------------------
# EvaluationSample tests
# ---------------------------------------------------------------------------


class TestEvaluationSample:
    """Test EvaluationSample model."""

    def test_sample_creation(self):
        """Sample can be created with required fields."""
        sample = EvaluationSample(
            question="什么是RAG?",
            ground_truth="RAG是检索增强生成技术。",
        )
        assert sample.question == "什么是RAG?"
        assert sample.ground_truth == "RAG是检索增强生成技术。"
        assert sample.answer == ""
        assert sample.contexts == []
        assert sample.metrics == {}

    def test_sample_with_full_data(self):
        """Sample can be created with all fields."""
        sample = EvaluationSample(
            question="RAG的核心组件是什么?",
            ground_truth="RAG包含检索器和生成器两个组件。",
            answer="RAG的核心组件是检索器和生成器。",
            contexts=["检索器负责从知识库检索...", "生成器基于上下文生成..."],
            metrics={
                "faithfulness": 0.9,
                "answer_relevancy": 0.85,
            },
        )
        assert len(sample.contexts) == 2
        assert sample.metrics["faithfulness"] == 0.9


# ---------------------------------------------------------------------------
# EvaluationRequest tests
# ---------------------------------------------------------------------------


class TestEvaluationRequest:
    """Test EvaluationRequest model validation."""

    def test_valid_request(self):
        """Valid request passes validation."""
        request = EvaluationRequest(
            questions=["问题1", "问题2"],
            ground_truths=["答案1", "答案2"],
            ground_truth_contexts=[["上下文1"], ["上下文2"]],
        )
        assert len(request.questions) == 2
        assert request.use_parent_chunks is False

    def test_default_metrics(self):
        """Default metrics are set correctly."""
        request = EvaluationRequest(
            questions=["问题"],
            ground_truths=["答案"],
            ground_truth_contexts=[["上下文"]],
        )
        assert "faithfulness" in request.metrics
        assert "answer_relevancy" in request.metrics
        assert "context_precision" in request.metrics
        assert "context_recall" in request.metrics

    def test_custom_metrics(self):
        """Custom metrics can be specified."""
        request = EvaluationRequest(
            questions=["问题"],
            ground_truths=["答案"],
            ground_truth_contexts=[["上下文"]],
            metrics=["faithfulness", "noise_sensitivity"],
        )
        assert len(request.metrics) == 2
        assert "faithfulness" in request.metrics


# ---------------------------------------------------------------------------
# EvaluationDataset tests
# ---------------------------------------------------------------------------


class TestEvaluationDataset:
    """Test EvaluationDataset model."""

    def test_dataset_to_ragas_dict(self):
        """to_ragas_dict produces expected format."""
        dataset = EvaluationDataset(
            name="test",
            samples=[
                EvaluationSample(
                    question="Q1",
                    ground_truth="A1",
                    answer="Generated1",
                    contexts=["C1"],
                    ground_truth_contexts=["GT1"],
                ),
                EvaluationSample(
                    question="Q2",
                    ground_truth="A2",
                    answer="Generated2",
                    contexts=["C2"],
                    ground_truth_contexts=["GT2"],
                ),
            ],
        )
        ragas_dict = dataset.to_ragas_dict()

        assert "user_input" in ragas_dict
        assert "reference" in ragas_dict
        assert ragas_dict["user_input"] == ["Q1", "Q2"]
        assert ragas_dict["reference"] == ["A1", "A2"]

    def test_dataset_from_ragas_dict(self):
        """from_ragas_dict creates correct structure."""
        data = {
            "user_input": ["Q1", "Q2"],
            "reference": ["A1", "A2"],
            "reference_contexts": [["GT1"], ["GT2"]],
            "response": ["G1", "G2"],
            "retrieved_contexts": [["C1"], ["C2"]],
        }
        dataset = EvaluationDataset.from_ragas_dict(data)

        assert len(dataset.samples) == 2
        assert dataset.samples[0].question == "Q1"
        assert dataset.samples[1].ground_truth == "A2"


# ---------------------------------------------------------------------------
# API Model tests
# ---------------------------------------------------------------------------


class TestEvaluationHistoryItem:
    """Test EvaluationHistoryItem model."""

    def test_history_item_creation(self):
        """History item can be created."""
        item = EvaluationHistoryItem(
            id="hist-123",
            created_at=datetime.utcnow(),
            dataset_name="eval_ds",
            total_samples=50,
            overall_score=0.85,
        )
        assert item.id == "hist-123"
        assert item.overall_score == 0.85


class TestMetricScore:
    """Test MetricScore model."""

    def test_metric_score_creation(self):
        """Metric score can be created."""
        score = MetricScore(
            name="faithfulness",
            score=0.92,
            threshold=0.8,
        )
        assert score.name == "faithfulness"
        assert score.score == 0.92
        assert score.passed is True  # 0.92 > 0.8

    def test_metric_score_not_passed(self):
        """Score correctly indicates failure."""
        score = MetricScore(
            name="context_precision",
            score=0.6,
            threshold=0.8,
        )
        assert score.passed is False


# ---------------------------------------------------------------------------
# RAGAS availability tests
# ---------------------------------------------------------------------------


class TestRagasAvailability:
    """Test RAGAS availability detection."""

    def test_check_ragas_availability_true(self):
        """Returns True when ragas is importable."""
        with patch("app.core.evaluator.EvaluationService._check_ragas_availability") as mock:
            mock.return_value = True
            service = EvaluationService()
            assert service._ragas_available is True

    def test_check_ragas_availability_false(self):
        """Returns False when ragas is not importable."""
        with patch("importlib.import_module", side_effect=ImportError):
            service = EvaluationService()
            # Should log warning but not raise
            assert service._ragas_available is False


# ---------------------------------------------------------------------------
# Integration test helpers
# ---------------------------------------------------------------------------


def _make_mock_sample(
    question: str = "什么是RAG?",
    ground_truth: str = "RAG是检索增强生成。",
    answer: str = "RAG是一种检索增强生成技术。",
    contexts: list[str] | None = None,
) -> EvaluationSample:
    """Create a mock evaluation sample for testing."""
    return EvaluationSample(
        question=question,
        ground_truth=ground_truth,
        answer=answer,
        contexts=contexts or ["RAG是一种结合检索和生成的技术..."],
        metrics={"faithfulness": 0.9, "answer_relevancy": 0.85},
    )


def _make_mock_report(
    num_samples: int = 5,
    overall_score: float = 0.85,
) -> EvaluationReport:
    """Create a mock evaluation report for testing."""
    samples = [_make_mock_sample() for _ in range(num_samples)]
    return EvaluationReport(
        id="mock-report-123",
        created_at=datetime.utcnow(),
        dataset_name="test_dataset",
        total_samples=num_samples,
        faithfulness_avg=overall_score,
        answer_relevancy_avg=overall_score,
        context_precision_avg=overall_score,
        context_recall_avg=overall_score,
        samples=samples,
    )
