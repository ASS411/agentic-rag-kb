"""Evaluation data models for RAGAS integration.

Provides Pydantic models for:
- Evaluation datasets and samples
- Evaluation reports and results
- API request/response schemas
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Evaluation Dataset Models
# ---------------------------------------------------------------------------


class EvaluationSample(BaseModel):
    """Single evaluation sample with ground truth and metrics."""

    question: str = Field(description="User question / query")
    ground_truth: str = Field(description="Reference answer / ground truth")
    answer: str = Field(default="", description="LLM generated answer")
    contexts: list[str] = Field(
        default_factory=list,
        description="Retrieved context chunks",
    )
    ground_truth_contexts: list[str] = Field(
        default_factory=list,
        description="Reference contexts for ground truth answer",
    )
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Metric scores for this sample",
    )


class EvaluationDataset(BaseModel):
    """Collection of evaluation samples."""

    name: str = Field(default="manual_dataset")
    description: str = Field(default="")
    samples: list[EvaluationSample] = Field(default_factory=list)

    def to_ragas_dict(self) -> dict[str, list]:
        """Convert to dict format expected by RAGAS."""
        return {
            "user_input": [s.question for s in self.samples],
            "reference": [s.ground_truth for s in self.samples],
            "reference_contexts": [
                s.ground_truth_contexts for s in self.samples
            ],
            "response": [s.answer for s in self.samples],
            "retrieved_contexts": [s.contexts for s in self.samples],
        }

    @classmethod
    def from_ragas_dict(cls, data: dict, name: str = "imported") -> EvaluationDataset:
        """Create from RAGAS output dict."""
        samples = []
        for i in range(len(data.get("user_input", []))):
            samples.append(
                EvaluationSample(
                    question=data["user_input"][i],
                    ground_truth=data["reference"][i],
                    ground_truth_contexts=data.get("reference_contexts", [[]])[i],
                    answer=data.get("response", [""])[i],
                    contexts=data.get("retrieved_contexts", [[]])[i],
                )
            )
        return cls(name=name, samples=samples)


# ---------------------------------------------------------------------------
# Evaluation Report Models
# ---------------------------------------------------------------------------


class MetricScore(BaseModel):
    """Individual metric score with metadata."""

    name: str = Field(description="Metric name (e.g., faithfulness)")
    score: float = Field(description="Score value (0.0 - 1.0)")
    threshold: Optional[float] = Field(
        default=None,
        description="Pass threshold for this metric",
    )

    @property
    def passed(self) -> bool | None:
        """Compute whether score meets threshold."""
        if self.threshold is None:
            return None
        return self.score >= self.threshold


class EvaluationReport(BaseModel):
    """Complete evaluation report with aggregated metrics and samples."""

    id: str = Field(description="Unique report identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    dataset_name: str = Field(default="")
    total_samples: int = Field(default=0)

    # Aggregated metric scores
    faithfulness_avg: float = Field(default=0.0, description="Faithfulness mean score")
    answer_relevancy_avg: float = Field(default=0.0, description="Answer relevancy mean score")
    context_precision_avg: float = Field(default=0.0, description="Context precision mean score")
    context_recall_avg: float = Field(default=0.0, description="Context recall mean score")
    noise_sensitivity_avg: float = Field(default=0.0, description="Noise sensitivity mean score")
    response_conciseness_avg: float = Field(default=0.0, description="Response conciseness mean score")

    # Individual sample results
    samples: list[EvaluationSample] = Field(default_factory=list)

    # Configuration used for this evaluation
    retrieval_config: dict[str, Any] = Field(default_factory=dict)
    generation_config: dict[str, Any] = Field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        """Calculate overall score as average of all metrics."""
        scores = [
            self.faithfulness_avg,
            self.answer_relevancy_avg,
            self.context_precision_avg,
            self.context_recall_avg,
        ]
        valid_scores = [s for s in scores if s > 0]
        return sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    @property
    def summary(self) -> dict:
        """Get a summary dict for quick overview."""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "total_samples": self.total_samples,
            "overall_score": round(self.overall_score, 4),
            "metrics": {
                "faithfulness": round(self.faithfulness_avg, 4),
                "answer_relevancy": round(self.answer_relevancy_avg, 4),
                "context_precision": round(self.context_precision_avg, 4),
                "context_recall": round(self.context_recall_avg, 4),
            },
        }


# ---------------------------------------------------------------------------
# API Request / Response Models
# ---------------------------------------------------------------------------


class EvaluationRequest(BaseModel):
    """Request to run evaluation on specified questions."""

    questions: list[str] = Field(
        min_length=1,
        description="List of questions to evaluate",
    )
    ground_truths: list[str] = Field(
        description="Ground truth answers corresponding to questions",
    )
    ground_truth_contexts: list[list[str]] = Field(
        description="Reference contexts for each ground truth answer",
    )
    metrics: list[str] = Field(
        default=["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        description="Metrics to compute",
    )
    use_parent_chunks: bool = Field(
        default=False,
        description="Use parent-child chunk retrieval",
    )
    top_k_recall: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of chunks to retrieve",
    )
    top_k_rerank: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks after reranking",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "questions": ["RAG是什么意思？", "RAG系统有哪些组件？"],
                    "ground_truths": [
                        "RAG是检索增强生成，是一种结合信息检索与文本生成的AI技术。",
                        "RAG系统包括检索器（Retriever）和生成器（Generator）两个组件。",
                    ],
                    "ground_truth_contexts": [
                        ["RAG（Retrieval-Augmented Generation）是一种..."],
                        ["RAG系统通常包含检索器和生成器..."],
                    ],
                    "metrics": ["faithfulness", "answer_relevancy"],
                    "use_parent_chunks": True,
                }
            ]
        }
    }


class EvaluationResponse(BaseModel):
    """Response containing evaluation results."""

    report: EvaluationReport
    duration_seconds: float = Field(description="Evaluation duration")


class EvalTestsetRequest(BaseModel):
    """Request to generate test set from documents."""

    document_ids: list[str] = Field(
        min_length=1,
        description="Document IDs to generate test set from",
    )
    num_samples: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Number of question-answer pairs to generate",
    )
    question_types: list[str] = Field(
        default=["simple", "reasoning"],
        description="Types of questions to generate",
    )


class EvalTestsetResponse(BaseModel):
    """Response containing generated test set."""

    testset: list[dict] = Field(description="Generated question-answer pairs")
    total: int = Field(description="Total number of samples")


class EvaluationHistoryItem(BaseModel):
    """Summary item for evaluation history list."""

    id: str
    created_at: datetime
    dataset_name: str
    total_samples: int
    overall_score: float

    model_config = {"from_attributes": True}
