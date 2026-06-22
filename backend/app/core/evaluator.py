"""RAGAS evaluation service for RAG system assessment.

Provides comprehensive evaluation capabilities using the RAGAS framework,
supporting metrics like faithfulness, answer relevancy, context precision,
and context recall.

Usage::

    evaluator = EvaluationService()
    report = await evaluator.run_evaluation(
        questions=["什么是RAG?"],
        ground_truths=["RAG是检索增强生成..."],
        ground_truth_contexts=[["上下文..."]],
    )
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from app.config import settings
from app.models.evaluation import (
    EvaluationReport,
    EvaluationSample,
)

if TYPE_CHECKING:
    from app.core.llm import LLMClient
    from app.core.embedder import Embedder
    from app.core.retriever import Retriever


# ---------------------------------------------------------------------------
# Metric descriptions
# ---------------------------------------------------------------------------

METRIC_DESCRIPTIONS = {
    "faithfulness": {
        "name": "忠实度 (Faithfulness)",
        "description": "衡量答案是否仅基于提供的上下文生成，检测幻觉",
        "higher_is_better": True,
    },
    "answer_relevancy": {
        "name": "答案相关性 (Answer Relevancy)",
        "description": "衡量答案与问题的语义相关程度",
        "higher_is_better": True,
    },
    "context_precision": {
        "name": "上下文精确率 (Context Precision)",
        "description": "衡量检索到的上下文中有多少是相关的",
        "higher_is_better": True,
    },
    "context_recall": {
        "name": "上下文召回率 (Context Recall)",
        "description": "衡量参考上下文中有多少被检索到",
        "higher_is_better": True,
    },
    "noise_sensitivity": {
        "name": "噪声敏感度 (Noise Sensitivity)",
        "description": "衡量系统对噪声上下文的鲁棒性（越低越好）",
        "higher_is_better": False,
    },
    "response_conciseness": {
        "name": "响应简洁性 (Response Conciseness)",
        "description": "衡量答案是否简洁不冗余",
        "higher_is_better": True,
    },
}


# ---------------------------------------------------------------------------
# EvaluationService
# ---------------------------------------------------------------------------


class EvaluationService:
    """RAGAS-based evaluation service for RAG systems.

    Executes the full RAG pipeline (retrieve → generate) for each question
    and computes RAGAS metrics comparing the generated answer against
    ground truth.

    Parameters
    ----------
    llm_client:
        LLM client for answer generation. Created lazily if not provided.
    embedder:
        Embedder for retrieval. Created lazily if not provided.
    retriever:
        Retriever instance. Created lazily if not provided.
    """

    def __init__(
        self,
        llm_client: "LLMClient | None" = None,
        embedder: "Embedder | None" = None,
        retriever: "Retriever | None" = None,
    ) -> None:
        self._llm_client = llm_client
        self._embedder = embedder
        self._retriever = retriever
        self._ragas_available = self._check_ragas_availability()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_evaluation(
        self,
        questions: list[str],
        ground_truths: list[str],
        ground_truth_contexts: list[list[str]],
        metrics: list[str] | None = None,
        use_parent_chunks: bool = False,
        top_k_recall: int = 10,
        top_k_rerank: int = 5,
        dataset_name: str = "manual_evaluation",
    ) -> EvaluationReport:
        """Run complete RAGAS evaluation on provided questions.

        For each question:
        1. Retrieve relevant chunks using the RAG retriever
        2. Generate answer using the LLM with retrieved context
        3. Compute RAGAS metrics comparing answer against ground truth

        Parameters
        ----------
        questions:
            List of user questions to evaluate.
        ground_truths:
            Reference/ground truth answers for each question.
        ground_truth_contexts:
            Reference contexts that ground truth answers are based on.
        metrics:
            List of metric names to compute.
            Options: faithfulness, answer_relevancy, context_precision,
            context_recall, noise_sensitivity, response_conciseness.
            Defaults to all four core metrics.
        use_parent_chunks:
            Use parent-child chunk retrieval (search child, return parent).
        top_k_recall:
            Number of chunks to retrieve per query.
        top_k_rerank:
            Number of chunks after reranking.
        dataset_name:
            Name identifier for this evaluation dataset.

        Returns
        -------
        EvaluationReport
            Complete report with aggregated metrics and per-sample results.
        """
        start_time = time.time()

        if len(questions) != len(ground_truths):
            raise ValueError(
                f"Questions ({len(questions)}) and ground truths ({len(ground_truths)}) "
                "must have the same length"
            )

        metrics = metrics or ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

        logger.info(
            "Starting evaluation: {} questions, {} metrics, parent_chunks={}",
            len(questions),
            len(metrics),
            use_parent_chunks,
        )

        # Step 1: Execute RAG pipeline for each question
        samples = await self._execute_rag_pipeline(
            questions=questions,
            ground_truths=ground_truths,
            ground_truth_contexts=ground_truth_contexts,
            use_parent_chunks=use_parent_chunks,
            top_k_recall=top_k_recall,
            top_k_rerank=top_k_rerank,
        )

        # Step 2: Compute RAGAS metrics
        if self._ragas_available and samples:
            samples = await self._compute_ragas_metrics(
                samples=samples,
                metrics=metrics,
            )

        # Step 3: Build report
        report = self._build_report(
            samples=samples,
            metrics=metrics,
            dataset_name=dataset_name,
            retrieval_config={
                "use_parent_chunks": use_parent_chunks,
                "top_k_recall": top_k_recall,
                "top_k_rerank": top_k_rerank,
            },
        )

        duration = time.time() - start_time
        logger.info(
            "Evaluation complete: {} samples in {:.2f}s, overall_score={:.4f}",
            len(samples),
            duration,
            report.overall_score,
        )

        return report

    def get_available_metrics(self) -> dict[str, dict]:
        """Return available metrics with descriptions."""
        return METRIC_DESCRIPTIONS.copy()

    # ------------------------------------------------------------------
    # Internal: RAG pipeline execution
    # ------------------------------------------------------------------

    async def _execute_rag_pipeline(
        self,
        questions: list[str],
        ground_truths: list[str],
        ground_truth_contexts: list[list[str]],
        use_parent_chunks: bool,
        top_k_recall: int,
        top_k_rerank: int,
    ) -> list[EvaluationSample]:
        """Execute RAG pipeline: retrieve → generate for each question."""
        from app.core.prompts import RAGPromptBuilder

        retriever = self._get_retriever()
        llm = self._get_llm_client()
        prompt_builder = RAGPromptBuilder()

        samples: list[EvaluationSample] = []

        for i, question in enumerate(questions):
            logger.debug("Processing question {}/{}", i + 1, len(questions))

            # Retrieve context
            try:
                if use_parent_chunks:
                    result = await retriever.retrieve_with_parent_lookup(
                        queries=[question],
                        top_k_recall=top_k_recall,
                        top_k_rerank=top_k_rerank,
                        rerank=True,
                    )
                else:
                    result = await retriever.retrieve(
                        queries=[question],
                        top_k_recall=top_k_recall,
                        top_k_rerank=top_k_rerank,
                        rerank=True,
                    )
            except Exception as exc:
                logger.warning("Retrieval failed for question {}: {}", i + 1, exc)
                result = None

            contexts = [chunk.content for chunk in result.chunks] if result else []

            # Generate answer
            answer = ""
            if contexts:
                try:
                    messages = prompt_builder.build(
                        query=question,
                        chunks=result.chunks,
                    )
                    answer = await llm.generate(messages)
                except Exception as exc:
                    logger.warning("Generation failed for question {}: {}", i + 1, exc)

            samples.append(
                EvaluationSample(
                    question=question,
                    ground_truth=ground_truths[i],
                    answer=answer,
                    contexts=contexts,
                    ground_truth_contexts=ground_truth_contexts[i],
                )
            )

        return samples

    # ------------------------------------------------------------------
    # Internal: RAGAS metrics computation
    # ------------------------------------------------------------------

    async def _compute_ragas_metrics(
        self,
        samples: list[EvaluationSample],
        metrics: list[str],
    ) -> list[EvaluationSample]:
        """Compute RAGAS metrics for evaluation samples."""
        try:
            from datasets import Dataset

            # Prepare data for RAGAS
            data = {
                "user_input": [s.question for s in samples],
                "reference": [s.ground_truth for s in samples],
                "reference_contexts": [s.ground_truth_contexts for s in samples],
                "response": [s.answer for s in samples],
                "retrieved_contexts": [s.contexts for s in samples],
            }

            dataset = Dataset.from_dict(data)

            # Select metrics
            selected_metrics = self._select_ragas_metrics(metrics)

            if not selected_metrics:
                logger.warning("No valid RAGAS metrics selected")
                return samples

            # Run evaluation
            from ragas import evaluate

            result = evaluate(dataset, metrics=selected_metrics)
            scores = result.scores

            # Update samples with computed metrics
            for i, sample in enumerate(samples):
                sample_metrics = {}
                for metric_name in metrics:
                    score_key = f"{metric_name}"
                    if score_key in scores and i < len(scores[score_key]):
                        sample_metrics[metric_name] = float(scores[score_key][i])
                    elif metric_name in scores and isinstance(scores[metric_name], list):
                        # Handle flat array format
                        if i < len(scores[metric_name]):
                            sample_metrics[metric_name] = float(scores[metric_name][i])
                sample.metrics = sample_metrics

            return samples

        except ImportError as exc:
            logger.warning("RAGAS not available: {}", exc)
            return samples
        except Exception as exc:
            logger.error("RAGAS evaluation failed: {}", exc)
            return samples

    def _select_ragas_metrics(self, metric_names: list[str]) -> list[Any]:
        """Map metric names to RAGAS metric objects."""
        try:
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                noise_sensitivity,
                response_conciseness,
            )

            metric_map = {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
                "context_recall": context_recall,
                "noise_sensitivity": noise_sensitivity,
                "response_conciseness": response_conciseness,
            }

            selected = []
            for name in metric_names:
                if name in metric_map:
                    selected.append(metric_map[name])

            return selected

        except ImportError:
            logger.warning("RAGAS metrics not importable")
            return []

    # ------------------------------------------------------------------
    # Internal: Report building
    # ------------------------------------------------------------------

    def _build_report(
        self,
        samples: list[EvaluationSample],
        metrics: list[str],
        dataset_name: str,
        retrieval_config: dict[str, Any],
    ) -> EvaluationReport:
        """Build evaluation report from samples."""
        report = EvaluationReport(
            id=str(uuid.uuid4()),
            created_at=datetime.utcnow(),
            dataset_name=dataset_name,
            total_samples=len(samples),
            samples=samples,
            retrieval_config=retrieval_config,
        )

        # Calculate metric averages
        for metric_name in metrics:
            scores = [s.metrics.get(metric_name, 0.0) for s in samples if s.metrics]
            if scores:
                avg = sum(scores) / len(scores)
                attr_name = f"{metric_name}_avg"
                if hasattr(report, attr_name):
                    setattr(report, attr_name, avg)

        return report

    # ------------------------------------------------------------------
    # Internal: Lazy initialization
    # ------------------------------------------------------------------

    def _get_llm_client(self) -> "LLMClient":
        """Get or create LLM client."""
        if self._llm_client is None:
            from app.core.llm import LLMClient
            self._llm_client = LLMClient()
        return self._llm_client

    def _get_embedder(self) -> "Embedder":
        """Get or create embedder."""
        if self._embedder is None:
            from app.core.embedder import Embedder
            self._embedder = Embedder()
        return self._embedder

    def _get_retriever(self) -> "Retriever":
        """Get or create retriever."""
        if self._retriever is None:
            from app.core.retriever import Retriever
            self._retriever = Retriever(
                embedder=self._get_embedder(),
            )
        return self._retriever

    @staticmethod
    def _check_ragas_availability() -> bool:
        """Check if RAGAS is installed and available."""
        try:
            import ragas
            return True
        except ImportError:
            logger.warning(
                "RAGAS not installed. Run: pip install ragas>=0.1.0 datasets>=2.14.0"
            )
            return False


# ---------------------------------------------------------------------------
# Convenience function for quick evaluation
# ---------------------------------------------------------------------------


async def quick_evaluate(
    questions: list[str],
    ground_truths: list[str],
    ground_truth_contexts: list[list[str]],
) -> EvaluationReport:
    """Run quick evaluation with default settings.

    This is a convenience function that creates an EvaluationService
    and runs evaluation with sensible defaults.

    Parameters
    ----------
    questions:
        Questions to evaluate.
    ground_truths:
        Ground truth answers.
    ground_truth_contexts:
        Ground truth contexts.

    Returns
    -------
    EvaluationReport
        Evaluation results.
    """
    service = EvaluationService()
    return await service.run_evaluation(
        questions=questions,
        ground_truths=ground_truths,
        ground_truth_contexts=ground_truth_contexts,
    )
