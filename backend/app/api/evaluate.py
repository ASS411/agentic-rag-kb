"""Evaluation API endpoints for RAGAS-based RAG assessment.

Provides REST endpoints for:
- Running evaluations with RAGAS metrics
- Generating test sets from documents
- Retrieving evaluation history and reports

Endpoints::

    POST /api/v1/evaluate/run          - Run evaluation on questions
    GET  /api/v1/evaluate/metrics      - Get available metrics
    GET  /api/v1/evaluate/history      - Get evaluation history
    GET  /api/v1/evaluate/report/{id}  - Get specific report
"""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from loguru import logger

from app.core.evaluator import EvaluationService
from app.models.evaluation import (
    EvaluationHistoryItem,
    EvaluationRequest,
    EvaluationResponse,
    EvaluationReport,
    EvalTestsetRequest,
    EvalTestsetResponse,
)
from app.models.response import APIResponse

router = APIRouter(prefix="/evaluate", tags=["evaluation"])

# Lazy-initialized service
_evaluator_service: EvaluationService | None = None


def get_evaluator() -> EvaluationService:
    """Get or create the evaluation service singleton."""
    global _evaluator_service
    if _evaluator_service is None:
        _evaluator_service = EvaluationService()
    return _evaluator_service


# ---------------------------------------------------------------------------
# Evaluation endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/run",
    response_model=APIResponse[EvaluationReport],
    summary="Run RAGAS Evaluation",
    description="""
    Run comprehensive RAG evaluation using RAGAS metrics.

    **Request Body:**
    - `questions`: List of questions to evaluate
    - `ground_truths`: Reference/ground truth answers
    - `ground_truth_contexts`: Reference contexts for each ground truth
    - `metrics`: Metrics to compute (faithfulness, answer_relevancy, context_precision, context_recall)
    - `use_parent_chunks`: Use parent-child retrieval strategy
    - `top_k_recall`: Number of chunks to retrieve
    - `top_k_rerank`: Number of chunks after reranking

    **Returns:**
    - Complete evaluation report with per-metric scores and per-sample results
    """,
)
async def run_evaluation(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
) -> APIResponse[EvaluationReport]:
    """Run RAGAS evaluation on specified questions."""
    evaluator = get_evaluator()

    if not evaluator._ragas_available:
        logger.warning("RAGAS not available, running basic evaluation without metrics")

    try:
        start_time = time.time()

        report = await evaluator.run_evaluation(
            questions=request.questions,
            ground_truths=request.ground_truths,
            ground_truth_contexts=request.ground_truth_contexts,
            metrics=request.metrics,
            use_parent_chunks=request.use_parent_chunks,
            top_k_recall=request.top_k_recall,
            top_k_rerank=request.top_k_rerank,
        )

        duration = time.time() - start_time

        # Store report in background if persistence is enabled
        if _should_store_reports():
            background_tasks.add_task(_store_evaluation_report, report)

        return APIResponse.ok(
            data=report,
            message=f"Evaluation completed in {duration:.2f}s",
        )

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Evaluation failed: {}", exc)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}")


@router.get(
    "/metrics",
    response_model=APIResponse[dict],
    summary="Get Available Metrics",
    description="Returns list of available RAGAS metrics with descriptions.",
)
async def get_available_metrics() -> APIResponse[dict]:
    """Get available evaluation metrics."""
    evaluator = get_evaluator()
    metrics = evaluator.get_available_metrics()
    return APIResponse.ok(data=metrics)


# ---------------------------------------------------------------------------
# Evaluation history endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/history",
    response_model=APIResponse[list[EvaluationHistoryItem]],
    summary="Get Evaluation History",
    description="Returns list of recent evaluation reports.",
)
async def get_evaluation_history(
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> APIResponse[list[EvaluationHistoryItem]]:
    """Get evaluation history with pagination."""
    try:
        reports = _get_evaluation_history(limit=limit, offset=offset)
        return APIResponse.ok(data=reports)
    except Exception as exc:
        logger.error("Failed to get evaluation history: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/report/{report_id}",
    response_model=APIResponse[EvaluationReport],
    summary="Get Evaluation Report",
    description="Returns detailed report for a specific evaluation.",
)
async def get_evaluation_report(
    report_id: str,
) -> APIResponse[EvaluationReport]:
    """Get specific evaluation report by ID."""
    try:
        report = _get_evaluation_report_by_id(report_id)
        if report is None:
            raise HTTPException(status_code=404, detail="Report not found")
        return APIResponse.ok(data=report)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get report {}: {}", report_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Testset generation endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/testset/generate",
    response_model=APIResponse[EvalTestsetResponse],
    summary="Generate Test Set",
    description="""
    Generate question-answer test set from uploaded documents.

    Uses LLM to automatically generate diverse questions covering
    different aspects of the document content.

    **Note:** Requires RAGAS testset generator which may require
    additional dependencies.
    """,
)
async def generate_testset(
    request: EvalTestsetRequest,
) -> APIResponse[EvalTestsetResponse]:
    """Generate test set from documents."""
    evaluator = get_evaluator()

    try:
        # Get document paths
        doc_paths = await _get_document_paths(request.document_ids)

        if not doc_paths:
            raise HTTPException(
                status_code=404,
                detail="No documents found for given IDs",
            )

        # Generate testset
        testset = await evaluator._generate_testset_from_docs(
            document_paths=doc_paths,
            num_samples=request.num_samples,
            question_types=request.question_types,
        )

        return APIResponse.ok(
            data=EvalTestsetResponse(
                testset=testset,
                total=len(testset),
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Testset generation failed: {}", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Testset generation failed: {exc}",
        )


# ---------------------------------------------------------------------------
# Health check endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=APIResponse[dict],
    summary="Check Evaluation Service Health",
    description="Check if RAGAS is available and service is ready.",
)
async def check_health() -> APIResponse[dict]:
    """Check evaluation service health."""
    evaluator = get_evaluator()
    return APIResponse.ok(
        data={
            "ragas_available": evaluator._ragas_available,
            "service_status": "ready",
        },
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _should_store_reports() -> bool:
    """Check if reports should be stored."""
    try:
        eval_settings = getattr(settings, "evaluation", None)
        if eval_settings is None:
            return True  # Default to enabled
        return bool(getattr(eval_settings, "store_reports", True))
    except Exception:
        return True


async def _store_evaluation_report(report: EvaluationReport) -> None:
    """Store evaluation report to persistent storage.

    Saves both the raw JSON report and a human-readable CSV summary in
    ``{base_dir}/evaluation_reports``.
    """
    try:
        # Import storage helper (lazy to avoid circular imports)
        from app.core.storage import FileStorage

        storage = FileStorage()
        reports_dir = storage.base_dir / "evaluation_reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        import csv
        import json

        # ── JSON ─────────────────────────────────────────────────────
        report_file = reports_dir / f"{report.id}.json"
        report_dict = report.model_dump(mode="json")

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)

        # ── CSV table ────────────────────────────────────────────────
        csv_file = reports_dir / f"{report.id}.csv"
        metric_names = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "noise_sensitivity",
            "response_conciseness",
        ]

        with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "question",
                    "ground_truth",
                    "answer",
                    "context_count",
                    *metric_names,
                ]
            )

            for sample in report.samples:
                writer.writerow(
                    [
                        sample.question,
                        sample.ground_truth,
                        sample.answer,
                        len(sample.contexts),
                        *[sample.metrics.get(m, "") for m in metric_names],
                    ]
                )

            # Summary row
            writer.writerow([])
            writer.writerow(["metric", "average_score"])
            for metric_name in metric_names:
                avg_value = getattr(report, f"{metric_name}_avg", 0.0)
                writer.writerow([metric_name, avg_value])

        logger.info(
            "Stored evaluation report: {} (json + csv)",
            report.id,
        )

    except Exception as exc:
        logger.error("Failed to store evaluation report: {}", exc)


def _get_evaluation_history(
    limit: int = 10,
    offset: int = 0,
) -> list[EvaluationHistoryItem]:
    """Get evaluation history from storage."""
    try:
        from app.core.storage import FileStorage
        import json

        storage = FileStorage()
        reports_dir = storage.base_dir / "evaluation_reports"

        if not reports_dir.exists():
            return []

        reports = []
        for report_file in sorted(
            reports_dir.glob("*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )[offset : offset + limit]:
            try:
                with open(report_file, encoding="utf-8") as f:
                    data = json.load(f)
                    reports.append(
                        EvaluationHistoryItem(
                            id=data["id"],
                            created_at=data["created_at"],
                            dataset_name=data.get("dataset_name", ""),
                            total_samples=data.get("total_samples", 0),
                            overall_score=data.get("overall_score", 0.0),
                        )
                    )
            except Exception:
                continue

        return reports

    except Exception as exc:
        logger.error("Failed to get evaluation history: {}", exc)
        return []


def _get_evaluation_report_by_id(report_id: str) -> EvaluationReport | None:
    """Get specific evaluation report from storage."""
    try:
        from app.core.storage import FileStorage
        import json

        storage = FileStorage()
        report_file = storage.base_dir / "evaluation_reports" / f"{report_id}.json"

        if not report_file.exists():
            return None

        with open(report_file, encoding="utf-8") as f:
            data = json.load(f)

        return EvaluationReport.model_validate(data)

    except Exception as exc:
        logger.error("Failed to get report {}: {}", report_id, exc)
        return None


async def _get_document_paths(document_ids: list[str]) -> list[str]:
    """Get document file paths from IDs."""
    try:
        from app.core.storage import FileStorage

        storage = FileStorage()
        paths = []

        for doc_id in document_ids:
            doc_dir = storage.base_dir / doc_id
            if doc_dir.exists():
                files = list(doc_dir.iterdir())
                if files:
                    paths.append(str(files[0]))

        return paths

    except Exception as exc:
        logger.error("Failed to get document paths: {}", exc)
        return []
