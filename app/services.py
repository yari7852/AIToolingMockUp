"""Business logic for the mock labeling portal service."""

from __future__ import annotations

import itertools
import math
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Tuple

from .database import DB
from .schemas import (
    Annotation,
    AnnotationCreate,
    AssignmentRequest,
    ConsensusResult,
    EvaluatorReport,
    Prediction,
    ReliabilityMetrics,
    RetrainingTrigger,
    Task,
    TaskCreate,
    TaskPriority,
    Vote,
    VoteCreate,
    WebhookPayload,
)


def _now() -> datetime:
    return datetime.utcnow()


# ------------------ Task management ------------------ #

def create_task(payload: TaskCreate) -> Task:
    task_id = str(uuid.uuid4())
    priority = _calculate_priority(payload.uncertainty, payload.difficulty)
    task = Task(
        id=task_id,
        video_id=payload.video_id,
        priority=priority,
        created_at=_now(),
        updated_at=_now(),
        uncertainty=payload.uncertainty,
    )
    DB.tasks[task_id] = task
    return task


def _calculate_priority(uncertainty: float, difficulty: TaskPriority) -> TaskPriority:
    scaled = uncertainty * 10 + difficulty
    if scaled >= 12:
        return TaskPriority.high
    if scaled >= 7:
        return TaskPriority.medium
    return TaskPriority.low


def list_tasks() -> List[Task]:
    return sorted(DB.tasks.values(), key=lambda t: (-t.priority.value, t.created_at))


def request_assignment(req: AssignmentRequest) -> Task | None:
    annotator_metrics = DB.annotator_metrics[req.annotator_id]
    reliability = annotator_metrics["reliability"]
    sorted_tasks = list_tasks()
    for task in sorted_tasks:
        if task.status != "pending":
            continue
        if task.priority is TaskPriority.high and reliability < 0.6:
            continue
        DB.tasks[task.id] = task.copy(
            update={"assigned_to": req.annotator_id, "status": "assigned", "updated_at": _now()}
        )
        return DB.tasks[task.id]
    return None


# ------------------ Prediction ingestion ------------------ #

def upsert_prediction(prediction: Prediction) -> dict:
    DB.predictions[prediction.video_id] = prediction.dict()
    return DB.predictions[prediction.video_id]


# ------------------ Annotation handling ------------------ #

def submit_annotation(task_id: str, payload: AnnotationCreate) -> Annotation:
    if task_id not in DB.tasks:
        raise KeyError("Task not found")
    annotation_id = str(uuid.uuid4())
    annotation = Annotation(
        id=annotation_id,
        task_id=task_id,
        annotator_id=payload.annotator_id,
        caption=payload.caption,
        created_at=_now(),
    )
    DB.annotations[annotation_id] = annotation
    DB.task_annotations[task_id].append(annotation_id)
    _update_metrics_on_completion(task_id, payload.annotator_id)
    _maybe_update_task_status(task_id)
    return annotation


def _update_metrics_on_completion(task_id: str, annotator_id: str) -> None:
    metrics = DB.annotator_metrics[annotator_id]
    metrics["completed"] += 1
    metrics["total_seconds"] += random.uniform(45, 120)
    consensus = DB.consensus.get(task_id)
    if consensus:
        annotations = [DB.annotations[ann_id] for ann_id in DB.task_annotations[task_id]]
        metrics["disagreements"] += sum(
            1 for ann in annotations if _semantic_similarity(ann.caption, consensus.consensus_caption) < 0.7
        )
    metrics["reliability"] = _compute_reliability(metrics)


def _compute_reliability(metrics: Dict[str, float]) -> float:
    if metrics["completed"] == 0:
        return 0.5
    agreement_ratio = 1 - (metrics["disagreements"] / max(metrics["completed"], 1))
    avg_time = metrics["total_seconds"] / metrics["completed"]
    speed_factor = min(1.0, 90 / max(avg_time, 1))
    return round(max(0.1, min(0.99, 0.4 * agreement_ratio + 0.6 * speed_factor)), 3)


def _maybe_update_task_status(task_id: str) -> None:
    task = DB.tasks[task_id]
    annotations = DB.task_annotations[task_id]
    if len(annotations) >= 3:
        DB.tasks[task_id] = task.copy(update={"status": "awaiting_review", "updated_at": _now()})


# ------------------ Voting and consensus ------------------ #

def submit_vote(task_id: str, payload: VoteCreate) -> Vote:
    if task_id not in DB.tasks:
        raise KeyError("Task not found")
    vote_id = str(uuid.uuid4())
    vote = Vote(
        id=vote_id,
        task_id=task_id,
        annotator_id=payload.annotator_id,
        score=payload.score,
        rationale=payload.rationale,
        created_at=_now(),
    )
    DB.votes[vote_id] = vote
    DB.task_votes[task_id].append(vote_id)
    return vote


def finalize_consensus(task_id: str) -> ConsensusResult:
    if task_id not in DB.tasks:
        raise KeyError("Task not found")
    annotations = [DB.annotations[ann_id] for ann_id in DB.task_annotations[task_id]]
    if not annotations:
        raise ValueError("No annotations available for consensus")
    caption, semantic_agreement = _aggregate_semantic(ann.caption for ann in annotations)
    llm_confidence = _mock_llm_evaluation(caption)
    consensus = ConsensusResult(
        task_id=task_id,
        consensus_caption=caption,
        semantic_agreement=semantic_agreement,
        llm_confidence=llm_confidence,
        finalized_at=_now(),
    )
    DB.consensus[task_id] = consensus
    DB.tasks[task_id] = DB.tasks[task_id].copy(update={"status": "finalized", "updated_at": _now()})
    return consensus


def _aggregate_semantic(captions: Iterable[str]) -> Tuple[str, float]:
    captions = list(captions)
    if not captions:
        raise ValueError("No captions to aggregate")
    centroid = captions[0]
    similarities = [
        _semantic_similarity(centroid, caption)
        for caption in captions
    ]
    avg_similarity = sum(similarities) / len(similarities)
    best_caption = max(captions, key=lambda c: _semantic_similarity(c, centroid))
    return best_caption, round(avg_similarity, 3)


def _semantic_similarity(a: str, b: str) -> float:
    overlap = len(set(a.lower().split()) & set(b.lower().split()))
    total = max(len(set(a.lower().split())) + len(set(b.lower().split())), 1)
    return round(overlap / total, 3)


def _mock_llm_evaluation(caption: str) -> float:
    return round(0.6 + min(0.4, len(caption) / 200), 3)


# ------------------ Retraining & evaluator ------------------ #

def trigger_retraining(payload: RetrainingTrigger) -> WebhookPayload:
    labeled = [DB.tasks[task_id].dict() for task_id in payload.labeled_task_ids if task_id in DB.tasks]
    webhook_payload = WebhookPayload(
        event="retraining.triggered",
        timestamp=_now(),
        payload={
            "model_version": payload.model_version,
            "mini_batch_id": payload.mini_batch_id,
            "labeled_tasks": labeled,
        },
    )
    return webhook_payload


def evaluate_retrained_model(task_id: str) -> EvaluatorReport:
    if task_id not in DB.tasks:
        raise KeyError("Task not found")
    original = DB.predictions.get(DB.tasks[task_id].video_id, {})
    consensus = DB.consensus.get(task_id)
    retrained_caption = _mutate_caption(consensus.consensus_caption if consensus else "")
    agreement = _semantic_similarity(original.get("caption", ""), retrained_caption)
    return EvaluatorReport(
        task_id=task_id,
        original_caption=original.get("caption", ""),
        retrained_caption=retrained_caption,
        agreement=agreement,
        reviewed_at=_now(),
    )


def _mutate_caption(caption: str) -> str:
    words = caption.split()
    if not words:
        return caption
    if len(words) > 5:
        words.insert(0, "Updated")
    else:
        words.append("refined")
    return " ".join(words)


# ------------------ Metrics & dashboards ------------------ #

def get_reliability(annotator_id: str) -> ReliabilityMetrics:
    metrics = DB.annotator_metrics[annotator_id]
    avg_time = metrics["total_seconds"] / max(metrics["completed"], 1)
    disagreement_rate = metrics["disagreements"] / max(metrics["completed"], 1)
    return ReliabilityMetrics(
        annotator_id=annotator_id,
        reliability=round(metrics["reliability"], 3),
        throughput=metrics["completed"],
        average_task_seconds=round(avg_time, 2),
        disagreement_rate=round(disagreement_rate, 3),
    )


def dashboard_snapshot() -> Dict[str, ReliabilityMetrics]:
    return {annotator: get_reliability(annotator) for annotator in DB.annotator_metrics.keys()}
