"""FastAPI entrypoint for the mock labeling portal service."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from . import services
from .schemas import (
    AnnotationCreate,
    AssignmentRequest,
    Prediction,
    RetrainingTrigger,
    TaskCreate,
    VoteCreate,
)

app = FastAPI(title="Labeling Portal Mock Service", version="0.1.0")


@app.post("/predictions")
def ingest_prediction(prediction: Prediction) -> dict:
    """Store a model prediction and its uncertainty score."""
    return services.upsert_prediction(prediction)


@app.post("/tasks", response_model=dict)
def create_task(payload: TaskCreate) -> dict:
    """Create a task from a high-uncertainty sample."""
    task = services.create_task(payload)
    return task.dict()


@app.get("/tasks")
def get_tasks() -> list[dict]:
    return [task.dict() for task in services.list_tasks()]


@app.post("/tasks/assign")
def assign_task(request: AssignmentRequest) -> dict:
    task = services.request_assignment(request)
    if not task:
        raise HTTPException(status_code=404, detail="No tasks available for this annotator")
    return task.dict()


@app.post("/tasks/{task_id}/annotations")
def annotate_task(task_id: str, payload: AnnotationCreate) -> dict:
    try:
        annotation = services.submit_annotation(task_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return annotation.dict()


@app.post("/tasks/{task_id}/votes")
def vote_on_task(task_id: str, payload: VoteCreate) -> dict:
    try:
        vote = services.submit_vote(task_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return vote.dict()


@app.post("/tasks/{task_id}/consensus")
def finalize_task(task_id: str) -> dict:
    try:
        consensus = services.finalize_consensus(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return consensus.dict()


@app.get("/annotators/{annotator_id}/metrics")
def annotator_metrics(annotator_id: str) -> dict:
    metrics = services.get_reliability(annotator_id)
    return metrics.dict()


@app.get("/dashboard")
def dashboard() -> dict:
    snapshot = services.dashboard_snapshot()
    return {annotator: metrics.dict() for annotator, metrics in snapshot.items()}


@app.post("/retraining/trigger")
def trigger_retraining(payload: RetrainingTrigger) -> dict:
    webhook_payload = services.trigger_retraining(payload)
    return webhook_payload.dict()


@app.get("/tasks/{task_id}/evaluation")
def evaluate_task(task_id: str) -> dict:
    try:
        report = services.evaluate_retrained_model(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return report.dict()
