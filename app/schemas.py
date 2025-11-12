"""Pydantic models for the labeling portal mock service."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class TaskPriority(int, Enum):
    """Priority tiers for labeling tasks."""

    low = 1
    medium = 5
    high = 10


class Prediction(BaseModel):
    video_id: str = Field(..., description="Unique identifier for the video clip")
    caption: str = Field(..., description="Model predicted caption")
    uncertainty: float = Field(..., ge=0, le=1, description="Uncertainty score between 0 and 1")
    model_version: str = Field(..., description="Model version used for the prediction")


class TaskCreate(BaseModel):
    video_id: str
    uncertainty: float = Field(..., ge=0, le=1)
    difficulty: TaskPriority = Field(TaskPriority.medium)


class Task(BaseModel):
    id: str
    video_id: str
    priority: TaskPriority
    assigned_to: Optional[str] = None
    status: str = "pending"
    created_at: datetime
    updated_at: datetime
    uncertainty: float


class AnnotationCreate(BaseModel):
    annotator_id: str
    caption: str


class Annotation(BaseModel):
    id: str
    task_id: str
    annotator_id: str
    caption: str
    created_at: datetime


class VoteCreate(BaseModel):
    annotator_id: str
    score: float = Field(..., ge=0, le=1)
    rationale: Optional[str] = None


class Vote(BaseModel):
    id: str
    task_id: str
    annotator_id: str
    score: float
    rationale: Optional[str]
    created_at: datetime


class ConsensusResult(BaseModel):
    task_id: str
    consensus_caption: str
    semantic_agreement: float
    llm_confidence: float
    finalized_at: datetime


class ReliabilityMetrics(BaseModel):
    annotator_id: str
    reliability: float
    throughput: int
    average_task_seconds: float
    disagreement_rate: float


class AssignmentRequest(BaseModel):
    annotator_id: str


class RetrainingTrigger(BaseModel):
    model_version: str
    mini_batch_id: str
    labeled_task_ids: List[str]


class WebhookPayload(BaseModel):
    event: str
    timestamp: datetime
    payload: dict


class EvaluatorReport(BaseModel):
    task_id: str
    original_caption: str
    retrained_caption: str
    agreement: float
    reviewed_at: datetime
