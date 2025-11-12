"""In-memory data store for the mock service."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Dict, List

from .schemas import Annotation, ConsensusResult, Task, Vote


class MemoryDatabase:
    """Lightweight data holder mimicking persistence."""

    def __init__(self) -> None:
        self.tasks: Dict[str, Task] = {}
        self.predictions: Dict[str, dict] = {}
        self.annotations: Dict[str, Annotation] = {}
        self.task_annotations: Dict[str, List[str]] = defaultdict(list)
        self.votes: Dict[str, Vote] = {}
        self.task_votes: Dict[str, List[str]] = defaultdict(list)
        self.consensus: Dict[str, ConsensusResult] = {}
        self.annotator_metrics: Dict[str, dict] = defaultdict(
            lambda: {
                "completed": 0,
                "total_seconds": 0.0,
                "disagreements": 0,
                "reliability": 0.5,
            }
        )

    def reset(self) -> None:
        self.__init__()


DB = MemoryDatabase()
