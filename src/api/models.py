# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from datetime import datetime
from enum import Enum
from typing import Any

from bson import ObjectId
from pydantic import BaseModel, Field

from src.api.schemas import Dataset, DeploymentStatus
from src.config import NIMConfig


class EvalType(str, Enum):
    """Types of evaluations that can be performed."""

    BASE = "base-eval"
    ICL = "icl-eval"
    CUSTOMIZED = "customized-eval"

    @classmethod
    def values(cls) -> set[str]:
        """Get all valid evaluation type values."""
        return {member.value for member in cls}

    def __str__(self) -> str:
        """String representation for easy MongoDB storage."""
        return self.value


class DatasetType(str, Enum):
    """Types of datasets that can be used for evaluations."""

    BASE = "base-dataset"
    ICL = "icl-dataset"
    TRAIN = "train-dataset"


class WorkloadClassification(str, Enum):
    """Types of workloads that can be generated."""

    GENERIC = "generic"
    TOOL_CALLING = "tool_calling"


class ToolEvalType(str, Enum):
    """Types of tool evaluations that can be performed."""

    TOOL_CALLING_METRIC = "tool-calling-metric"
    TOOL_CALLING_JUDGE = "tool-calling-judge"


class JobStatus(BaseModel):
    """Generic job status tracking"""

    job_id: str | None = None
    percent_done: float | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    status: str | None = None


class EvaluationResult(JobStatus):
    """Individual evaluation result"""

    scores: dict[str, float] = Field(default_factory=dict)


class CustomizationResult(JobStatus):
    """Customization job result"""

    model_name: str | None = None
    evaluation_id: str | None = None  # Store evaluation ID instead of full object
    epochs_completed: int | None = None
    steps_completed: int | None = None


class TaskResult(BaseModel):
    """Result from any task in the workflow"""

    status: str | None = None
    workload_id: str | None = None
    client_id: str | None = None
    flywheel_run_id: str | None = None
    nim: NIMConfig | None = None
    workload_type: WorkloadClassification | None = None
    datasets: dict[str, str] = {}
    # Base evaluations
    evaluations: dict[EvalType, EvaluationResult] = {}

    # Separate customization tracking
    customization: CustomizationResult | None = None
    llm_judge_config: NIMConfig | None = None
    # Store error message if any stage fails so downstream tasks can short-circuit
    error: str | None = None

    def add_evaluation(self, eval_type: EvalType, result: EvaluationResult):
        """Helper method to add/update evaluation results"""
        self.evaluations[eval_type] = result

    def get_evaluation(self, eval_type: EvalType) -> EvaluationResult | None:
        """Helper method to get evaluation results"""
        return self.evaluations.get(eval_type)

    def update_customization(
        self,
        job_id: str,
        model_name: str,
        started_at: datetime,
        finished_at: datetime | None = None,
        percent_done: float = 0.0,
        epochs_completed: int | None = None,
        steps_completed: int | None = None,
    ) -> None:
        """Helper method to update customization job status with training progress"""
        if not self.customization:
            self.customization = CustomizationResult(
                job_id=job_id,
                model_name=model_name,
                started_at=started_at,
                finished_at=finished_at,
                percent_done=percent_done,
                epochs_completed=epochs_completed,
                steps_completed=steps_completed,
            )
        else:
            # Update all fields
            for key, value in locals().items():
                if key not in ["self"] and hasattr(self.customization, key) and value is not None:
                    setattr(self.customization, key, value)

    def get_customization_progress(self) -> dict[str, Any]:
        """Helper method to get detailed customization progress"""
        if not self.customization:
            return {}

        return {
            "percent_done": self.customization.percent_done,
            "epochs_completed": self.customization.epochs_completed,
            "steps_completed": self.customization.steps_completed,
        }


class NIMEvaluation(BaseModel):
    """Results from a NIM evaluation run."""

    id: ObjectId | None = Field(default_factory=ObjectId, alias="_id")
    nim_id: ObjectId
    eval_type: EvalType
    scores: dict[str, float]
    started_at: datetime
    finished_at: datetime | None = None
    runtime_seconds: float
    progress: float  # Progress percentage (0-100)
    nmp_uri: str | None = None
    error: str | None = None

    model_config = {
        "arbitrary_types_allowed": True,
        "json_encoders": {
            ObjectId: str,
            EvalType: str,  # Automatically convert enum to string
        },
    }

    def to_mongo(self) -> dict[str, Any]:
        """Convert NIMEvaluation model to MongoDB document."""
        return {
            "_id": self.id,
            "nim_id": self.nim_id,
            "eval_type": self.eval_type,
            "scores": self.scores,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "runtime_seconds": self.runtime_seconds,
            "progress": self.progress,
            "nmp_uri": self.nmp_uri,
        }

    @classmethod
    def from_mongo(cls, doc: dict[str, Any]) -> "NIMEvaluation":
        """Create NIMEvaluation from MongoDB document."""
        return cls(
            id=doc["_id"],
            nim_id=doc["nim_id"],
            eval_type=doc["eval_type"],
            scores=doc["scores"],
            started_at=doc["started_at"],
            finished_at=doc["finished_at"],
            runtime_seconds=doc["runtime_seconds"],
            progress=doc["progress"],
            nmp_uri=doc.get("nmp_uri"),
        )


class NIMCustomization(BaseModel):
    """Results from a NIM customization run."""

    id: ObjectId | None = Field(default_factory=ObjectId, alias="_id")
    nim_id: ObjectId
    workload_id: str
    base_model: str
    customized_model: str | None = None  # Make this optional since it's set after job starts
    started_at: datetime
    finished_at: datetime | None = None
    runtime_seconds: float = 0.0
    progress: float = 0.0  # Progress percentage (0-100)
    epochs_completed: int | None = None
    steps_completed: int | None = None
    nmp_uri: str | None = None
    error: str | None = None

    model_config = {"arbitrary_types_allowed": True, "json_encoders": {ObjectId: str}}

    def to_mongo(self) -> dict:
        """Convert NIMCustomization model to MongoDB document."""
        return self.model_dump(by_alias=True)

    @classmethod
    def from_mongo(cls, data: dict) -> "NIMCustomization":
        """Create NIMCustomization from MongoDB document."""
        if not data:
            return None
        return cls(**data)


class NIMRunStatus(str, Enum):
    DEPLOYING = "deploying-nim"
    RUNNING = "running-evals"
    COMPLETED = "complete"
    ERROR = "error"


class NIMRun(BaseModel):
    """Results from a NIM run."""

    id: ObjectId | None = Field(default_factory=ObjectId, alias="_id")
    flywheel_run_id: ObjectId
    model_name: str
    started_at: datetime
    finished_at: datetime
    runtime_seconds: float
    evaluations: list[NIMEvaluation] = []
    status: NIMRunStatus | None = None
    deployment_status: DeploymentStatus | None = None
    model_config = {"arbitrary_types_allowed": True, "json_encoders": {ObjectId: str}}
    error: str | None = None

    def to_mongo(self) -> dict:
        """Convert NIMRun model to MongoDB document."""
        return self.model_dump(by_alias=True)

    @classmethod
    def from_mongo(cls, data: dict) -> "NIMRun":
        """Create NIMRun from MongoDB document."""
        if not data:
            return None
        return cls(**data)


class LLMJudgeRun(BaseModel):
    """Status of LLM Judge run."""

    id: ObjectId | None = Field(default_factory=ObjectId, alias="_id")
    flywheel_run_id: ObjectId
    model_name: str
    deployment_status: DeploymentStatus | None = None
    model_config = {"arbitrary_types_allowed": True, "json_encoders": {ObjectId: str}}
    error: str | None = None

    def to_mongo(self) -> dict:
        """Convert NIMRun model to MongoDB document."""
        return self.model_dump(by_alias=True)

    @classmethod
    def from_mongo(cls, data: dict) -> "NIMRun":
        """Create NIMRun from MongoDB document."""
        if not data:
            return None
        return cls(**data)


class FlywheelRun(BaseModel):
    """Results from a Flywheel run."""

    id: ObjectId | None = Field(default_factory=ObjectId, alias="_id")
    workload_id: str
    started_at: datetime
    client_id: str = None
    finished_at: datetime | None = None
    num_records: int | None = None
    nims: list[NIMRun] = []
    # Keep the fixed datasets field that matches the database structure
    datasets: list[Dataset] = []
    error: str | None = None
    model_config = {"arbitrary_types_allowed": True, "json_encoders": {ObjectId: str}}

    def to_mongo(self) -> dict:
        """Convert FlywheelRun model to MongoDB document."""
        return self.model_dump(by_alias=True)

    @classmethod
    def from_mongo(cls, data: dict) -> "FlywheelRun":
        """Create FlywheelRun from MongoDB document."""
        if not data:
            return None
        return cls(**data)
