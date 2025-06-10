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

from pydantic import BaseModel, Field

from src.config import DataSplitConfig

# ============================================================================
# ENUMS (Shared between API and Database)
# ============================================================================


class DeploymentStatus(str, Enum):
    """Status details of the deployment."""

    CREATED = "created"
    PENDING = "pending"
    RUNNING = "running"
    CANCELLED = "cancelled"
    CANCELLING = "cancelling"
    FAILED = "failed"
    COMPLETED = "completed"
    READY = "ready"
    UNKNOWN = "unknown"


class NIMRunStatus(str, Enum):
    """Status of a NIM run workflow."""

    PENDING = "pending"
    DEPLOYING = "deploying-nim"
    RUNNING = "running-evals"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


class FlywheelRunStatus(str, Enum):
    """Status of a Flywheel run workflow."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


# ============================================================================
# API REQUEST MODELS
# ============================================================================


class JobRequest(BaseModel):
    """Request model for creating a new NIM workflow job."""

    workload_id: str = Field(
        ...,
        description="The unique identifier of the workload to process",
        examples=["workload_123"],
    )

    client_id: str = Field(
        ...,
        description="The unique identifier of the client to process",
        examples=["client_123"],
    )

    data_split_config: DataSplitConfig | None = Field(
        None,
        description="Optional configuration for data splitting. If not provided, default config will be used.",
    )


# ============================================================================
# API RESPONSE MODELS
# ============================================================================


class JobResponse(BaseModel):
    """Response model for job creation."""

    id: str = Field(
        ...,
        description="The unique identifier of the created job",
        examples=["65f8a1b2c3d4e5f6a7b8c9d0"],
    )
    status: str = Field(
        ...,
        description="Current status of the job",
        examples=["queued"],
        enum=["queued", "running", "completed", "failed"],
    )
    message: str = Field(
        ...,
        description="Human-readable message about the job status",
        examples=["NIM workflow started"],
    )


class JobDeleteResponse(BaseModel):
    """Response model for job deletion."""

    id: str = Field(
        ...,
        description="The unique identifier of the created job",
        examples=["65f8a1b2c3d4e5f6a7b8c9d0"],
    )
    message: str = Field(
        ...,
        description="Human-readable message about the job status",
        examples=["NIM workflow started"],
    )


class JobCancelResponse(BaseModel):
    """Response model for job cancellation."""

    id: str = Field(
        ...,
        description="The unique identifier of the job",
        examples=["65f8a1b2c3d4e5f6a7b8c9d0"],
    )
    message: str = Field(
        ...,
        description="Human-readable message about the cancellation",
        examples=["Job cancellation initiated"],
    )


class Dataset(BaseModel):
    """Model representing a dataset."""

    name: str = Field(..., description="Name of the dataset", examples=["dataset_123"])
    num_records: int = Field(
        ..., description="Number of records in the dataset", examples=[1000], ge=0
    )
    nmp_uri: str | None = Field(
        None,
        description="URI of the dataset stored in NMP",
        examples=["https://nmp.host/v1/datasets/dataset-123"],
    )


class JobListItem(BaseModel):
    """Model representing a job in the list of jobs."""

    id: str = Field(
        ..., description="The unique identifier of the job", examples=["65f8a1b2c3d4e5f6a7b8c9d0"]
    )
    workload_id: str = Field(
        ...,
        description="The unique identifier of the workload being processed",
        examples=["workload_123"],
    )
    client_id: str | None = Field(
        None,
        description="The unique identifier of the client to process",
        examples=["client_123"],
    )
    status: str = Field(
        ...,
        description="Current status of the job",
        examples=["running", "completed", "failed"],
    )
    started_at: datetime = Field(
        ...,
        description="Timestamp when the job started processing",
        examples=["2024-03-15T14:30:00Z"],
    )
    finished_at: datetime | None = Field(
        None,
        description="Timestamp when the job completed or failed",
        examples=["2024-03-15T15:30:00Z"],
    )
    datasets: list[Dataset] = Field(
        default_factory=list, description="List of datasets used in this job"
    )
    error: str | None = Field(
        None,
        description="Error message if the job failed",
        examples=["Job failed: Timeout"],
    )


class JobsListResponse(BaseModel):
    """Response model for listing all jobs."""

    jobs: list[JobListItem] = Field(..., description="List of all jobs, both active and completed")


class Evaluation(BaseModel):
    """Model representing an evaluation result for a NIM."""

    eval_type: str = Field(
        ...,
        description="Type of evaluation performed",
        examples=["accuracy"],
    )

    scores: dict[str, float] = Field(
        default_factory=dict,
        description="Dictionary of scores for the evaluation",
        examples=[{"score": 0.85, "function_name_and_args_accuracy": 0.95}],
    )

    started_at: datetime = Field(
        ..., description="Timestamp when the evaluation started", examples=["2024-03-15T14:30:00Z"]
    )
    finished_at: datetime | None = Field(
        None,
        description="Timestamp when the evaluation completed",
        examples=["2024-03-15T14:35:00Z"],
    )
    runtime_seconds: float = Field(
        ...,
        description="Time taken to complete the evaluation in seconds",
        examples=[300.5],
        ge=0.0,
    )
    progress: float = Field(
        ...,
        description="Progress of the evaluation as a percentage",
        examples=[100.0],
        ge=0.0,
        le=100.0,
    )
    nmp_uri: str | None = Field(
        None,
        description="URI of the evaluation job in NMP",
        examples=["https://nmp.host/v1/evaluation/jobs/eval-123"],
    )
    error: str | None = Field(
        None,
        description="Error message if the evaluation failed",
        examples=["Evaluation failed: Timeout"],
    )


class Customization(BaseModel):
    """Model representing a customization result for a NIM."""

    started_at: datetime = Field(
        ...,
        description="Timestamp when the customization started",
        examples=["2024-03-15T14:30:00Z"],
    )
    finished_at: datetime | None = Field(
        None,
        description="Timestamp when the customization completed",
        examples=["2024-03-15T14:35:00Z"],
    )
    runtime_seconds: float = Field(
        ...,
        description="Time taken to complete the customization in seconds",
        examples=[300.5],
        ge=0.0,
    )
    progress: float = Field(
        ...,
        description="Progress of the customization as a percentage",
        examples=[100.0],
        ge=0.0,
        le=100.0,
    )
    epochs_completed: int = Field(
        ..., description="Number of epochs completed", examples=[10], ge=0
    )
    steps_completed: int = Field(..., description="Number of steps completed", examples=[100], ge=0)
    nmp_uri: str | None = Field(
        None,
        description="URI of the customization job in NMP",
        examples=["https://nmp.host/v1/customization/jobs/custom-123"],
    )
    error: str | None = Field(
        None,
        description="Error message if the customization failed",
        examples=["Customization failed: Timeout"],
    )


class NIMResponse(BaseModel):
    """Model representing a NIM and its evaluations."""

    model_name: str = Field(..., description="Name of the NIM model", examples=["gpt-4"])

    status: NIMRunStatus = Field(
        ...,
        description="Status of the NIM run",
        examples=["pending", "running", "completed", "error"],
    )

    deployment_status: DeploymentStatus = Field(
        ..., description="Status of the NIM deployment", examples=["deployed"]
    )

    runtime_seconds: float = Field(
        ...,
        description="Time taken for the NIM run in seconds",
        examples=[300.5],
        ge=0.0,
    )

    evaluations: list[Evaluation] = Field(
        default_factory=list, description="List of evaluations performed on this NIM"
    )
    customizations: list[Customization] = Field(
        default_factory=list, description="List of customizations performed on this NIM"
    )

    error: str | None = Field(
        None,
        description="Error message if the NIM failed",
        examples=["NIM failed: Deployment timeout"],
    )


class LLMJudgeResponse(BaseModel):
    """Model representing a LLM Judge status"""

    model_name: str = Field(..., description="Name of the LLM Judge model", examples=["gpt-4"])
    type: str = Field(..., description="Type of LLM Judge", examples=["remote", "local"])

    deployment_status: DeploymentStatus = Field(
        ..., description="Status of the LLM Judge deployment", examples=["deployed"]
    )
    error: str | None = Field(
        None,
        description="Error message if the LLM Judge failed",
        examples=["LLM Judge failed: Connection timeout"],
    )


class JobDetailResponse(BaseModel):
    """Detailed response model for a specific job."""

    id: str = Field(
        ..., description="The unique identifier of the job", examples=["65f8a1b2c3d4e5f6a7b8c9d0"]
    )
    workload_id: str = Field(
        ...,
        description="The unique identifier of the workload being processed",
        examples=["workload_123"],
    )
    client_id: str = Field(
        ...,
        description="The unique identifier of the client to process",
        examples=["client_123"],
    )
    status: str = Field(
        ...,
        description="Current status of the job",
        examples=["running", "completed", "failed"],
    )
    started_at: datetime = Field(
        ...,
        description="Timestamp when the job started processing",
        examples=["2024-03-15T14:30:00Z"],
    )
    finished_at: datetime | None = Field(
        None,
        description="Timestamp when the job completed or failed",
        examples=["2024-03-15T15:30:00Z"],
    )
    num_records: int = Field(
        ..., description="Number of records processed in this job", examples=[1000], ge=0
    )
    llm_judge: LLMJudgeResponse | None = Field(None, description="LLM Judge status for this job")
    nims: list[NIMResponse] = Field(
        default_factory=list, description="List of NIMs and their evaluation results for this job"
    )
    datasets: list[Dataset] = Field(
        default_factory=list, description="List of datasets used in this job"
    )
    error: str | None = Field(
        None,
        description="Error message if the job failed",
        examples=["Job failed: Timeout"],
    )
