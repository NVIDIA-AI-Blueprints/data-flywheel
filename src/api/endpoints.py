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

from fastapi import APIRouter

from src.api.db import get_db
from src.api.job_service import get_job_details
from src.api.models import FlywheelRun
from src.api.schemas import (
    JobDetailResponse,
    JobListItem,
    JobRequest,
    JobResponse,
    JobsListResponse,
)
from src.log_utils import setup_logging
from src.tasks.tasks import run_nim_workflow_dag

logger = setup_logging("data_flywheel.api.endpoints")

router = APIRouter()


@router.post("/jobs", response_model=JobResponse)
async def create_job(request: JobRequest) -> JobResponse:
    """
    Create a new job that runs the NIM workflow.
    """
    # create entry for current time, workload_id, and model_name
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"Request received at {current_time} for workload_id {request.workload_id} and client_id {request.client_id}"
    logger.info(entry)

    # Create FlywheelRun document
    flywheel_run = FlywheelRun(
        workload_id=request.workload_id,
        client_id=request.client_id,
        started_at=datetime.utcnow(),
        num_records=0,  # Will be updated when datasets are created
        nims=[],
    )

    # Save to MongoDB
    db = get_db()
    result = db.flywheel_runs.insert_one(flywheel_run.to_mongo())
    flywheel_run.id = str(result.inserted_id)

    # Call the NIM workflow task asynchronously. This will be executed
    # in the background.
    run_nim_workflow_dag.delay(
        workload_id=request.workload_id,
        flywheel_run_id=flywheel_run.id,
        client_id=request.client_id,
    )

    return JobResponse(id=flywheel_run.id, status="queued", message="NIM workflow started")


@router.get("/jobs", response_model=JobsListResponse)
async def get_jobs() -> JobsListResponse:
    """
    Get a list of all active and recent jobs.
    """
    db = get_db()
    jobs: list[JobListItem] = []

    # Get all FlywheelRun documents
    for doc in db.flywheel_runs.find():
        flywheel_run = FlywheelRun.from_mongo(doc)
        job = JobListItem(
            id=str(flywheel_run.id),
            workload_id=flywheel_run.workload_id,
            client_id=flywheel_run.client_id,
            status="completed" if flywheel_run.finished_at else "running",
            started_at=flywheel_run.started_at,
            finished_at=flywheel_run.finished_at,
            datasets=flywheel_run.datasets,
            error=flywheel_run.error,
        )
        jobs.append(job)

    return JobsListResponse(jobs=jobs)


@router.get("/jobs/{job_id}", response_model=JobDetailResponse)
async def get_job(job_id: str) -> JobDetailResponse:
    """
    Get the status and result of a job, including detailed information about all tasks in the workflow.
    """
    return get_job_details(job_id)
