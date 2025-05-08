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
from bson import ObjectId
from fastapi import HTTPException

from src.api.db import get_db
from src.api.models import DeploymentStatus, FlywheelRun
from src.api.schemas import (
    Customization,
    Evaluation,
    JobDetailResponse,
    LLMJudgeResponse,
    NIMResponse,
)


def get_job_details(job_id: str) -> JobDetailResponse:
    """
    Get the status and result of a job, including detailed information about all tasks in the workflow.
    """
    db = get_db()
    doc = db.flywheel_runs.find_one({"_id": ObjectId(job_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Job not found")

    flywheel_run = FlywheelRun.from_mongo(doc)

    # Get all NIMs for this flywheel run
    nims = list(db.nims.find({"flywheel_run_id": ObjectId(job_id)}))

    # Get all evaluations for these NIMs
    nim_ids = [nim["_id"] for nim in nims]
    evaluations = list(db.evaluations.find({"nim_id": {"$in": nim_ids}}))

    # Group evaluations by NIM
    nim_evaluations: dict[ObjectId, list[Evaluation]] = {}
    for eval in evaluations:
        if eval["nim_id"] not in nim_evaluations:
            nim_evaluations[eval["nim_id"]] = []
        nim_evaluations[eval["nim_id"]].append(
            Evaluation(
                eval_type=eval["eval_type"],
                scores=eval["scores"],
                started_at=eval["started_at"],
                finished_at=eval["finished_at"],
                runtime_seconds=eval["runtime_seconds"],
                progress=eval["progress"],
                nmp_uri=eval["nmp_uri"],
            )
        )

    # Group customizations by NIM
    customizations = list(db.customizations.find({"nim_id": {"$in": nim_ids}}))
    nim_customizations: dict[ObjectId, list[Customization]] = {}
    for custom in customizations:
        if custom["nim_id"] not in nim_customizations:
            nim_customizations[custom["nim_id"]] = []
        nim_customizations[custom["nim_id"]].append(
            Customization(
                started_at=custom["started_at"],
                finished_at=custom["finished_at"],
                runtime_seconds=custom["runtime_seconds"],
                progress=custom["progress"],
                epochs_completed=custom["epochs_completed"],
                steps_completed=custom["steps_completed"],
                nmp_uri=custom["nmp_uri"],
            )
        )

    llm_judge = db.llm_judge_runs.find_one({"flywheel_run_id": flywheel_run.id})
    if llm_judge:
        llm_judge_response = LLMJudgeResponse(
            model_name=llm_judge["model_name"],
            deployment_status=DeploymentStatus(
                llm_judge["deployment_status"] or DeploymentStatus.PENDING
            ),
        )
    else:
        llm_judge_response = None

    return JobDetailResponse(
        id=str(flywheel_run.id),
        workload_id=flywheel_run.workload_id,
        client_id=flywheel_run.client_id,
        status="completed" if flywheel_run.finished_at else "running",
        started_at=flywheel_run.started_at,
        finished_at=flywheel_run.finished_at,
        num_records=flywheel_run.num_records or 0,
        llm_judge=llm_judge_response,
        nims=[
            NIMResponse(
                model_name=nim["model_name"],
                deployment_status=DeploymentStatus(
                    nim["deployment_status"] or DeploymentStatus.PENDING
                ),
                evaluations=nim_evaluations.get(nim["_id"], []),
                customizations=nim_customizations.get(nim["_id"], []),
            )
            for nim in nims
        ],
        datasets=flywheel_run.datasets,
    )
