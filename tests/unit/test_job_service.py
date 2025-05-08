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
from datetime import datetime, timezone

import pytest
from bson import ObjectId
from fastapi import HTTPException

from src.api.job_service import get_job_details
from src.api.models import FlywheelRun
from src.api.schemas import (
    DeploymentStatus,
    JobDetailResponse,
)


@pytest.fixture
def test_db():
    """Fixture to set up test database with sample data"""
    from src.api.db import get_db, init_db

    init_db()
    db = get_db()

    # Clear existing data
    db.flywheel_runs.delete_many({})
    db.nims.delete_many({})
    db.evaluations.delete_many({})
    db.customizations.delete_many({})

    # Create test flywheel run
    flywheel_run = FlywheelRun(
        workload_id="test_workload",
        client_id="test_client",
        started_at=datetime.now(timezone.utc),
        num_records=100,
        nims=[],
    )
    result = db.flywheel_runs.insert_one(flywheel_run.to_mongo())
    flywheel_run_id = result.inserted_id

    # update flywheel run with datasets
    datasets = [
        {
            "name": f"test_dataset{f'_{i+1}'}",
            "num_records": 100,
            "nmp_uri": f"test_uri{f'_{i+1}'}",
        }
        for i in range(1, 5)
    ]

    db.flywheel_runs.update_one({"_id": flywheel_run_id}, {"$set": {"datasets": datasets}})

    # Create test NIMs
    nim1_id = ObjectId()
    nim2_id = ObjectId()
    db.nims.insert_many(
        [
            {
                "_id": nim1_id,
                "flywheel_run_id": flywheel_run_id,
                "model_name": "test_model_1",
                "deployment_status": "completed",
            },
            {
                "_id": nim2_id,
                "flywheel_run_id": flywheel_run_id,
                "model_name": "test_model_2",
                "deployment_status": "pending",
            },
        ]
    )

    # Create test evaluations
    db.evaluations.insert_many(
        [
            {
                "nim_id": nim1_id,
                "eval_type": "accuracy",
                "scores": {"function_name": 0.95, "function_name_and_args_accuracy": 0.98},
                "started_at": datetime.now(timezone.utc),
                "finished_at": datetime.now(timezone.utc),
                "runtime_seconds": 10.5,
                "progress": 1.0,
                "nmp_uri": "test_uri_1",
            },
            {
                "nim_id": nim2_id,
                "eval_type": "accuracy",
                "scores": {"function_name": 0.85, "function_name_and_args_accuracy": 0.88},
                "started_at": datetime.now(timezone.utc),
                "finished_at": datetime.now(timezone.utc),
                "runtime_seconds": 8.5,
                "progress": 1.0,
                "nmp_uri": "test_uri_2",
            },
        ]
    )

    # Create test customizations
    db.customizations.insert_many(
        [
            {
                "nim_id": nim1_id,
                "started_at": datetime.now(timezone.utc),
                "finished_at": datetime.now(timezone.utc),
                "runtime_seconds": 20.5,
                "progress": 1.0,
                "epochs_completed": 10,
                "steps_completed": 100,
                "nmp_uri": "test_uri_1",
            },
            {
                "nim_id": nim2_id,
                "started_at": datetime.now(timezone.utc),
                "finished_at": datetime.now(timezone.utc),
                "runtime_seconds": 15.5,
                "progress": 1.0,
                "epochs_completed": 8,
                "steps_completed": 80,
                "nmp_uri": "test_uri_2",
            },
        ]
    )

    # Add LLM Judge data to flywheel run
    llm_judge_data = {
        "flywheel_run_id": flywheel_run_id,
        "model_name": "test-llm-judge",
        "deployment_status": "ready",
    }

    db.llm_judge_runs.insert_one(llm_judge_data)

    return {
        "flywheel_run_id": str(flywheel_run_id),
        "nim1_id": str(nim1_id),
        "nim2_id": str(nim2_id),
    }


def test_get_job_details_datasets(test_db):
    """Test successful retrieval of job details"""
    result = get_job_details(test_db["flywheel_run_id"])
    assert len(result.datasets) == 4
    assert isinstance(result.datasets, list)


def test_get_job_details_success(test_db):
    """Test successful retrieval of job details"""
    result = get_job_details(test_db["flywheel_run_id"])

    assert isinstance(result, JobDetailResponse)
    assert result.id == test_db["flywheel_run_id"]
    assert result.workload_id == "test_workload"
    assert result.status == "running"  # Since finished_at is None
    assert result.num_records == 100
    assert len(result.datasets) == 4
    # Verify NIMs
    assert len(result.nims) == 2
    nim1 = next(nim for nim in result.nims if nim.model_name == "test_model_1")
    nim2 = next(nim for nim in result.nims if nim.model_name == "test_model_2")

    assert nim1.deployment_status == DeploymentStatus.COMPLETED
    assert nim2.deployment_status == DeploymentStatus.PENDING

    # Verify evaluations
    assert len(nim1.evaluations) == 1
    assert len(nim2.evaluations) == 1
    assert nim1.evaluations[0].scores["function_name"] == 0.95
    assert nim2.evaluations[0].scores["function_name"] == 0.85

    # Verify customizations
    assert len(nim1.customizations) == 1
    assert len(nim2.customizations) == 1
    assert nim1.customizations[0].epochs_completed == 10
    assert nim2.customizations[0].epochs_completed == 8

    # Add LLM Judge assertions
    assert result.llm_judge is not None
    assert result.llm_judge.model_name == "test-llm-judge"
    assert result.llm_judge.deployment_status == DeploymentStatus.READY


def test_get_job_details_not_found():
    """Test handling of non-existent job ID"""
    with pytest.raises(HTTPException) as exc_info:
        get_job_details(str(ObjectId()))

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Job not found"


def test_get_job_details_completed(test_db):
    """Test retrieval of completed job details"""
    from src.api.db import get_db

    db = get_db()

    # Update flywheel run to be completed
    db.flywheel_runs.update_one(
        {"_id": ObjectId(test_db["flywheel_run_id"])},
        {"$set": {"finished_at": datetime.now(timezone.utc)}},
    )

    result = get_job_details(test_db["flywheel_run_id"])
    assert result.status == "completed"
    assert result.finished_at is not None


def test_get_job_details_no_nims(test_db):
    """Test retrieval of job details with no NIMs"""
    from src.api.db import get_db

    db = get_db()

    # Delete all NIMs
    db.nims.delete_many({"flywheel_run_id": ObjectId(test_db["flywheel_run_id"])})

    result = get_job_details(test_db["flywheel_run_id"])
    assert len(result.nims) == 0


def test_get_job_details_no_evaluations(test_db):
    """Test retrieval of job details with no evaluations"""
    from src.api.db import get_db

    db = get_db()

    # Delete all evaluations
    db.evaluations.delete_many({})

    result = get_job_details(test_db["flywheel_run_id"])
    for nim in result.nims:
        assert len(nim.evaluations) == 0


def test_get_job_details_no_customizations(test_db):
    """Test retrieval of job details with no customizations"""
    from src.api.db import get_db

    db = get_db()

    # Delete all customizations
    db.customizations.delete_many({})

    result = get_job_details(test_db["flywheel_run_id"])
    for nim in result.nims:
        assert len(nim.customizations) == 0


def test_get_job_details_llm_judge(test_db):
    """Test retrieval of job details with LLM Judge information"""
    result = get_job_details(test_db["flywheel_run_id"])
    print(result)
    assert result.llm_judge is not None
    assert result.llm_judge.model_name == "test-llm-judge"
    assert result.llm_judge.deployment_status == DeploymentStatus.READY
