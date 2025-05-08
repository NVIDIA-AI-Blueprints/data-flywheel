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
import os
import sys
import uuid
from collections.abc import Generator
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient

from src.api.db import get_db
from src.api.models import EvalType
from src.api.schemas import DeploymentStatus

os.environ["ELASTICSEARCH_URL"] = "http://localhost:9200"
os.environ["ES_COLLECTION_NAME"] = "flywheel-test"
os.environ["REDIS_URL"] = "redis://localhost:6379/1"
os.environ["MONGODB_URL"] = "mongodb://localhost:27017"
os.environ["MONGODB_DB_NAME"] = "flywheel-test"

WORKLOAD_ID = f"test-workload-{uuid.uuid4()}"
CLIENT_ID = f"test-client-{uuid.uuid4()}"


@pytest.fixture
def load_test_data_fixture() -> Generator:
    """Fixture to provide the load_test_data function."""

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    sys.path.insert(0, root_dir)
    from src.scripts.load_test_data import load_data_to_elasticsearch

    yield load_data_to_elasticsearch(WORKLOAD_ID, CLIENT_ID)


def create_openai_request_response(data: dict[str, Any]) -> dict[str, Any]:
    """Transform the data into an OpenAI-style request/response pair."""
    # Create a timestamp for the request
    timestamp = datetime.now(timezone.utc).isoformat()

    # Create the request structure
    request = {
        "model": "not-a-model",
        "messages": [{"role": "user", "content": data["messages"][0]["content"]}],
        "temperature": 0.7,
        "max_tokens": 1000,
    }

    # Create the response structure
    response = {
        "id": f"chatcmpl-{timestamp}",
        "object": "chat.completion",
        "created": int(datetime.now(timezone.utc).timestamp()),
        "model": "not-a-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": data["messages"][1]["content"]},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(data["messages"][0]["content"].split()),
            "completion_tokens": len(data["messages"][1]["content"].split()),
            "total_tokens": len(data["messages"][0]["content"].split())
            + len(data["messages"][1]["content"].split()),
        },
    }

    return {"timestamp": timestamp, "request": request, "response": response}


@pytest.fixture
def test_client() -> TestClient:
    """Create a test client for the FastAPI app"""
    from src.api.db import init_db
    from src.app import app
    from src.tasks.tasks import celery_app

    celery_app.conf.update(task_always_eager=True, task_eager_propagates=True)

    init_db()

    return TestClient(app)


@pytest.fixture
def mock_external_services() -> Generator[dict[str, MagicMock], None, None]:
    """Mock external service responses"""
    with (
        patch("src.lib.nemo.data_uploader.DataUploader.upload_data") as mock_upload_data,
        patch("src.lib.nemo.data_uploader.DataUploader.get_file_uri") as mock_get_file_uri,
        patch("src.lib.nemo.evaluator.requests") as mock_requests,
        patch("src.lib.nemo.dms_client.requests") as mock_dms_requests,
        patch("src.lib.nemo.customizer.requests") as mock_customizer_requests,
    ):
        output_model = f"custom-model-{uuid.uuid4()}"

        def mock_dms_requests_side_effect(method, url, **kwargs):
            if method == "post" and "/v1/deployment/model-deployments" in url:
                return MagicMock(status_code=200, json=lambda: {"deployment_id": "deployment-123"})
            elif method == "get" and "/v1/deployment/model-deployments" in url:
                return MagicMock(
                    status_code=200,
                    json=lambda: {"status_details": {"status": DeploymentStatus.READY}},
                )
            elif method == "get" and "/v1/models" in url:
                return MagicMock(
                    status_code=200,
                    json=lambda: {
                        "data": [
                            {"id": "meta/llama-3.2-1b-instruct", "status": "ready"},
                            {"id": "test-model-id", "status": "ready"},
                        ]
                    },
                )
            else:
                raise ValueError(f"Unexpected request: {method} {url}")

        mock_dms_requests.post.side_effect = lambda url, **kwargs: mock_dms_requests_side_effect(
            "post", url, **kwargs
        )
        mock_dms_requests.get.side_effect = lambda url, **kwargs: mock_dms_requests_side_effect(
            "get", url, **kwargs
        )

        # Mock requests responses based on URL
        def mock_requests_side_effect(method, url, **kwargs):
            if method == "post" and "/v1/evaluation/jobs" in url:
                return MagicMock(status_code=200, json=lambda: {"id": "eval-job-123"})
            elif method == "get" and "/v1/evaluation/jobs/eval-job-123/results" in url:
                return MagicMock(
                    status_code=200,
                    json=lambda: {
                        "tasks": {
                            "llm-as-judge": {
                                "metrics": {
                                    "llm-judge": {"scores": {"similarity": {"value": 0.85}}}
                                }
                            }
                        }
                    },
                )
            elif method == "get" and "/v1/evaluation/jobs/eval-job-123" in url:
                return MagicMock(
                    status_code=200,
                    json=lambda: {"status": "completed", "status_details": {"progress": 100}},
                )
            else:
                raise ValueError(f"Unexpected request: {method} {url}")

        mock_requests.post.side_effect = lambda url, **kwargs: mock_requests_side_effect(
            "post", url, **kwargs
        )
        mock_requests.get.side_effect = lambda url, **kwargs: mock_requests_side_effect(
            "get", url, **kwargs
        )

        # Mock customizer responses
        def mock_customizer_requests_side_effect(method, url, **kwargs):
            if method == "post" and "/v1/customization/jobs" in url:
                job_id = f"custom-job-{uuid.uuid4()}"
                return MagicMock(
                    status_code=200,
                    json=lambda: {"id": job_id, "output_model": output_model},
                )
            elif method == "get" and "/v1/customization/jobs/" in url and "/status" in url:
                return MagicMock(
                    status_code=200,
                    json=lambda: {
                        "status": "completed",
                        "percentage_done": 100,
                        "epochs_completed": 2,
                        "steps_completed": 100,
                        "status_logs": [],
                    },
                )
            elif method == "get" and "/v1/models/" in url:
                return MagicMock(
                    status_code=200,
                    json=lambda: {
                        "name": url.split("/")[-1],
                        "status": "ready",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            elif method == "get" and "/v1/models" in url:
                return MagicMock(
                    status_code=200,
                    json=lambda: {"data": [{"id": output_model}]},
                )
            else:
                raise ValueError(f"Unexpected customizer request: {method} {url}")

        mock_customizer_requests.post.side_effect = (
            lambda url, **kwargs: mock_customizer_requests_side_effect("post", url, **kwargs)
        )
        mock_customizer_requests.get.side_effect = (
            lambda url, **kwargs: mock_customizer_requests_side_effect("get", url, **kwargs)
        )

        # Mock upload_data to return the file path
        mock_upload_data.side_effect = lambda data, file_path: file_path
        mock_get_file_uri.side_effect = lambda: "test_uri"

        yield {
            "requests": mock_requests,
            "upload_data": mock_upload_data,
            "get_file_uri": mock_get_file_uri,
            "dms_requests": mock_dms_requests,
            "customizer_requests": mock_customizer_requests,
        }


@pytest.fixture
def cleanup_test_data():
    """Cleanup test data before and after tests"""
    db = get_db()

    # Cleanup before test
    db.flywheel_runs.drop()
    db.nims.drop()
    db.evaluations.drop()

    yield

    # Cleanup after test
    db.flywheel_runs.drop()
    db.nims.drop()
    db.evaluations.drop()


@pytest.mark.integration
def test_full_job_evaluation_flow(
    test_client: TestClient,
    mock_external_services: dict[str, MagicMock],
    cleanup_test_data,
    load_test_data_fixture,
):
    """Test the complete job evaluation flow from POST /jobs to completion"""
    # 1. Create a new job - since tasks run synchronously, it will complete immediately
    response = test_client.post(
        "/api/jobs", json={"workload_id": WORKLOAD_ID, "client_id": CLIENT_ID}
    )
    assert response.status_code == 200
    job_data = response.json()
    job_id = job_data["id"]

    # Job should be completed immediately since tasks run synchronously
    assert job_data["status"] == "queued"

    # Verify that there is only one job in the system
    response = test_client.get("/api/jobs")
    assert response.status_code == 200
    jobs_data = response.json()
    assert len(jobs_data["jobs"]) == 1
    assert jobs_data["jobs"][-1]["id"] == job_id

    # Hit the GET /jobs/:job_id endpoint to verify job details
    response = test_client.get(f"/api/jobs/{job_id}")
    assert response.status_code == 200
    job_details = response.json()

    # Verify the job details
    assert job_details["id"] == job_id
    assert job_details["workload_id"] == WORKLOAD_ID
    assert job_details["status"] in ["completed", "running"]
    assert job_details["started_at"] is not None
    assert job_details["num_records"] > 0

    # Verify NIMs and evaluations in job details
    assert len(job_details["nims"]) > 0
    for nim in job_details["nims"]:
        assert nim["model_name"] is not None
        assert len(nim["evaluations"]) > 0

        for eval in nim["evaluations"]:
            assert eval["eval_type"] in [EvalType.BASE, EvalType.ICL, EvalType.CUSTOMIZED]
            assert eval["scores"] is not None
            assert eval["progress"] == 100.0
            assert eval["started_at"] is not None
            assert eval["runtime_seconds"] > 0

    # 3. Verify database state
    db = get_db()

    # Verify flywheel run
    flywheel_run = db.flywheel_runs.find_one({"_id": ObjectId(job_id)})
    assert flywheel_run is not None
    assert flywheel_run["workload_id"] == WORKLOAD_ID
    assert flywheel_run["num_records"] > 0

    # Verify NIMs
    nims = list(db.nims.find({"flywheel_run_id": ObjectId(job_id)}))
    assert len(nims) > 0
    for nim in nims:
        assert nim["model_name"] is not None

        # Verify evaluations for this NIM
        evaluations = list(db.evaluations.find({"nim_id": nim["_id"]}))
        assert len(evaluations) > 0
        for eval in evaluations:
            assert eval["eval_type"] in [EvalType.BASE, EvalType.ICL, EvalType.CUSTOMIZED]
            assert eval["scores"] is not None
            assert eval["progress"] == 100.0

    # 4. Verify external service calls
    mock_requests = mock_external_services["requests"]

    # Verify evaluator calls
    assert mock_requests.post.called
    assert mock_requests.get.called
    assert mock_requests.post.call_count >= 3  # At least 3 calls: status checks and results
