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
from unittest.mock import ANY, Mock, patch

import pytest
from bson import ObjectId

from src.api.db_manager import TaskDBManager
from src.api.models import (
    EvalType,
    FlywheelRunStatus,
    LLMJudgeRun,
    NIMCustomization,
    NIMEvaluation,
    NIMRun,
    NIMRunStatus,
)
from src.api.schemas import DeploymentStatus


@pytest.fixture
def db_manager():
    with patch("src.api.db_manager.get_db") as mock_get_db:
        # Create mock collections
        mock_db = Mock()
        mock_db.flywheel_runs = Mock()
        mock_db.nims = Mock()
        mock_db.evaluations = Mock()
        mock_db.customizations = Mock()
        mock_db.llm_judge_runs = Mock()

        mock_get_db.return_value = mock_db
        return TaskDBManager()


class TestTaskDBManager:
    def test_mark_flywheel_run_completed(self, db_manager):
        run_id = "507f1f77bcf86cd799439011"
        finished_at = datetime.utcnow()

        db_manager.mark_flywheel_run_completed(run_id, finished_at)

        db_manager._flywheel_runs.update_one.assert_called_once_with(
            {"_id": ObjectId(run_id), "error": None},
            {"$set": {"finished_at": finished_at, "status": FlywheelRunStatus.COMPLETED}},
        )

    def test_mark_flywheel_run_error(self, db_manager):
        run_id = "507f1f77bcf86cd799439011"
        error_msg = "Test error"
        finished_at = datetime.utcnow()
        db_manager.mark_flywheel_run_error(run_id, error_msg, finished_at=finished_at)

        db_manager._flywheel_runs.update_one.assert_called_once_with(
            {"_id": ObjectId(run_id), "error": None},
            {
                "$set": {
                    "error": error_msg,
                    "status": FlywheelRunStatus.FAILED,
                    "finished_at": finished_at,
                }
            },
        )

    def test_create_nim_run(self, db_manager):
        nim_run = NIMRun(
            flywheel_run_id=ObjectId(),
            model_name="test-model",
            started_at=datetime.utcnow(),
            runtime_seconds=0.0,
            status=NIMRunStatus.RUNNING,
        )
        mock_result = Mock()
        mock_result.inserted_id = ObjectId()
        db_manager._nims.insert_one.return_value = mock_result

        result = db_manager.create_nim_run(nim_run)

        assert isinstance(result, ObjectId)
        db_manager._nims.insert_one.assert_called_once()

    def test_set_nim_status(self, db_manager):
        nim_id = ObjectId()
        status = NIMRunStatus.RUNNING

        # Test without optional parameters
        db_manager.set_nim_status(nim_id, status)
        db_manager._nims.update_one.assert_called_with(
            {"_id": nim_id}, {"$set": {"status": status}}
        )

        # Test with error
        db_manager.set_nim_status(nim_id, status, error="Test error")
        db_manager._nims.update_one.assert_called_with(
            {"_id": nim_id}, {"$set": {"status": status, "error": "Test error"}}
        )

        # Test with deployment status
        db_manager.set_nim_status(nim_id, status, deployment_status=DeploymentStatus.RUNNING)
        db_manager._nims.update_one.assert_called_with(
            {"_id": nim_id},
            {"$set": {"status": status, "deployment_status": DeploymentStatus.RUNNING}},
        )

    def test_update_nim_deployment_status(self, db_manager):
        nim_id = ObjectId()
        deployment_status = DeploymentStatus.RUNNING
        runtime_seconds = 123.45

        db_manager.update_nim_deployment_status(nim_id, deployment_status, runtime_seconds)

        db_manager._nims.update_one.assert_called_once_with(
            {"_id": nim_id},
            {"$set": {"deployment_status": deployment_status, "runtime_seconds": runtime_seconds}},
        )

    def test_mark_nim_completed(self, db_manager):
        nim_id = ObjectId()
        started_at = datetime.utcnow()

        with patch("src.api.db_manager.datetime") as mock_datetime:
            mock_datetime.utcnow.return_value = started_at

            db_manager.mark_nim_completed(nim_id, started_at)

            db_manager._nims.update_one.assert_called_once()

    def test_mark_nim_error(self, db_manager):
        nim_id = ObjectId()
        error_msg = "Test error"

        # Setup initial database state - document exists with no error
        initial_doc = {"_id": nim_id, "error": None, "status": NIMRunStatus.RUNNING}

        # Setup final state after update
        final_doc = {
            "_id": nim_id,
            "error": error_msg,
            "status": NIMRunStatus.FAILED,
            "deployment_status": DeploymentStatus.FAILED,
        }

        # Mock find_one to return different results before and after update
        db_manager._nims.find_one.side_effect = [initial_doc, final_doc]

        # Get state before update
        before_state = db_manager._nims.find_one({"_id": nim_id})
        assert before_state["error"] is None

        db_manager.mark_nim_error(nim_id, error_msg)

        # Verify the update query was correct
        db_manager._nims.update_one.assert_called_once_with(
            {
                "_id": nim_id,
                "error": None,  # Only update if error is None
            },
            {
                "$set": {
                    "error": error_msg,
                    "status": NIMRunStatus.FAILED,
                    "deployment_status": DeploymentStatus.FAILED,
                    "finished_at": ANY,
                }
            },
        )

        # Get state after update
        after_state = db_manager._nims.find_one({"_id": nim_id})

        # Verify the document was actually modified
        assert after_state["error"] == error_msg
        assert after_state["status"] == NIMRunStatus.FAILED
        assert after_state["deployment_status"] == DeploymentStatus.FAILED

    def test_mark_nim_error_when_error_exists(self, db_manager):
        nim_id = ObjectId()
        existing_error = "Existing error"
        new_error_msg = "New error"

        # Setup initial database state - document exists with error
        doc_with_error = {
            "_id": nim_id,
            "error": existing_error,
            "status": NIMRunStatus.FAILED,
            "deployment_status": DeploymentStatus.FAILED,
        }

        # Mock find_one to return same document before and after (no change)
        db_manager._nims.find_one.return_value = doc_with_error

        # Get state before update
        before_state = db_manager._nims.find_one({"_id": nim_id})
        assert before_state["error"] == existing_error

        db_manager.mark_nim_error(nim_id, new_error_msg)

        # Verify the update was attempted with correct query
        db_manager._nims.update_one.assert_called_once_with(
            {
                "_id": nim_id,
                "error": None,  # This ensures we only update if no error exists
            },
            {
                "$set": {
                    "error": new_error_msg,
                    "status": NIMRunStatus.FAILED,
                    "deployment_status": DeploymentStatus.FAILED,
                    "finished_at": ANY,
                }
            },
        )

        # Get state after update attempt
        after_state = db_manager._nims.find_one({"_id": nim_id})

        # Verify that document was not modified
        assert after_state["error"] == existing_error  # Error should remain unchanged
        assert after_state["status"] == NIMRunStatus.FAILED
        assert after_state["deployment_status"] == DeploymentStatus.FAILED

    def test_find_nim_run(self, db_manager):
        flywheel_run_id = "507f1f77bcf86cd799439011"
        model_name = "test-model"
        expected_result = {"_id": ObjectId(), "model_name": model_name}
        db_manager._nims.find_one.return_value = expected_result

        result = db_manager.find_nim_run(flywheel_run_id, model_name)

        assert result == expected_result
        db_manager._nims.find_one.assert_called_once_with(
            {"flywheel_run_id": ObjectId(flywheel_run_id), "model_name": model_name}
        )

    def test_insert_evaluation(self, db_manager):
        evaluation = NIMEvaluation(
            flywheel_run_id=ObjectId(),
            nim_id=ObjectId(),
            model_name="test-model",
            eval_type=EvalType.BASE,
            scores={"accuracy": 0.95},
            started_at=datetime.utcnow(),
            runtime_seconds=0.0,
            progress=0.0,
        )

        db_manager.insert_evaluation(evaluation)

        db_manager._evaluations.insert_one.assert_called_once()

    def test_update_evaluation(self, db_manager):
        eval_id = ObjectId()
        update_fields = {"status": "completed"}

        db_manager.update_evaluation(eval_id, update_fields)

        db_manager._evaluations.update_one.assert_called_once_with(
            {"_id": eval_id}, {"$set": update_fields}
        )

    def test_insert_customization(self, db_manager):
        customization = NIMCustomization(
            flywheel_run_id=ObjectId(),
            nim_id=ObjectId(),
            customized_model="test-model",
            workload_id="test-workload",
            base_model="base-model",
            started_at=datetime.utcnow(),
        )

        db_manager.insert_customization(customization)

        db_manager._customizations.insert_one.assert_called_once()

    def test_update_customization(self, db_manager):
        custom_id = ObjectId()
        update_fields = {"status": "completed"}

        db_manager.update_customization(custom_id, update_fields)

        db_manager._customizations.update_one.assert_called_once_with(
            {"_id": custom_id}, {"$set": update_fields}
        )

    def test_find_customization(self, db_manager):
        workload_id = "test-workload"
        model_name = "test-model"
        expected_result = {"_id": ObjectId(), "model_name": model_name}
        db_manager._customizations.find_one.return_value = expected_result

        result = db_manager.find_customization(workload_id, model_name)

        assert result == expected_result
        db_manager._customizations.find_one.assert_called_once_with(
            {"workload_id": workload_id, "customized_model": model_name}
        )

    def test_create_llm_judge_run(self, db_manager):
        llm_judge_run = LLMJudgeRun(
            flywheel_run_id=ObjectId(), model_name="test-model", type="remote"
        )
        mock_result = Mock()
        mock_result.inserted_id = ObjectId()
        db_manager.llm_judge_runs.insert_one.return_value = mock_result

        result = db_manager.create_llm_judge_run(llm_judge_run)

        assert isinstance(result, ObjectId)
        db_manager.llm_judge_runs.insert_one.assert_called_once()

    def test_update_llm_judge_deployment_status(self, db_manager):
        llm_judge_id = ObjectId()
        deployment_status = DeploymentStatus.RUNNING

        db_manager.update_llm_judge_deployment_status(llm_judge_id, deployment_status)

        db_manager.llm_judge_runs.update_one.assert_called_once_with(
            {"_id": llm_judge_id}, {"$set": {"deployment_status": deployment_status}}
        )

    def test_mark_llm_judge_error(self, db_manager):
        llm_judge_id = ObjectId()
        error_msg = "Test error"

        db_manager.mark_llm_judge_error(llm_judge_id, error_msg)

        db_manager.llm_judge_runs.update_one.assert_called_once_with(
            {"_id": llm_judge_id},
            {
                "$set": {
                    "error": error_msg,
                    "deployment_status": DeploymentStatus.FAILED,
                }
            },
        )

    def test_find_llm_judge_run(self, db_manager):
        flywheel_run_id = "507f1f77bcf86cd799439011"
        expected_result = {"_id": ObjectId(), "model_name": "test-model"}
        db_manager.llm_judge_runs.find_one.return_value = expected_result

        result = db_manager.find_llm_judge_run(flywheel_run_id)

        assert result == expected_result
        db_manager.llm_judge_runs.find_one.assert_called_once_with(
            {"flywheel_run_id": ObjectId(flywheel_run_id)}
        )

    def test_get_flywheel_run(self, db_manager):
        job_id = "507f1f77bcf86cd799439011"
        expected_result = {"_id": ObjectId(job_id)}
        db_manager._flywheel_runs.find_one.return_value = expected_result

        result = db_manager.get_flywheel_run(job_id)

        assert result == expected_result
        db_manager._flywheel_runs.find_one.assert_called_once_with({"_id": ObjectId(job_id)})

    def test_find_nims_for_job(self, db_manager):
        job_id = ObjectId()
        expected_results = [{"_id": ObjectId(), "model_name": "test-model"}]
        db_manager._nims.find.return_value = expected_results

        results = db_manager.find_nims_for_job(job_id)

        assert results == expected_results
        db_manager._nims.find.assert_called_once_with({"flywheel_run_id": job_id})

    def test_find_customizations_for_nim(self, db_manager):
        nim_id = ObjectId()
        expected_results = [{"_id": ObjectId(), "model_name": "test-model"}]
        db_manager._customizations.find.return_value = expected_results

        results = db_manager.find_customizations_for_nim(nim_id)

        assert results == expected_results
        db_manager._customizations.find.assert_called_once_with({"nim_id": nim_id})

    def test_find_evaluations_for_nim(self, db_manager):
        nim_id = ObjectId()
        expected_results = [{"_id": ObjectId(), "model_name": "test-model"}]
        db_manager._evaluations.find.return_value = expected_results

        results = db_manager.find_evaluations_for_nim(nim_id)

        assert results == expected_results
        db_manager._evaluations.find.assert_called_once_with({"nim_id": nim_id})

    def test_delete_job_records(self, db_manager):
        job_id = ObjectId()

        db_manager.delete_job_records(job_id)

        # Verify all delete operations were called
        db_manager._evaluations.delete_many.assert_called_once_with({"flywheel_run_id": job_id})
        db_manager._customizations.delete_many.assert_called_once_with({"flywheel_run_id": job_id})
        db_manager._nims.delete_many.assert_called_once_with({"flywheel_run_id": job_id})
        db_manager.llm_judge_runs.delete_many.assert_called_once_with({"flywheel_run_id": job_id})
        db_manager._flywheel_runs.delete_one.assert_called_once_with({"_id": job_id})

    def test_init_with_uninitialized_db(self, db_manager):
        with (
            patch("src.api.db_manager.get_db", side_effect=RuntimeError),
            patch("src.api.db_manager.init_db") as mock_init_db,
        ):
            mock_db = Mock()
            mock_db.flywheel_runs = Mock()
            mock_db.nims = Mock()
            mock_db.evaluations = Mock()
            mock_db.customizations = Mock()
            mock_db.llm_judge_runs = Mock()
            mock_init_db.return_value = mock_db

            # This should not raise an exception
            TaskDBManager()

            # Verify init_db was called
            mock_init_db.assert_called_once()
