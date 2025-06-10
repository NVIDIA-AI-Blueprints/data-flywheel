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
"""High-level repository for all database writes performed inside Celery tasks.

This indirection gives the tasks an intention-revealing API while
centralising MongoDB specifics in one place.  It also enables unit tests that
mock **one** narrow surface instead of many scattered ``db.<collection>``
calls.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from bson import ObjectId

from src.api.db import get_db, init_db
from src.api.models import (
    FlywheelRunStatus,
    LLMJudgeRun,
    NIMCustomization,
    NIMEvaluation,
    NIMRun,
    NIMRunStatus,
)
from src.api.schemas import DeploymentStatus

# NOTE: we only import the enum for typing -the collections still store raw
# dictionaries produced by ``model.to_mongo()`` so they stay backward
# compatible with existing data.

_db_manager: TaskDBManager | None = None


def get_db_manager() -> TaskDBManager:
    """Get the database manager instance, creating it if necessary."""
    global _db_manager
    if _db_manager is None:
        _db_manager = TaskDBManager()
    return _db_manager


class TaskDBManager:
    """Lightweight façade over Mongo collections used by Celery tasks."""

    def __init__(self) -> None:
        # Lazily resolve the singleton Database instance - init_db() must have
        # been called from the worker bootstrap code before any method is used.
        try:
            db = get_db()
        except RuntimeError:
            # Database not yet initialised (e.g. during unit-test import). Do it now.
            db = init_db()
        self._db = db
        self._flywheel_runs = db.flywheel_runs
        self._nims = db.nims
        self._evaluations = db.evaluations
        self._customizations = db.customizations
        self.llm_judge_runs = db.llm_judge_runs

    @property
    def db(self):
        """Access to the database instance for direct queries when needed."""
        return self._db

    # ---------------------------------------------------------------------
    # Flywheel-run helpers
    # ---------------------------------------------------------------------

    def update_flywheel_run_status(
        self, flywheel_run_id: str | ObjectId, status: FlywheelRunStatus
    ) -> None:
        """Update the status of a FlywheelRun document."""
        # Only update if there is no error on the flywheel run document.
        self._flywheel_runs.update_one(
            {"_id": ObjectId(flywheel_run_id), "error": None},
            {"$set": {"status": status}},
        )

    def mark_flywheel_run_completed(
        self, flywheel_run_id: str | ObjectId, finished_at: datetime
    ) -> None:
        """Mark a FlywheelRun as completed.
        Only update if there is no error on the flywheel run document.
        """
        self._flywheel_runs.update_one(
            {"_id": ObjectId(flywheel_run_id), "error": None},
            {"$set": {"finished_at": finished_at, "status": FlywheelRunStatus.COMPLETED}},
        )

    def mark_flywheel_run_error(
        self, flywheel_run_id: str | ObjectId, error_msg: str, finished_at: datetime | None = None
    ) -> None:
        """Save an *error* status onto a FlywheelRun document.

        Only update if there is no error on the flywheel run document.
        """
        self._flywheel_runs.update_one(
            {
                "_id": ObjectId(flywheel_run_id),
                "error": None,
            },
            {
                "$set": {
                    "error": error_msg,
                    "status": FlywheelRunStatus.FAILED,
                    "finished_at": finished_at,
                }
            },
        )

    def mark_flywheel_run_cancelled(
        self,
        flywheel_run_id: str | ObjectId,
        error_msg: str | None = None,
    ) -> None:
        """Mark a FlywheelRun as cancelled.

        Only update if there is no error on the flywheel run document.
        """
        update_fields = {"status": FlywheelRunStatus.CANCELLED}
        update_fields["finished_at"] = datetime.utcnow()
        if error_msg:
            update_fields["error"] = error_msg
        self._flywheel_runs.update_one(
            {"_id": ObjectId(flywheel_run_id), "finished_at": None},
            {"$set": update_fields},
        )

    def is_flywheel_run_cancelled(self, flywheel_run_id: str | ObjectId) -> bool:
        """Check if a FlywheelRun is cancelled."""
        doc = self._flywheel_runs.find_one({"_id": ObjectId(flywheel_run_id)}, {"status": 1})
        return doc and doc.get("status") == FlywheelRunStatus.CANCELLED

    # ---------------------------------------------------------------------
    # NIM-run helpers
    # ---------------------------------------------------------------------
    def create_nim_run(self, nim_run: NIMRun) -> ObjectId:
        """Insert a new NIMRun and return its ``ObjectId``."""
        result = self._nims.insert_one(nim_run.to_mongo())
        return result.inserted_id

    def set_nim_status(
        self,
        nim_id: ObjectId,
        status: NIMRunStatus,
        *,
        error: str | None = None,
        deployment_status: DeploymentStatus | None = None,
    ) -> None:
        """Update the status (and optional error/deployment info) of a NIM run."""
        update: dict[str, Any] = {"status": status}
        if error is not None:
            update["error"] = error
        if deployment_status is not None:
            update["deployment_status"] = deployment_status
        self._nims.update_one({"_id": nim_id}, {"$set": update})

    def update_nim_deployment_status(
        self, nim_id: ObjectId, deployment_status: DeploymentStatus, runtime_seconds: float
    ) -> None:
        """Update the deployment status of a NIM run."""
        self._nims.update_one(
            {"_id": nim_id},
            {"$set": {"deployment_status": deployment_status, "runtime_seconds": runtime_seconds}},
        )

    def mark_nim_completed(self, nim_id: ObjectId, started_at: datetime) -> None:
        """Mark a NIM run as completed once the deployment has been shut down.

        Only update if there is no error on the NIM run document.
        """
        finished_time = datetime.utcnow()
        runtime_seconds: float = 0.0
        if started_at:
            runtime_seconds = (finished_time - started_at).total_seconds()
        self._nims.update_one(
            {"_id": nim_id, "error": None},
            {
                "$set": {
                    "status": NIMRunStatus.COMPLETED,
                    "deployment_status": DeploymentStatus.COMPLETED,
                    "finished_at": finished_time,
                    "runtime_seconds": runtime_seconds,
                }
            },
        )

    def mark_nim_cancelled(self, nim_id: ObjectId, error_msg: str | None = None) -> None:
        """Mark a NIM run as cancelled.

        Only update if there is no error on the NIM run document.
        """
        finished_time = datetime.utcnow()
        self._nims.update_one(
            {"_id": nim_id, "error": None},
            {
                "$set": {
                    "status": NIMRunStatus.CANCELLED,
                    "deployment_status": DeploymentStatus.CANCELLED,
                    "finished_at": finished_time,
                    "error": error_msg,
                }
            },
        )

    def mark_nim_error(self, nim_id: ObjectId, error_msg: str) -> None:
        """Mark a NIM run as failed.

        Only update if there is no error on the NIM run document.
        """
        self._nims.update_one(
            {
                "_id": nim_id,
                "error": None,  # Only match documents where error is None
            },
            {
                "$set": {
                    "error": error_msg,
                    "status": NIMRunStatus.FAILED,
                    "deployment_status": DeploymentStatus.FAILED,
                    "finished_at": datetime.utcnow(),
                }
            },
        )

    def find_nim_run(self, flywheel_run_id: str, model_name: str) -> Mapping[str, Any] | None:
        """Find a NIM run by its associated flywheel run ID and model name."""
        return self._nims.find_one(
            {"flywheel_run_id": ObjectId(flywheel_run_id), "model_name": model_name}
        )

    def mark_all_nims_status(
        self, flywheel_run_id: str | ObjectId, status: NIMRunStatus, error_msg: str | None = None
    ) -> None:
        """Mark all NIMs associated with a flywheel run as failed.

        Only update if there is no error on the NIM run document.
        Args:
            flywheel_run_id: ID of the flywheel run whose NIMs should be marked as failed
            error_msg: Error message to set on the NIMs
        """
        self._nims.update_many(
            {
                "flywheel_run_id": ObjectId(flywheel_run_id),
                "error": None,  # Only update NIMs that don't already have an error
            },
            {
                "$set": {
                    "error": error_msg,
                    "status": status,
                    "finished_at": datetime.utcnow(),
                }
            },
        )

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def insert_evaluation(self, evaluation: NIMEvaluation) -> ObjectId:
        """Insert a new NIMEvaluation and return its ObjectId."""
        self._evaluations.insert_one(evaluation.to_mongo())
        return evaluation.id  # type: ignore[return-value]

    def update_evaluation(self, eval_id: ObjectId, update_fields: dict[str, Any]) -> None:
        """Update a NIMEvaluation."""
        self._evaluations.update_one({"_id": eval_id}, {"$set": update_fields})

    # ------------------------------------------------------------------
    # Customization helpers
    # ------------------------------------------------------------------
    def insert_customization(self, customization: NIMCustomization) -> ObjectId:
        self._customizations.insert_one(customization.to_mongo())
        return customization.id  # type: ignore[return-value]

    def update_customization(self, custom_id: ObjectId, update_fields: dict[str, Any]) -> None:
        """Update a NIMCustomization."""
        self._customizations.update_one({"_id": custom_id}, {"$set": update_fields})

    def find_customization(self, workload_id: str, model_name: str):
        """Find a NIMCustomization by its workload ID and model name."""
        return self._customizations.find_one(
            {"workload_id": workload_id, "customized_model": model_name}
        )

    # ------------------------------------------------------------------
    # LLM Judge helpers
    # ------------------------------------------------------------------
    def create_llm_judge_run(self, llm_judge_run: LLMJudgeRun) -> ObjectId:
        """Insert a new LLMJudgeRun and return its ObjectId."""
        result = self.llm_judge_runs.insert_one(llm_judge_run.to_mongo())
        return result.inserted_id

    def update_llm_judge_deployment_status(
        self, llm_judge_id: ObjectId, deployment_status: DeploymentStatus
    ) -> None:
        """Update the deployment status of an LLM judge run."""
        self.llm_judge_runs.update_one(
            {"_id": llm_judge_id},
            {"$set": {"deployment_status": deployment_status}},
        )

    def mark_llm_judge_error(self, llm_judge_id: ObjectId, error_msg: str) -> None:
        """Mark an LLM judge run as having encountered an error."""
        self.llm_judge_runs.update_one(
            {"_id": llm_judge_id},
            {
                "$set": {
                    "error": error_msg,
                    "deployment_status": DeploymentStatus.FAILED,
                }
            },
        )

    def find_llm_judge_run(self, flywheel_run_id: str | ObjectId) -> Mapping[str, Any] | None:
        """Find an LLM judge run by its associated flywheel run ID."""
        return self.llm_judge_runs.find_one({"flywheel_run_id": ObjectId(flywheel_run_id)})

    # ------------------------------------------------------------------
    # Job deletion helpers
    # ------------------------------------------------------------------
    def get_flywheel_run(self, job_id: str | ObjectId) -> Mapping[str, Any] | None:
        """Get a flywheel run by ID.

        Args:
            job_id: ID of the flywheel run to retrieve

        Returns:
            The flywheel run document if found, None otherwise
        """
        return self._flywheel_runs.find_one({"_id": ObjectId(job_id)})

    def find_running_flywheel_runs(self) -> list[Mapping[str, Any]]:
        """Find all flywheel runs that are currently running.

        Returns:
            List of flywheel run documents with PENDING or RUNNING status
        """
        running_statuses = [FlywheelRunStatus.PENDING.value, FlywheelRunStatus.RUNNING.value]
        return list(self._flywheel_runs.find({"status": {"$in": running_statuses}}))

    def find_running_nims_for_flywheel(self, flywheel_run_id: ObjectId) -> list[Mapping[str, Any]]:
        """Find all NIMs with RUNNING or PENDING status for a flywheel run.

        Args:
            flywheel_run_id: ID of the flywheel run to find NIMs for

        Returns:
            List of NIM documents with RUNNING or PENDING status
        """
        return list(
            self._nims.find(
                {
                    "flywheel_run_id": flywheel_run_id,
                    "status": {
                        "$in": [
                            NIMRunStatus.RUNNING.value,
                            NIMRunStatus.PENDING.value,
                        ]
                    },
                }
            )
        )

    def mark_llm_judge_cancelled(
        self, flywheel_run_id: ObjectId, error_msg: str | None = None
    ) -> None:
        """Mark an LLM judge run as cancelled.

        Args:
            flywheel_run_id: ID of the flywheel run associated with the LLM judge
            error_msg: Optional error message to set
        """
        update_fields = {"deployment_status": DeploymentStatus.CANCELLED.value}
        if error_msg:
            update_fields["error"] = error_msg

        self.llm_judge_runs.update_one(
            {"flywheel_run_id": flywheel_run_id},
            {"$set": update_fields},
        )

    def find_nims_for_job(self, job_id: ObjectId) -> list[Mapping[str, Any]]:
        """Find all NIMs for a given job.

        Args:
            job_id: ID of the job to find NIMs for

        Returns:
            List of NIM documents associated with the job
        """
        return list(self._nims.find({"flywheel_run_id": job_id}))

    def find_customizations_for_nim(self, nim_id: ObjectId) -> list[Mapping[str, Any]]:
        """Find all customizations for a given NIM.

        Args:
            nim_id: ID of the NIM to find customizations for

        Returns:
            List of customization documents associated with the NIM
        """
        return list(self._customizations.find({"nim_id": nim_id}))

    def find_evaluations_for_nim(self, nim_id: ObjectId) -> list[Mapping[str, Any]]:
        """Find all evaluations for a given NIM.

        Args:
            nim_id: ID of the NIM to find evaluations for

        Returns:
            List of evaluation documents associated with the NIM
        """
        return list(self._evaluations.find({"nim_id": nim_id}))

    def delete_job_records(self, job_id: ObjectId) -> None:
        """Delete all MongoDB records related to a job.

        This deletes all associated records in the following collections:
        - evaluations
        - customizations
        - nims
        - llm_judge_runs
        - flywheel_runs

        Args:
            job_id: ID of the job to delete records for
        """
        self._evaluations.delete_many({"flywheel_run_id": job_id})
        self._customizations.delete_many({"flywheel_run_id": job_id})
        self._nims.delete_many({"flywheel_run_id": job_id})
        self.llm_judge_runs.delete_many({"flywheel_run_id": job_id})
        self._flywheel_runs.delete_one({"_id": job_id})
