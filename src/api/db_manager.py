# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
        self._flywheel_runs = db.flywheel_runs
        self._nims = db.nims
        self._evaluations = db.evaluations
        self._customizations = db.customizations
        self.llm_judge_runs = db.llm_judge_runs

    # ---------------------------------------------------------------------
    # Flywheel-run helpers
    # ---------------------------------------------------------------------
    def mark_flywheel_run_completed(
        self, flywheel_run_id: str | ObjectId, finished_at: datetime
    ) -> None:
        """Save the *finished_at* timestamp on a FlywheelRun document.

        The calling code is responsible for deciding the appropriate
        *finished_at* value (typically the current UTC time).  We keep the
        logic here so Celery tasks do not need to know collection names or
        MongoDB update syntax.
        """
        self._flywheel_runs.update_one(
            {"_id": ObjectId(flywheel_run_id)},
            {"$set": {"finished_at": finished_at}},
        )

    def mark_flywheel_run_error(self, flywheel_run_id: str | ObjectId, error_msg: str) -> None:
        """Save an *error* status onto a FlywheelRun document."""
        self._flywheel_runs.update_one(
            {"_id": ObjectId(flywheel_run_id)},
            {"$set": {"error": error_msg, "status": "error"}},
        )

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
        self, nim_id: ObjectId, deployment_status: DeploymentStatus
    ) -> None:
        self._nims.update_one({"_id": nim_id}, {"$set": {"deployment_status": deployment_status}})

    def mark_nim_completed(self, nim_id: ObjectId, finished_at: datetime, runtime_s: float) -> None:
        """Mark a NIM run as completed once the deployment has been shut down."""
        self._nims.update_one(
            {"_id": nim_id},
            {
                "$set": {
                    "status": NIMRunStatus.COMPLETED,
                    "deployment_status": DeploymentStatus.COMPLETED,
                    "finished_at": finished_at,
                    "runtime_seconds": runtime_s,
                }
            },
        )

    def mark_nim_error(self, nim_id: ObjectId, error_msg: str) -> None:
        self._nims.update_one(
            {"_id": nim_id},
            {
                "$set": {
                    "error": error_msg,
                    "status": NIMRunStatus.ERROR,
                    "deployment_status": DeploymentStatus.FAILED,
                }
            },
        )

    def find_nim_run(self, flywheel_run_id: str, model_name: str) -> Mapping[str, Any] | None:
        return self._nims.find_one(
            {"flywheel_run_id": ObjectId(flywheel_run_id), "model_name": model_name}
        )

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def insert_evaluation(self, evaluation: NIMEvaluation) -> ObjectId:
        self._evaluations.insert_one(evaluation.to_mongo())
        return evaluation.id  # type: ignore[return-value]

    def update_evaluation(self, eval_id: ObjectId, update_fields: dict[str, Any]) -> None:
        self._evaluations.update_one({"_id": eval_id}, {"$set": update_fields})

    # ------------------------------------------------------------------
    # Customization helpers
    # ------------------------------------------------------------------
    def insert_customization(self, customization: NIMCustomization) -> ObjectId:
        self._customizations.insert_one(customization.to_mongo())
        return customization.id  # type: ignore[return-value]

    def update_customization(self, custom_id: ObjectId, update_fields: dict[str, Any]) -> None:
        self._customizations.update_one({"_id": custom_id}, {"$set": update_fields})

    def find_customization(self, workload_id: str, model_name: str):
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
            {"_id": llm_judge_id}, {"$set": {"deployment_status": deployment_status}}
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
