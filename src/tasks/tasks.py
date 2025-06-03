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
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any, Optional, List

from bson import ObjectId
from celery import Celery, chain, group, signals

from src.api.db import init_db
from src.api.db_manager import TaskDBManager
from src.api.models import (
    DatasetType,
    EvalType,
    EvaluationResult,
    LLMJudgeRun,
    NIMCustomization,
    NIMEvaluation,
    NIMRun,
    NIMRunStatus,
    TaskResult,
    ToolEvalType,
    WorkloadClassification,
)
from src.api.schemas import DeploymentStatus
from src.config import NIMConfig, settings
from src.lib.flywheel.util import (
    identify_workload_type,
)
from src.lib.integration.dataset_creator import DatasetCreator
from src.lib.integration.record_exporter_es import ElasticsearchRecordExporter
from src.lib.integration.record_exporter_weave import WeaveRecordExporter
from src.lib.nemo.customizer import Customizer
from src.lib.nemo.dms_client import DMSClient
from src.lib.nemo.evaluator import Evaluator
from src.log_utils import setup_logging

logger = setup_logging("data_flywheel.tasks")

# Centralised DB helper - keeps Mongo specifics out of individual tasks
db_manager = TaskDBManager()

# Initialize Celery
celery_app = Celery(
    "llm_api",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)


@signals.worker_process_init.connect
def init_worker(**kwargs):
    """Initialize database connection after worker process is forked."""
    init_db()


@celery_app.task(name="tasks.create_datasets", pydantic=True)
def create_datasets(
    workload_id: str,
    flywheel_run_id: str,
    client_id: str,
    output_dataset_prefix: str = "",
    from_weave: bool = True,
    weave_op_names: Optional[List[str]] = None,
) -> TaskResult:
    """Pull data from Elasticsearch and create train/val/eval datasets.

    This function:
    1. Retrieves data from Elasticsearch for the given workload
    2. Splits, validates and uploads the data into evaluation, training, and validation sets based on split_config

    Args:
        workload_id: Unique identifier for this workload
        flywheel_run_id: ID of the FlywheelRun document
        client_id: ID of the client
        output_dataset_prefix: Optional prefix for dataset names
        from_weave: Whether to pull data from Weave or Elasticsearch
        weave_op_names: Optional list of operation names to filter records
    """
    try:
        if from_weave:
            records = WeaveRecordExporter(op_names=weave_op_names).get_records(client_id, workload_id)
        else:
            records = ElasticsearchRecordExporter().get_records(client_id, workload_id)
        
        logger.info(f"Example record: {records[0]}")

        workload_type = identify_workload_type(records)
        logger.info(f"Workload type: {workload_type}")

        datasets = DatasetCreator(
            records, flywheel_run_id, output_dataset_prefix, workload_id
        ).create_datasets()

        return TaskResult(
            workload_id=workload_id,
            flywheel_run_id=flywheel_run_id,
            client_id=client_id,
            workload_type=workload_type,
            datasets=datasets,
        )
    except Exception as e:
        error_msg = f"Error creating datasets: {e!s}"
        logger.error(error_msg)
        # Update flywheel run with error via the DB manager
        db_manager.mark_flywheel_run_error(flywheel_run_id, error_msg)
        # Return a TaskResult so that downstream tasks can gracefully short-circuit
        raise e


@celery_app.task(name="tasks.wait_for_llm_as_judge", pydantic=True)
def wait_for_llm_as_judge(previous_result: TaskResult) -> TaskResult:
    """
    Ensures LLM as judge is ready
    Takes the result from the previous task as input.
    """
    # This is a quirk of celery, we need to assert the types here
    # https://github.com/celery/celery/blob/main/examples/pydantic/tasks.py
    assert isinstance(previous_result, TaskResult)
    judge_cfg = settings.llm_judge_config
    if judge_cfg.is_remote():
        logger.info("Remote LLM Judge will be used")
        previous_result.llm_judge_config = None
        return previous_result

    llm_judge_config = judge_cfg.get_local_nim_config()
    previous_result.llm_judge_config = llm_judge_config

    # Create LLM judge run using TaskDBManager
    llm_judge_run = LLMJudgeRun(
        flywheel_run_id=ObjectId(previous_result.flywheel_run_id),
        model_name=llm_judge_config.model_name,
    )

    # Insert using TaskDBManager
    llm_judge_run.id = db_manager.create_llm_judge_run(llm_judge_run)

    dms_client = DMSClient(nmp_config=settings.nmp_config, nim=llm_judge_config)

    def progress_callback(status: dict):
        db_manager.update_llm_judge_deployment_status(
            llm_judge_run.id,
            DeploymentStatus(status.get("status", "unknown")),
        )

    dms_client.wait_for_deployment(progress_callback=progress_callback)
    dms_client.wait_for_model_sync(llm_judge_config.target_model_for_evaluation())

    return previous_result


@celery_app.task(name="tasks.spin_up_nim", pydantic=True)
def spin_up_nim(previous_result: TaskResult, nim_config: dict) -> TaskResult:
    """
    Spin up a NIM instance.
    Takes the result from the previous task as input.

    Args:
        previous_result: Result from the previous task
        nim_config: Full NIM configuration including model_name, context_length, etc.
    """

    # This is a quirk of celery, we need to assert the types here
    # https://github.com/celery/celery/blob/main/examples/pydantic/tasks.py
    assert isinstance(previous_result, TaskResult)

    ## reset previous_result.error as new nim starts
    previous_result.error = None

    nim_config = NIMConfig(**nim_config)
    previous_result.nim = nim_config

    # Create NIM run in the nims collection
    start_time = datetime.utcnow()
    nim_run = NIMRun(
        flywheel_run_id=ObjectId(previous_result.flywheel_run_id),
        model_name=nim_config.model_name,
        evaluations=[],
        started_at=start_time,
        finished_at=start_time,  # Will be updated when evaluations complete
        runtime_seconds=0,  # Will be updated when evaluations complete
    )

    # Persist and mark status via DB manager
    nim_run.id = db_manager.create_nim_run(nim_run)
    db_manager.set_nim_status(nim_run.id, NIMRunStatus.DEPLOYING)

    try:
        dms_client = DMSClient(nmp_config=settings.nmp_config, nim=nim_config)

        if not dms_client.is_deployed():
            logger.info(f"Deploying NIM {nim_config.model_name}")

            try:
                dms_client.deploy_model()
            except Exception as e:
                logger.error(f"Error deploying NIM {nim_config.model_name}: {e}")
                db_manager.set_nim_status(nim_run.id, NIMRunStatus.ERROR, error=str(e))
                previous_result.error = str(e)
                return previous_result
        else:
            logger.info(f"NIM {nim_config.model_name} is already deployed")

        def progress_callback(status: dict):
            db_manager.update_nim_deployment_status(
                nim_run.id,
                DeploymentStatus(status.get("status", "unknown")),
            )

        dms_client.wait_for_deployment(progress_callback=progress_callback)

        dms_client.wait_for_model_sync(nim_config.target_model_for_evaluation())

        db_manager.set_nim_status(nim_run.id, NIMRunStatus.RUNNING)

        return previous_result
    except Exception as e:
        error_msg = f"Error spinning up NIM: {e!s}"
        logger.error(error_msg)
        # Persist error on NIM run
        db_manager.set_nim_status(
            nim_run.id,
            NIMRunStatus.ERROR,
            error=error_msg,
            deployment_status=DeploymentStatus.FAILED,
        )
        dms_client.shutdown_deployment()
        previous_result.error = error_msg
        return previous_result


@celery_app.task(name="tasks.run_base_eval", pydantic=True)
def run_base_eval(previous_result: TaskResult) -> TaskResult:
    return run_generic_eval(previous_result, EvalType.BASE, DatasetType.BASE)


@celery_app.task(name="tasks.run_icl_eval", pydantic=True)
def run_icl_eval(previous_result: TaskResult) -> TaskResult:
    return run_generic_eval(previous_result, EvalType.ICL, DatasetType.ICL)


def run_generic_eval(
    previous_result: TaskResult, eval_type: EvalType, dataset_type: DatasetType
) -> TaskResult:
    """
    Run the Base/ICL/Customization evaluation against the NIM based on the eval_type.
    Takes the NIM details from the previous task.
    """
    if _should_skip_stage(previous_result, f"run_{eval_type}_eval"):
        return previous_result

    logger.info(f"Running {eval_type} evaluation")
    evaluator = Evaluator(llm_judge_config=previous_result.llm_judge_config)
    start_time = datetime.utcnow()

    tool_eval_types = [None]
    if previous_result.workload_type == WorkloadClassification.TOOL_CALLING:
        tool_eval_types = [ToolEvalType.TOOL_CALLING_METRIC]  # , ToolEvalType.TOOL_CALLING_JUDGE]

    jobs: list[dict[str, Any]] = []

    for tool_eval_type in tool_eval_types:
        # Find the NIM run for this model
        nim_run = db_manager.find_nim_run(
            previous_result.flywheel_run_id,
            previous_result.nim.model_name,
        )
        if not nim_run:
            msg = f"No NIM run found for model {previous_result.nim.model_name}"
            logger.error(msg)
            raise ValueError(msg)

        # Create evaluation document first
        evaluation = NIMEvaluation(
            nim_id=nim_run["_id"],
            eval_type=eval_type,
            scores={},  # Will be updated when evaluation completes
            started_at=start_time,
            finished_at=None,  # Will be updated when evaluation completes
            runtime_seconds=0.0,  # Will be updated when evaluation completes
            progress=0.0,  # Will be updated during evaluation
        )

        # Add evaluation to the database
        db_manager.insert_evaluation(evaluation)

        # Fix: Create closure with bound variables
        def make_progress_callback(manager: TaskDBManager, eval_instance):
            def callback(update_data):
                """Update evaluation document with progress"""
                manager.update_evaluation(eval_instance.id, update_data)

            return callback

        # Create callback with properly bound variables
        progress_callback = make_progress_callback(db_manager, evaluation)

        # Run the evaluation
        try:
            # Use customized model name for customization evaluation
            target_model = (
                previous_result.customization.model_name
                if eval_type == EvalType.CUSTOMIZED
                else previous_result.nim.target_model_for_evaluation()
            )

            job_id = evaluator.run_evaluation(
                namespace=settings.nmp_config.nmp_namespace,
                dataset_name=previous_result.datasets[dataset_type],
                workload_type=previous_result.workload_type,
                target_model=target_model,  # Use the selected target model
                test_file="eval_data.jsonl",
                tool_eval_type=tool_eval_type,
                limit=settings.data_split_config.limit,
            )
            logger.info("Evaluation job id: %s", job_id)

            # update uri in evaluation
            evaluation.nmp_uri = evaluator.get_job_uri(job_id)
            progress_callback({"nmp_uri": evaluation.nmp_uri})

            jobs.append(
                {
                    "job_id": job_id,
                    "evaluation": evaluation,
                    "progress_callback": progress_callback,
                    "tool_eval_type": tool_eval_type,
                }
            )
        except Exception as e:
            error_msg = f"Error running {eval_type} evaluation: {e!s}"
            logger.error(error_msg)
            db_manager.update_evaluation(
                evaluation.id,
                {
                    "error": error_msg,
                    "finished_at": datetime.utcnow(),
                    "progress": 0.0,
                },
            )
            previous_result.error = error_msg
            return previous_result

    for job in jobs:
        # Wait for completion with progress updates
        try:
            evaluator.wait_for_evaluation(
                job_id=job["job_id"],
                evaluation=job["evaluation"],
                polling_interval=5,
                timeout=3600,
                progress_callback=job["progress_callback"],
            )

            # Get final results
            results = evaluator.get_evaluation_results(job["job_id"])
            logger.info(results)

            # Update final results
            finished_time = datetime.utcnow()
            scores: dict[str, float] = {}
            if previous_result.workload_type == WorkloadClassification.TOOL_CALLING:
                if results["tasks"]["custom-tool-calling"]:
                    scores["function_name"] = results["tasks"]["custom-tool-calling"]["metrics"][
                        "tool-calling-accuracy"
                    ]["scores"]["function_name_accuracy"]["value"]
                    scores["function_name_and_args_accuracy"] = results["tasks"][
                        "custom-tool-calling"
                    ]["metrics"]["tool-calling-accuracy"]["scores"][
                        "function_name_and_args_accuracy"
                    ]["value"]

                if results["tasks"]["custom-tool-calling"]["metrics"]["correctness"]:
                    scores["tool_calling_correctness"] = results["tasks"]["custom-tool-calling"][
                        "metrics"
                    ]["correctness"]["scores"]["rating"]["value"]
            else:
                scores["similarity"] = results["tasks"]["llm-as-judge"]["metrics"]["llm-judge"][
                    "scores"
                ]["similarity"]["value"]

            job["progress_callback"](
                {
                    "scores": scores,
                    "finished_at": finished_time,
                    "runtime_seconds": (finished_time - start_time).total_seconds(),
                    "progress": 100.0,
                }
            )
            previous_result.add_evaluation(
                eval_type,
                EvaluationResult(
                    job_id=job["job_id"],
                    scores=scores,
                    started_at=start_time,
                    finished_at=finished_time,
                    percent_done=100.0,
                ),
            )
        except Exception as e:
            error_msg = f"Error running {eval_type} evaluation: {e!s}"
            logger.error(error_msg)
            db_manager.update_evaluation(
                job["evaluation"].id,
                {
                    "error": error_msg,
                    "finished_at": datetime.utcnow(),
                    "progress": 0.0,
                },
            )
            # no need to raise error here, the error is captured, let the task continue to spin down the deployment
            previous_result.error = error_msg

    return previous_result


@celery_app.task(name="tasks.start_customization", pydantic=True)
def start_customization(previous_result: TaskResult) -> TaskResult:
    """
    Start customization process for the NIM.
    Takes the previous evaluation results.

    Args:
        previous_result: Result from the previous task containing workload_id and target_llm_model
    """
    if _should_skip_stage(previous_result, "start_customization"):
        return previous_result

    if not previous_result.nim.customization_enabled:
        logger.info(
            f"Customization skipped for {previous_result.nim.model_name} because it is using an external NIM"
        )
        return previous_result

    workload_id = previous_result.workload_id
    target_llm_model = previous_result.nim.model_name
    logger.info(
        f"Starting NIM customization for workload {workload_id} on model {target_llm_model}"
    )

    # Find the NIM run
    nim_run = db_manager.find_nim_run(
        previous_result.flywheel_run_id,
        previous_result.nim.model_name,
    )
    if not nim_run:
        msg = f"No NIM run found for model {target_llm_model}"
        logger.error(msg)
        raise ValueError(msg)

    start_time = datetime.utcnow()
    customizer = Customizer()

    # Create customization document with training tracking fields
    customization = NIMCustomization(
        nim_id=nim_run["_id"],
        workload_id=workload_id,
        base_model=target_llm_model,
        customized_model=None,  # Will be set when job starts
        started_at=start_time,
        progress=0.0,
        epochs_completed=0,
        steps_completed=0,
    )

    # Add customization to database
    db_manager.insert_customization(customization)

    def progress_callback(update_data):
        """Update customization document with progress"""
        db_manager.update_customization(customization.id, update_data)

    output_model_name = f"customized-{target_llm_model}".replace("/", "-")

    try:
        # Start customization job
        customization_job_id, customized_model = customizer.start_training_job(
            namespace=settings.nmp_config.nmp_namespace,
            name=f"customization-{workload_id}-{target_llm_model}",
            base_model=previous_result.nim.model_name,
            output_model_name=output_model_name,
            dataset_name=previous_result.datasets[DatasetType.TRAIN],
            training_config=settings.training_config,
        )
        logger.info(f"Customization job id: {customization_job_id}")

        # update uri in customization
        customization.nmp_uri = customizer.get_job_uri(customization_job_id)
        db_manager.update_customization(customization.id, {"nmp_uri": customization.nmp_uri})

        # Update customization with model name
        progress_callback({"customized_model": customized_model})

        # Wait for completion with progress updates
        customizer.wait_for_customization(customization_job_id, progress_callback=progress_callback)

        customizer.wait_for_model_sync(customized_model)

        # Update completion status
        finished_time = datetime.utcnow()
        final_update = {
            "finished_at": finished_time,
            "runtime_seconds": (finished_time - start_time).total_seconds(),
            "progress": 100.0,
        }
        progress_callback(final_update)

        # Final TaskResult update
        previous_result.update_customization(
            job_id=customization_job_id,
            model_name=customized_model,
            started_at=start_time,
            finished_at=finished_time,
            percent_done=100.0,
        )

    except Exception as e:
        error_msg = f"Error starting customization: {e!s}"
        logger.error(error_msg)
        db_manager.update_customization(
            customization.id,
            {
                "error": error_msg,
                "finished_at": datetime.utcnow(),
                "progress": 0.0,
            },
        )
        previous_result.error = error_msg
    return previous_result


@celery_app.task(name="tasks.run_customization_eval", pydantic=True)
def run_customization_eval(previous_result: TaskResult) -> TaskResult:
    """Run evaluation on the customized model."""
    if _should_skip_stage(previous_result, "run_customization_eval"):
        return previous_result

    try:
        if not previous_result.nim.customization_enabled:
            logger.info(f"Customization disabled for {previous_result.nim.model_name}")
            return previous_result

        if not previous_result.customization or not previous_result.customization.model_name:
            msg = "No customized model available for evaluation"
            logger.error(msg)
            raise ValueError(msg)

        customization_model = previous_result.customization.model_name
        workload_id = previous_result.workload_id

        # Find the customization document
        customization_doc = db_manager.find_customization(
            workload_id,
            customization_model,
        )
        if not customization_doc:
            msg = f"No customization found for model {customization_model}"
            logger.error(msg)
            raise ValueError(msg)

        print(
            f"Running evaluation on customized model {customization_model} for workload {workload_id}"
        )

        next_result = run_generic_eval(previous_result, EvalType.CUSTOMIZED, DatasetType.BASE)

        return next_result
    except Exception as e:
        error_msg = f"Error running customization evaluation: {e!s}"
        logger.error(error_msg)
        previous_result.error = error_msg
        return previous_result


@celery_app.task(name="tasks.shutdown_deployment", pydantic=True)
def shutdown_deployment(previous_results: list[TaskResult] | TaskResult) -> TaskResult:
    """Shutdown the NIM deployment.

    Args:
        previous_results: Either a single ``TaskResult`` or a list of them produced by a ``group``.
    """

    previous_result: TaskResult | None = None
    try:
        previous_result = _extract_previous_result(
            previous_results,
            validator=lambda r: getattr(r, "nim", None) is not None,
            error_msg="No valid TaskResult with NIM config found in results",
        )
        if (
            previous_result.llm_judge_config
            and previous_result.llm_judge_config.model_name == previous_result.nim.model_name
        ):
            logger.info(
                f"Skip shutting down NIM {previous_result.nim.model_name} as it is the same as the LLM Judge"
            )
            return previous_result

        dms_client = DMSClient(nmp_config=settings.nmp_config, nim=previous_result.nim)
        dms_client.shutdown_deployment()

        # Mark the NIM run as completed now that the deployment is shut down
        try:
            nim_run_doc = db_manager.find_nim_run(
                previous_result.flywheel_run_id,
                previous_result.nim.model_name,
            )
            if nim_run_doc:
                finished_time = datetime.utcnow()
                started_at = nim_run_doc.get("started_at")
                runtime_seconds: float = 0.0
                if started_at:
                    runtime_seconds = (finished_time - started_at).total_seconds()

                db_manager.mark_nim_completed(nim_run_doc["_id"], finished_time, runtime_seconds)
        except Exception as update_err:
            logger.error("Failed to update NIM run status to COMPLETED: %s", update_err)

        return previous_result
    except Exception as e:
        error_msg = f"Error shutting down NIM deployment: {e!s}"
        logger.error(error_msg)

        # ``previous_result`` may not be available if extraction failed.
        previous_result = locals().get("previous_result")  # type: ignore[arg-type]
        if not previous_result:
            return previous_result  # type: ignore[return-value]

        nim_run_doc = db_manager.find_nim_run(
            previous_result.flywheel_run_id,
            previous_result.nim.model_name,
        )
        if not nim_run_doc:
            logger.error(
                f"Could not find NIM run for flywheel_run_id: {previous_result.flywheel_run_id}"
            )
            return previous_result

        # Update nim document with error
        db_manager.mark_nim_error(nim_run_doc["_id"], error_msg)
        previous_result.error = error_msg
    return previous_result


@celery_app.task(name="tasks.finalize_flywheel_run", pydantic=True)
def finalize_flywheel_run(previous_results: list[TaskResult] | TaskResult) -> TaskResult:
    """Finalize the Flywheel run by setting its ``finished_at`` timestamp.

    Args:
        previous_results: Either a single ``TaskResult`` or a list returned by a ``group``.
    """

    previous_result: TaskResult | None = None
    try:
        previous_result = _extract_previous_result(
            previous_results,
            validator=lambda r: bool(r.flywheel_run_id),
            error_msg="Could not determine flywheel_run_id when finalizing Flywheel run",
        )
        # sleeping for 2 minutes to allow the deployment to be deleted
        time.sleep(120)
        # Delegate to the shared DB helper so we keep raw MongoDB access
        # centralised in a single place.
        db_manager.mark_flywheel_run_completed(previous_result.flywheel_run_id, datetime.utcnow())

        logger.info(
            "Flywheel run %s marked as finished at %s",
            previous_result.flywheel_run_id,
            datetime.utcnow(),
        )
        return previous_result
    except Exception as e:
        error_msg = f"Error finalizing Flywheel run: {e!s}"
        logger.error(error_msg)
        if previous_result:
            previous_result.error = error_msg
            return previous_result
        # sleeping for 2 minutes to allow the deployment to be deleted
        time.sleep(120)
        # If we cannot obtain previous_result, construct a minimal one
        return TaskResult(error=error_msg)


@celery_app.task(name="tasks.run_nim_workflow_dag", pydantic=True, queue="parent_queue")
def run_nim_workflow_dag(workload_id: str, flywheel_run_id: str, client_id: str) -> dict:
    """
    Execute the NIM workflow as a DAG where:
    - Data upload must complete first
    - Then NIMs can be spun up in parallel
    - Each NIM runs its evaluations in parallel
    - Finally, NIMs are shut down
    """

    logger.info(f"Running NIM workflow DAG for workload {workload_id} with flywheel_run_id {flywheel_run_id} and client_id {client_id}")
    # Create a group of chains for each NIM
    nim_chains = []
    for nim in settings.nims:
        assert isinstance(nim, NIMConfig)
        # For each NIM, create a chain: spin_up_nim -> parallel_evals
        nim_chain = chain(
            spin_up_nim.s(nim_config=nim.model_dump()),  # Convert NIMConfig to dict
            group(
                run_base_eval.s(),
                run_icl_eval.s(),
                chain(
                    start_customization.s(),
                    run_customization_eval.s(),
                ),
            ),
            shutdown_deployment.s(),
        )
        nim_chains.append(nim_chain)

    # Create the complete workflow
    workflow = chain(
        create_datasets.s(
            workload_id=workload_id, flywheel_run_id=flywheel_run_id, client_id=client_id
        ),
        wait_for_llm_as_judge.s(),  ## spin up llm-judge
        chain(*nim_chains),
        finalize_flywheel_run.s(),
    )

    # Submit the workflow to Celery and block until completion.
    # The application is not currently aware of how
    # many GPUs are available to it, so it serializes all calls
    # to `run_nim_workflow_dag` to prevent spinning up NIMs from taking
    # up all the GPUs and not leaving any available for customizations.
    # The following call to `get` will block. Since this task is running
    # on the ``parent_queue`` it will be serialized with all other tasks
    # on that queue. All other tasks run on the default celery queue,
    # which has a concurrency limit of 50.
    async_result = workflow.apply_async()
    return async_result.get(disable_sync_subtasks=False)


# -------------------------------------------------------------
# Helper utilities
# -------------------------------------------------------------


def _should_skip_stage(previous_result: TaskResult | None, stage_name: str) -> bool:
    """Return True if the previous_result already carries an error.

    When a stage fails we record the error message on the TaskResult instance.
    Any subsequent stage that receives a TaskResult with ``error`` set will
    short-circuit by returning immediately so the overall DAG keeps running
    serially without raising.
    """

    if isinstance(previous_result, TaskResult) and previous_result.error:
        logger.warning(
            "Skipping %s because a previous stage failed: %s",
            stage_name,
            previous_result.error,
        )
        return True
    return False


# -------------------------------------------------------------
# Shared helpers
# -------------------------------------------------------------


def _extract_previous_result(
    previous_results: list[TaskResult] | TaskResult | dict,
    *,
    validator: Callable[[TaskResult], bool] | None = None,
    error_msg: str = "No valid TaskResult found",
) -> TaskResult:
    """Return a single ``TaskResult`` from *previous_results*.

    Celery tasks can receive either a single ``TaskResult`` instance, a raw
    ``dict`` serialized version of it, or a list containing any mix of those.
    This helper normalises that input so downstream code can safely assume it
    has a *TaskResult* instance to work with.

    If *previous_results* is a list the items are inspected **in reverse
    order** (i.e. most-recent first) until one satisfies the *validator*
    callable.  When *validator* is *None* the first item is returned.

    Args:
        previous_results: The value passed by the upstream Celery task.
        validator: Optional callable that returns *True* for a result that
            should be selected.
        error_msg: Message to include in the raised ``ValueError`` when no
            suitable result can be found.

    Returns:
        TaskResult: The selected result.

    Raises:
        ValueError: If *previous_results* does not contain a suitable
            ``TaskResult``.
    """

    # Fast-path: a single object (TaskResult or dict)
    if not isinstance(previous_results, list):
        if isinstance(previous_results, dict):
            return TaskResult(**previous_results)
        assert isinstance(previous_results, TaskResult)
        return previous_results

    # It is a list - iterate from the end (latest first)
    for result in reversed(previous_results):
        if isinstance(result, dict):
            result = TaskResult(**result)
        if not isinstance(result, TaskResult):
            continue
        if validator is None or validator(result):
            return result

    # Nothing matched - raise so caller can handle
    logger.error(error_msg)
    raise ValueError(error_msg)
