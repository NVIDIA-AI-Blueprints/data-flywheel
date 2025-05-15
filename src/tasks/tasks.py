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
import json
import os
from datetime import datetime
from typing import Any

from bson import ObjectId
from celery import Celery, chain, group, signals

from src.api.db import get_db, init_db
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
    format_training_data,
    generate_icl_records,
    identify_workload_type,
    split_records,
    validate_records,
)
from src.lib.integration.es_client import ES_COLLECTION_NAME, get_es_client
from src.lib.nemo.customizer import Customizer
from src.lib.nemo.data_uploader import DataUploader
from src.lib.nemo.dms_client import DMSClient
from src.lib.nemo.evaluator import Evaluator
from src.log_utils import setup_logging

logger = setup_logging("data_flywheel.tasks")

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
    """
    try:
        logger.info(f"Pulling data from Elasticsearch for workload {workload_id}")
        # Define the search query
        es_client = get_es_client()
        search_query = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"client_id": client_id}},
                        {"match": {"workload_id": workload_id}},
                    ]
                }
            },
            "sort": [{"timestamp": {"order": "desc"}}],
            "size": settings.data_split_config.limit,
        }

        # Execute the search query
        response = es_client.search(index=ES_COLLECTION_NAME, body=search_query)

        # Check if any records were found
        if not response["hits"]["hits"]:
            msg = f"No records found for the given client_id {client_id} and workload_id {workload_id}"
            logger.error(msg)
            raise ValueError(msg)

        # Extract the records
        records = [hit["_source"] for hit in response["hits"]["hits"]]
        logger.info(
            f"Found {len(records)} records for client_id {client_id} and workload_id {workload_id}"
        )

        workload_type = identify_workload_type(records)
        logger.info(f"Workload type: {workload_type}")

        # Deduplicate records based on request.messages and response.choices
        unique_records = {}
        for record in records:
            # Convert dictionaries to JSON strings for hashing
            messages_str = json.dumps(record.get("request", {}).get("messages", []), sort_keys=True)
            choices_str = json.dumps(record.get("response", {}).get("choices", []), sort_keys=True)
            key = (messages_str, choices_str)
            if key not in unique_records:
                unique_records[key] = record

        # Update records with deduplicated records
        records = list(unique_records.values())

        logger.info(f"Deduplicated down to {len(records)} records for workload {workload_id}")

        validate_records(records, workload_id, settings.data_split_config)

        # Update FlywheelRun document with number of records
        db = get_db()
        db.flywheel_runs.update_one(
            {"_id": ObjectId(flywheel_run_id)}, {"$set": {"num_records": len(records)}}
        )

        # split the jsonl data into train and val
        eval_records, train_records, val_records = split_records(
            records, settings.data_split_config
        )
        logger.info(
            f"Split {len(records)} records into {len(eval_records)} eval, {len(train_records)} train, {len(val_records)} val"
        )
        ## format the training data
        train_records = format_training_data(train_records)
        val_records = format_training_data(val_records)

        # Convert all record sets to JSONL format
        eval_jsonl_data, train_jsonl_data, val_jsonl_data = (
            "\n".join(json.dumps(record) for record in records)
            for records in [eval_records, train_records, val_records]
        )

        records = {}
        ts = int(datetime.utcnow().timestamp())

        eval_dataset_name = f"flywheel-eval-{output_dataset_prefix + '-' if output_dataset_prefix else ''}{workload_id}-{ts}"
        eval_uploader = DataUploader(
            namespace=settings.nmp_config.nmp_namespace, dataset_name=eval_dataset_name
        )
        eval_uploader.upload_data(eval_jsonl_data, "eval_data.jsonl")

        icl_records = generate_icl_records(eval_records)
        icl_jsonl_data = "\n".join(json.dumps(record) for record in icl_records)
        icl_dataset_name = f"flywheel-icl-{output_dataset_prefix + '-' if output_dataset_prefix else ''}{workload_id}-{ts}"
        icl_uploader = DataUploader(
            namespace=settings.nmp_config.nmp_namespace, dataset_name=icl_dataset_name
        )
        icl_uploader.upload_data(icl_jsonl_data, "eval_data.jsonl")

        train_dataset_name = f"flywheel-train-{output_dataset_prefix + '-' if output_dataset_prefix else ''}{workload_id}-{ts}"
        train_uploader = DataUploader(
            namespace=settings.nmp_config.nmp_namespace, dataset_name=train_dataset_name
        )
        train_uploader.upload_data(train_jsonl_data, "training/train_data.jsonl")
        train_uploader.upload_data(val_jsonl_data, "validation/val_data.jsonl")

        # update the flywheel run with the dataset names
        db.flywheel_runs.update_one(
            {"_id": ObjectId(flywheel_run_id)},
            {
                "$set": {
                    "datasets": [
                        {
                            "name": eval_dataset_name,
                            "num_records": len(eval_records),
                            "nmp_uri": eval_uploader.get_file_uri(),
                        },
                        {
                            "name": icl_dataset_name,
                            "num_records": len(icl_records),
                            "nmp_uri": icl_uploader.get_file_uri(),
                        },
                        {
                            "name": train_dataset_name,
                            "num_records": len(train_records),
                            "nmp_uri": train_uploader.get_file_uri(),
                        },
                    ],
                }
            },
        )

        return TaskResult(
            workload_id=workload_id,
            flywheel_run_id=flywheel_run_id,
            client_id=client_id,
            workload_type=workload_type,
            datasets={
                DatasetType.BASE: eval_dataset_name,
                DatasetType.ICL: icl_dataset_name,  # as testing record are converted to icl records and uploaded
                DatasetType.TRAIN: train_dataset_name,
            },
        )
    except Exception as e:
        error_msg = f"Error creating datasets: {e!s}"
        logger.error(error_msg)
        # Update flywheel run with error
        db = get_db()
        db.flywheel_runs.update_one(
            {"_id": ObjectId(flywheel_run_id)},
            {"$set": {"error": error_msg, "status": "error"}},
        )
        raise e


@celery_app.task(name="tasks.spin_up_llm_judge", pydantic=True)
def spin_up_llm_judge(previous_result: TaskResult) -> TaskResult:
    """
    Spin up a LLM Judge instance.
    Takes the result from the previous task as input.
    """
    try:
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

        # Create NIM run in the nims collection
        db = get_db()
        llm_judge_run = LLMJudgeRun(
            flywheel_run_id=ObjectId(previous_result.flywheel_run_id),
            model_name=llm_judge_config.model_name,
        )

        # Insert into llm_judge_runs collection
        result = db.llm_judge_runs.insert_one(llm_judge_run.to_mongo())
        llm_judge_run.id = result.inserted_id

        dms_client = DMSClient(nmp_config=settings.nmp_config, nim=llm_judge_config)

        if not dms_client.is_deployed():
            logger.info(f"Deploying LLM Judge {llm_judge_config.model_name}")

            try:
                dms_client.deploy_model()
            except Exception as e:
                logger.error(f"Error deploying LLM Judge {llm_judge_config.model_name}: {e}")
                raise e
        else:
            logger.info(f"LLM Judge {llm_judge_config.model_name} is already deployed")

        def progress_callback(status: dict):
            db.llm_judge_runs.update_one(
                {"_id": llm_judge_run.id},
                {"$set": {"deployment_status": DeploymentStatus(status.get("status", "unknown"))}},
            )

        dms_client.wait_for_deployment(progress_callback=progress_callback)

        dms_client.wait_for_model_sync(llm_judge_config.target_model_for_evaluation())

        return previous_result
    except Exception as e:
        error_msg = f"Error spinning up LLM Judge: {e!s}"
        logger.error(error_msg)
        db = get_db()
        llm_judge_run = LLMJudgeRun(
            flywheel_run_id=ObjectId(previous_result.flywheel_run_id),
            model_name=llm_judge_config.model_name,
        )
        # Update llm_judge_runs with error
        db.llm_judge_runs.update_one(
            {"_id": llm_judge_run.id},
            {"$set": {"error": error_msg, "deployment_status": DeploymentStatus.FAILED}},
        )
        dms_client.shutdown_deployment()
        raise e


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

    nim_config = NIMConfig(**nim_config)
    previous_result.nim = nim_config

    # Create NIM run in the nims collection
    db = get_db()
    start_time = datetime.utcnow()
    nim_run = NIMRun(
        flywheel_run_id=ObjectId(previous_result.flywheel_run_id),
        model_name=nim_config.model_name,
        evaluations=[],
        started_at=start_time,
        finished_at=start_time,  # Will be updated when evaluations complete
        runtime_seconds=0,  # Will be updated when evaluations complete
    )

    # Insert into nims collection
    result = db.nims.insert_one(nim_run.to_mongo())
    nim_run.id = result.inserted_id

    db.nims.update_one({"_id": nim_run.id}, {"$set": {"status": NIMRunStatus.DEPLOYING}})

    try:
        dms_client = DMSClient(nmp_config=settings.nmp_config, nim=nim_config)

        if not dms_client.is_deployed():
            logger.info(f"Deploying NIM {nim_config.model_name}")

            try:
                dms_client.deploy_model()
            except Exception as e:
                logger.error(f"Error deploying NIM {nim_config.model_name}: {e}")
                db.nims.update_one(
                    {"_id": nim_run.id}, {"$set": {"status": NIMRunStatus.ERROR, "error": str(e)}}
                )
                raise e
        else:
            logger.info(f"NIM {nim_config.model_name} is already deployed")

        def progress_callback(status: dict):
            db.nims.update_one(
                {"_id": nim_run.id},
                {"$set": {"deployment_status": DeploymentStatus(status.get("status", "unknown"))}},
            )

        dms_client.wait_for_deployment(progress_callback=progress_callback)

        dms_client.wait_for_model_sync(nim_config.target_model_for_evaluation())

        db.nims.update_one({"_id": nim_run.id}, {"$set": {"status": NIMRunStatus.RUNNING}})

        return previous_result
    except Exception as e:
        error_msg = f"Error spinning up NIM: {e!s}"
        logger.error(error_msg)
        db = get_db()
        # Update nims collection with error
        db.nims.update_one(
            {"_id": nim_run.id},
            {
                "$set": {
                    "error": error_msg,
                    "status": NIMRunStatus.ERROR,
                    "deployment_status": DeploymentStatus.FAILED,
                }
            },
        )
        dms_client.shutdown_deployment()
        raise e


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
    logger.info(f"Running {eval_type} evaluation")
    evaluator = Evaluator(llm_judge_config=previous_result.llm_judge_config)
    start_time = datetime.utcnow()

    tool_eval_types = [None]
    if previous_result.workload_type == WorkloadClassification.TOOL_CALLING:
        tool_eval_types = [ToolEvalType.TOOL_CALLING_METRIC]  # , ToolEvalType.TOOL_CALLING_JUDGE]

    jobs: list[dict[str, Any]] = []

    for tool_eval_type in tool_eval_types:
        # Find the NIM run for this model
        db = get_db()
        nim_run = db.nims.find_one(
            {
                "flywheel_run_id": ObjectId(previous_result.flywheel_run_id),
                "model_name": previous_result.nim.model_name,
            }
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
        db.evaluations.insert_one(evaluation.to_mongo())

        # Fix: Create closure with bound variables
        def make_progress_callback(db_instance, eval_instance):
            def callback(update_data):
                """Update evaluation document with progress"""
                db_instance.evaluations.update_one({"_id": eval_instance.id}, {"$set": update_data})

            return callback

        # Create callback with properly bound variables
        progress_callback = make_progress_callback(db, evaluation)

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
            db.evaluations.update_one(
                {"_id": evaluation.id},
                {
                    "$set": {
                        "error": error_msg,
                        "finished_at": datetime.utcnow(),
                        "progress": 0.0,
                    }
                },
            )

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
            db = get_db()
            # Update evaluation document with error
            db.evaluations.update_one(
                {"_id": job["evaluation"].id},
                {
                    "$set": {
                        "error": error_msg,
                        "finished_at": datetime.utcnow(),
                        "progress": 0.0,
                    }
                },
            )
            # no need to raise error here, the error is captured, let the task continue to spin down the deployment

    return previous_result


@celery_app.task(name="tasks.start_customization", pydantic=True)
def start_customization(previous_result: TaskResult) -> TaskResult:
    """
    Start customization process for the NIM.
    Takes the previous evaluation results.

    Args:
        previous_result: Result from the previous task containing workload_id and target_llm_model
    """
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
    db = get_db()
    nim_run = db.nims.find_one(
        {
            "flywheel_run_id": ObjectId(previous_result.flywheel_run_id),
            "model_name": previous_result.nim.model_name,
        }
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
    db.customizations.insert_one(customization.to_mongo())

    def progress_callback(update_data):
        """Update customization document with progress"""
        db.customizations.update_one({"_id": customization.id}, {"$set": update_data})

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
        db.customizations.update_one(
            {"_id": customization.id}, {"$set": {"nmp_uri": customization.nmp_uri}}
        )

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
        db = get_db()
        # Update customization document with error
        db.customizations.update_one(
            {"_id": customization.id},
            {
                "$set": {
                    "error": error_msg,
                    "finished_at": datetime.utcnow(),
                    "progress": 0.0,
                }
            },
        )
    return previous_result


@celery_app.task(name="tasks.run_customization_eval", pydantic=True)
def run_customization_eval(previous_result: TaskResult) -> TaskResult:
    """Run evaluation on the customized model."""
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
        db = get_db()
        customization_doc = db.customizations.find_one(
            {"workload_id": workload_id, "customized_model": customization_model}
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
        return previous_result


@celery_app.task(name="tasks.shutdown_deployment", pydantic=True)
def shutdown_deployment(previous_results: list[TaskResult] | TaskResult) -> TaskResult:
    """Shutdown the NIM deployment.

    Args:
        previous_results: Either a single TaskResult or a list of TaskResults from a group
    """

    # Handle both single TaskResult and list of results from group
    try:
        if isinstance(previous_results, list):
            # Take the last successful result that has nim config
            for result in reversed(previous_results):
                if isinstance(result, dict):
                    result = TaskResult(**result)
                if result and hasattr(result, "nim") and result.nim:
                    previous_result = result
                    break
            else:
                msg = "No valid TaskResult with NIM config found in results"
                logger.error(msg)
                raise ValueError(msg)
        else:
            previous_result = (
                previous_results
                if isinstance(previous_results, TaskResult)
                else TaskResult(**previous_results)
            )
        dms_client = DMSClient(nmp_config=settings.nmp_config, nim=previous_result.nim)
        dms_client.shutdown_deployment()

        return previous_result
    except Exception as e:
        error_msg = f"Error shutting down NIM deployment: {e!s}"
        logger.error(error_msg)
        db = get_db()
        nim_run = db.nims.find_one(
            {
                "flywheel_run_id": ObjectId(previous_result.flywheel_run_id),
                "model_name": previous_result.nim.model_name,
            }
        )
        if not nim_run:
            logger.error(
                f"Could not find NIM run for flywheel_run_id: {previous_result.flywheel_run_id}"
            )
            return previous_result

        nim_run = NIMRun.from_mongo(nim_run)
        # Update nim document with error
        db.nims.update_one(
            {"_id": nim_run.id},
            {
                "$set": {
                    "error": error_msg,
                    "status": NIMRunStatus.ERROR,
                    "deployment_status": DeploymentStatus.FAILED,
                }
            },
        )
    return previous_result


@celery_app.task(name="tasks.shutdown_llm_judge_deployment", pydantic=True)
def shutdown_llm_judge_deployment(previous_results: list[TaskResult] | TaskResult) -> TaskResult:
    """Shutdown the LLM Judge deployment.

    Args:
        previous_results: Either a single TaskResult or a list of TaskResults from a group
    """
    try:
        # Handle both single TaskResult and list of results from group
        if isinstance(previous_results, list):
            # Take the first result since they should all have the same llm_judge_config
            result = previous_results[0]
            previous_result = result if isinstance(result, TaskResult) else TaskResult(**result)
        else:
            previous_result = (
                previous_results
                if isinstance(previous_results, TaskResult)
                else TaskResult(**previous_results)
            )
        # Update the flywheel run to finished
        db = get_db()
        db.flywheel_runs.update_one(
            {"_id": ObjectId(previous_result.flywheel_run_id)},
            {"$set": {"finished_at": datetime.utcnow()}},
        )
        if not previous_result.llm_judge_config:
            logger.info("Remote LLM Judge is used, skipping shutdown")
            return previous_result

        dms_client = DMSClient(nmp_config=settings.nmp_config, nim=previous_result.llm_judge_config)
        dms_client.shutdown_deployment()
    except Exception as e:
        error_msg = f"Error shutting down LLM Judge deployment: {e!s}"
        logger.error(error_msg)
        db = get_db()
        # Update llm_judge_runs with error
        llm_judge_run = db.llm_judge_runs.find_one(
            {"flywheel_run_id": ObjectId(previous_result.flywheel_run_id)}
        )
        if not llm_judge_run:
            logger.error(
                f"Could not find LLM Judge run for flywheel_run_id: {previous_result.flywheel_run_id}"
            )
            return previous_result

        llm_judge_run = LLMJudgeRun.from_mongo(llm_judge_run)
        db.llm_judge_runs.update_one(
            {"_id": llm_judge_run.id},
            {
                "$set": {
                    "error": error_msg,
                    "deployment_status": DeploymentStatus.FAILED,
                }
            },
        )
    return previous_result


@celery_app.task(name="tasks.run_nim_workflow_dag", pydantic=True)
def run_nim_workflow_dag(workload_id: str, flywheel_run_id: str, client_id: str) -> dict:
    """
    Execute the NIM workflow as a DAG where:
    - Data upload must complete first
    - Then NIMs can be spun up in parallel
    - Each NIM runs its evaluations in parallel
    - Finally, NIMs are shut down
    """

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
        spin_up_llm_judge.s(),  ## spin up llm-judge
        group(*nim_chains, max_parallel=settings.nmp_config.nim_parallelism),
        shutdown_llm_judge_deployment.s(),
    )
    # Execute the DAG and return the AsyncResult
    return workflow.apply_async()
