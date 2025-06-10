import json
from datetime import datetime
from typing import Any

from bson import ObjectId

from src.api.db import get_db
from src.api.models import DatasetType, WorkloadClassification
from src.config import DataSplitConfig, settings
from src.lib.flywheel.util import (
    format_evaluator,
    format_training_data,
    generate_icl_records,
    select_icl_examples,
    split_records,
)
from src.lib.integration.data_validator import DataValidator
from src.lib.nemo.data_uploader import DataUploader
from src.log_utils import setup_logging

logger = setup_logging("data_flywheel.dataset_creator")


class DatasetCreator:
    records: list[dict[str, Any]]
    flywheel_run_id: str
    output_dataset_prefix: str
    workload_id: str
    ts: int
    split_config: DataSplitConfig

    def __init__(
        self,
        records: list[dict[str, Any]],
        flywheel_run_id: str,
        output_dataset_prefix: str,
        workload_id: str,
        split_config: DataSplitConfig | None = None,
    ):
        self.records = records
        self.flywheel_run_id = flywheel_run_id
        self.output_dataset_prefix = output_dataset_prefix
        self.workload_id = workload_id
        self.ts = int(datetime.utcnow().timestamp())
        self.split_config = split_config or settings.data_split_config

    def create_datasets(self, workload_type: WorkloadClassification) -> dict[str, str]:
        # Validate and clean records
        validator = DataValidator()
        validated_records = validator.validate_records(
            self.records,
            workload_type,
            split_config=self.split_config,
        )

        logger.info(
            f"Validation completed: {len(validated_records)} valid records from {len(self.records)} original records"
        )

        self.records = validated_records

        # Update FlywheelRun document with number of records
        db = get_db()
        db.flywheel_runs.update_one(
            {"_id": ObjectId(self.flywheel_run_id)}, {"$set": {"num_records": len(self.records)}}
        )

        # split the jsonl data into train and val
        eval_records, train_records, val_records = split_records(self.records, self.split_config)
        logger.info(
            f"Split {len(self.records)} records into {len(eval_records)} eval, {len(train_records)} train, {len(val_records)} val"
        )

        # Select ICL examples from training records
        icl_examples = select_icl_examples(train_records, settings.icl_config, workload_type)
        msg = f"ICL Examples:\n Workload Type: {workload_type}\n"
        for tool_name, examples in icl_examples.items():
            msg += f"Selected {len(examples)} examples for tool {tool_name}\n"
        logger.info(msg)
        logger.info("\n\n")

        ## format the training data
        train_records = format_training_data(train_records, workload_type)
        val_records = format_training_data(val_records, workload_type)

        # Format evaluation data for OpenAI API compatibility (convert tool call args to strings)
        eval_records = format_evaluator(eval_records)

        # Convert all record sets to JSONL format
        eval_jsonl_data, train_jsonl_data, val_jsonl_data = (
            "\n".join(json.dumps(record) for record in records)
            for records in [eval_records, train_records, val_records]
        )

        eval_dataset_name = f"flywheel-eval-{self.output_dataset_prefix + '-' if self.output_dataset_prefix else ''}{self.workload_id}-{self.ts}"
        eval_uploader = DataUploader(dataset_name=eval_dataset_name)
        eval_uploader.upload_data(eval_jsonl_data, "eval_data.jsonl")

        icl_records = generate_icl_records(eval_records, selected_examples=icl_examples)
        # Format ICL records for evaluation as well
        icl_records = format_evaluator(icl_records)
        icl_jsonl_data = "\n".join(json.dumps(record) for record in icl_records)
        icl_dataset_name = f"flywheel-icl-{self.output_dataset_prefix + '-' if self.output_dataset_prefix else ''}{self.workload_id}-{self.ts}"
        icl_uploader = DataUploader(dataset_name=icl_dataset_name)
        icl_uploader.upload_data(icl_jsonl_data, "eval_data.jsonl")

        train_dataset_name = f"flywheel-train-{self.output_dataset_prefix + '-' if self.output_dataset_prefix else ''}{self.workload_id}-{self.ts}"
        train_uploader = DataUploader(dataset_name=train_dataset_name)
        train_uploader.upload_data(train_jsonl_data, "training/train_data.jsonl")
        train_uploader.upload_data(val_jsonl_data, "validation/val_data.jsonl")

        # update the flywheel run with the dataset names
        db.flywheel_runs.update_one(
            {"_id": ObjectId(self.flywheel_run_id)},
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

        return {
            DatasetType.BASE: eval_dataset_name,
            DatasetType.ICL: icl_dataset_name,  # as testing record are converted to icl records and uploaded
            DatasetType.TRAIN: train_dataset_name,
        }
