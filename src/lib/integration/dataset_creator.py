import json
from datetime import datetime
from typing import Any

from bson import ObjectId

from src.api.db import get_db, init_db
from src.api.models import DatasetType
from src.config import settings
from src.lib.flywheel.util import format_training_data, generate_icl_records, split_records
from src.lib.nemo.data_uploader import DataUploader
from src.log_utils import setup_logging

logger = setup_logging("data_flywheel.dataset_creator")


class DatasetCreator:
    records: list[dict[str, Any]]
    flywheel_run_id: str
    output_dataset_prefix: str
    workload_id: str
    ts: int

    def __init__(
        self,
        records: list[dict[str, Any]],
        flywheel_run_id: str,
        output_dataset_prefix: str,
        workload_id: str,
    ):
        self.records = records
        self.flywheel_run_id = flywheel_run_id
        self.output_dataset_prefix = output_dataset_prefix
        self.workload_id = workload_id
        self.ts = int(datetime.utcnow().timestamp())
        init_db()

    def create_datasets(self) -> dict[str, str]:
        # Update FlywheelRun document with number of records
        db = get_db()
        db.flywheel_runs.update_one(
            {"_id": ObjectId(self.flywheel_run_id)}, {"$set": {"num_records": len(self.records)}}
        )

        # split the jsonl data into train and val
        eval_records, train_records, val_records = split_records(
            self.records, settings.data_split_config
        )
        logger.info(
            f"Split {len(self.records)} records into {len(eval_records)} eval, {len(train_records)} train, {len(val_records)} val"
        )
        ## format the training data
        train_records = format_training_data(train_records)
        val_records = format_training_data(val_records)

        # Convert all record sets to JSONL format
        eval_jsonl_data, train_jsonl_data, val_jsonl_data = (
            "\n".join(json.dumps(record) for record in records)
            for records in [eval_records, train_records, val_records]
        )

        eval_dataset_name = f"flywheel-eval-{self.output_dataset_prefix + '-' if self.output_dataset_prefix else ''}{self.workload_id}-{self.ts}"
        eval_uploader = DataUploader(
            namespace=settings.nmp_config.nmp_namespace, dataset_name=eval_dataset_name
        )
        eval_uploader.upload_data(eval_jsonl_data, "eval_data.jsonl")

        icl_records = generate_icl_records(eval_records)
        icl_jsonl_data = "\n".join(json.dumps(record) for record in icl_records)
        icl_dataset_name = f"flywheel-icl-{self.output_dataset_prefix + '-' if self.output_dataset_prefix else ''}{self.workload_id}-{self.ts}"
        icl_uploader = DataUploader(
            namespace=settings.nmp_config.nmp_namespace, dataset_name=icl_dataset_name
        )
        icl_uploader.upload_data(icl_jsonl_data, "eval_data.jsonl")

        train_dataset_name = f"flywheel-train-{self.output_dataset_prefix + '-' if self.output_dataset_prefix else ''}{self.workload_id}-{self.ts}"
        train_uploader = DataUploader(
            namespace=settings.nmp_config.nmp_namespace, dataset_name=train_dataset_name
        )
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
