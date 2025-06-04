import json
from typing import Optional, List

import weave
from src.config import settings
from src.lib.flywheel.util import validate_records
from src.log_utils import setup_logging
from weave.trace.context.weave_client_context import (
     get_weave_client,
 )
from weave import Dataset

logger = setup_logging("data_flywheel.record_exporter_weave")


class WeaveRecordExporter:
    def __init__(
        self, 
        op_names: Optional[List[str]] = None
    ):
        """Initialize the Weave record exporter.
        
        Args:
            project_name: The name of the Weave project to export from
            op_names: List of operation names to filter on. If None, will fetch all operations.
                     Can include wildcards, e.g. ["*:generate", "call_chat:*"]
        """
        self.client = get_weave_client()
        if not self.client:
            self.client = weave.init(settings.wandb_config.project)
        self.op_names = op_names or ["*"]  # Default to all operations if none specified

    def get_records(self, client_id: str, workload_id: str) -> list[dict]:
        """Get records from Weave for a specific client and workload.
        
        Args:
            client_id: The client identifier
            workload_id: The workload identifier
            
        Returns:
            List of records matching the criteria
        """
        logger.info(f"Pulling data from Weave for workload {workload_id}")
        
        # Construct the filter and query
        filter_dict = {}
        if self.op_names != ["*"]:
            filter_dict["op_names"] = [
                f"weave:///{self.client.project_name}/op/{op_name}" 
                for op_name in self.op_names
            ]
        
        # Use $expr and $contains for client_id and workload_id matching
        query_dict = {
            "$expr": {
                "$and": [
                    {
                        "$contains": {
                            "input": {"$getField": "output.client_id"},
                            "substr": {"$literal": client_id}
                        }
                    },
                    {
                        "$contains": {
                            "input": {"$getField": "output.workload_id"},
                            "substr": {"$literal": workload_id}
                        }
                    }
                ]
            }
        }
        
        # Get calls from Weave matching our criteria
        calls = self.client.get_calls(
            filter=filter_dict,
            query=query_dict,
            sort_by=[{"field": "started_at", "direction": "desc"}],
            limit=settings.data_split_config.limit
        )

        # Check if any records were found
        if not calls:
            msg = f"No records found for the given client_id {client_id} and workload_id {workload_id}"
            logger.error(msg)
            raise ValueError(msg)

        # Extract the records from the calls
        records = []
        for call in calls:
            # Get the inner response structure
            inner_response = call.output.get("response", {})
            
            # Convert the call output to our record format with flattened response
            record = {
                "timestamp": call.started_at.timestamp(),
                "client_id": call.output.get("client_id"),
                "workload_id": call.output.get("workload_id"),
                "request": call.inputs,  # Contains model, messages, temperature, max_tokens
                "response": inner_response  # Use the inner response directly
            }
            records.append(record)

        logger.info(
            f"Found {len(records)} records for client_id {client_id} and workload_id {workload_id}"
        )

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

        return records
    
    def save_to_weave_dataset(self, records: list[dict], client_id: str, workload_id: str):
        """Save records to Weave evals.
        
        Args:
            records: List of records to save
            workload_id: The workload identifier
            run_id: The run identifier
        """
        logger.info(f"Saving {len(records)} records to Weave for client {client_id} and workload {workload_id}")
        
        # Create a dataset
        dataset = Dataset(
            name=f"flywheel-eval-{client_id}-{workload_id}",
            rows=records
        )

        # Publish the dataset
        weave.publish(dataset)

        return
