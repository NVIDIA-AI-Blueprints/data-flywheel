import json

from elasticsearch import Elasticsearch

from src.config import settings
from src.lib.flywheel.util import (
    validate_records,
)
from src.lib.integration.es_client import ES_COLLECTION_NAME, get_es_client
from src.log_utils import setup_logging

logger = setup_logging("data_flywheel.record_exporter")


class RecordExporter:
    es_client: Elasticsearch

    def __init__(self):
        self.es_client = get_es_client()

    def get_records(self, client_id: str, workload_id: str) -> list[dict]:
        logger.info(f"Pulling data from Elasticsearch for workload {workload_id}")
        # Define the search query
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
        response = self.es_client.search(index=ES_COLLECTION_NAME, body=search_query)

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
