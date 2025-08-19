from elasticsearch import Elasticsearch

from src.config import DataSplitConfig
from src.lib.integration.es_client import ES_COLLECTION_NAME, get_es_client
from src.log_utils import setup_logging

logger = setup_logging("data_flywheel.record_exporter")


class RecordExporter:
    es_client: Elasticsearch

    def __init__(self):
        self.es_client = get_es_client()

    def get_records(
        self, client_id: str, workload_id: str, split_config: DataSplitConfig
    ) -> list[dict]:
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
        }

        # Use scroll API for large datasets
        max_records = split_config.limit * 2
        scroll_size = min(1000, max_records)  # Process in chunks of 1000

        # Initial search with scroll
        response = self.es_client.search(
            index=ES_COLLECTION_NAME,
            body=search_query,
            scroll="2m",  # Keep scroll context for 2 minutes
            size=scroll_size,
        )

        records = []

        try:
            # Extract records from initial response first
            hits = response["hits"]["hits"]
            records.extend([hit["_source"] for hit in hits])

            # Check scroll_id after processing initial hits
            scroll_id = response.get("_scroll_id")
            if scroll_id is None:
                logger.warning(
                    "No scroll_id in initial response, cannot continue scrolling but collected initial batch"
                )
                return records  # Return what we have and exit early

            # Continue scrolling until we have enough records or no more data
            while len(hits) > 0 and len(records) < max_records:
                response = self.es_client.scroll(scroll_id=scroll_id, scroll="2m")
                hits = response["hits"]["hits"]

                # Process hits first, even if scroll context might be lost
                remaining_needed = max_records - len(records)
                new_records = [hit["_source"] for hit in hits[:remaining_needed]]
                records.extend(new_records)

                # Check scroll_id after processing hits
                scroll_id = response.get("_scroll_id")
                if scroll_id is None:
                    logger.warning(
                        "Scroll context lost after processing current batch, exiting scroll loop"
                    )
                    break
        except Exception as e:
            logger.error(f"Error pulling data from Elasticsearch: {e}")
            raise e
        finally:
            # Clear the scroll context - always executed even on failure
            if scroll_id:
                try:
                    self.es_client.clear_scroll(scroll_id=scroll_id)
                except Exception as e:
                    logger.warning(f"Failed to clear scroll context: {e}")

        logger.info(
            f"Found {len(records)} records for client_id {client_id} and workload_id {workload_id}"
        )
        return records
