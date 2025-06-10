#!/usr/bin/env python3
"""
Data Flywheel Blueprint Cleanup Script

This script cleans up all running resources by:
1. Finding all running flywheel runs from MongoDB
2. Finding all NIMs with RUNNING deployment status
3. Deleting all evaluation jobs associated with running customizations
4. Shutting down all NIMs
5. Shutting down LLM judge

The script should only be run when docker compose is down.
It temporarily starts the service to get information, then shuts it down.
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from bson import ObjectId
from pymongo import MongoClient

# Add the project root directory to the Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import basic modules that don't require database connection
from src.api.models import FlywheelRunStatus, NIMRunStatus  # noqa: E402
from src.api.schemas import DeploymentStatus  # noqa: E402
from src.log_utils import setup_logging  # noqa: E402

logger = setup_logging("cleanup_script")


class CleanupManager:
    """Manages the cleanup of all running resources."""

    def __init__(self):
        self.mongo_client = None
        self.db = None
        self.api_base_url = "http://localhost:8000"
        self.evaluator = None
        self.customizer = None
        self.cleanup_errors = []
        # Delay importing modules that require database connection
        self.settings = None
        self._dms_client_class = None
        self._evaluator_class = None
        self._customizer_class = None

    def _import_dependencies(self):
        """Import modules that require database connection."""
        if self.settings is None:
            from src.config import settings

            self.settings = settings

        if self._dms_client_class is None:
            from src.lib.nemo.dms_client import DMSClient

            self._dms_client_class = DMSClient

        if self._evaluator_class is None:
            from src.lib.nemo.evaluator import Evaluator

            self._evaluator_class = Evaluator

        if self._customizer_class is None:
            from src.lib.nemo.customizer import Customizer

            self._customizer_class = Customizer

    def check_docker_compose_status(self) -> bool:
        """Check if docker compose services are running."""
        try:
            result = subprocess.run(
                ["docker", "compose", "-f", "deploy/docker-compose.yaml", "ps", "--format", "json"],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=True,
            )

            if result.stdout.strip():
                # If there's output, services are running
                return True
            return False
        except subprocess.CalledProcessError:
            return False

    def start_service_temporarily(self):
        """Start MongoDB service temporarily to get information."""
        logger.info("Starting MongoDB service temporarily...")
        try:
            subprocess.run(
                ["docker", "compose", "-f", "deploy/docker-compose.yaml", "up", "-d", "mongodb"],
                cwd=project_root,
                check=True,
            )
            # Wait for MongoDB to be ready
            time.sleep(10)

            # Connect to MongoDB
            self.mongo_client = MongoClient("mongodb://localhost:27017")
            self.db = self.mongo_client["flywheel"]
            logger.info("Connected to MongoDB")

            # Now it's safe to import database-dependent modules
            self._import_dependencies()

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start MongoDB service: {e}")
            raise

    def stop_service(self):
        """Stop the temporarily started service."""
        logger.info("Stopping MongoDB service...")
        try:
            if self.mongo_client:
                self.mongo_client.close()

            subprocess.run(
                ["docker", "compose", "-f", "deploy/docker-compose.yaml", "down"],
                cwd=project_root,
                check=True,
            )
            logger.info("MongoDB service stopped")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to stop MongoDB service: {e}")

    def find_running_flywheel_runs(self) -> list[dict[str, Any]]:
        """Find all flywheel runs that are currently running."""
        logger.info("Finding running flywheel runs...")

        running_statuses = [FlywheelRunStatus.PENDING.value, FlywheelRunStatus.RUNNING.value]

        running_runs = list(self.db.flywheel_runs.find({"status": {"$in": running_statuses}}))

        logger.info(f"Found {len(running_runs)} running flywheel runs")
        return running_runs

    def find_running_nims(self, flywheel_run_id: ObjectId) -> list[dict[str, Any]]:
        """Find all NIMs with RUNNING or PENDING deployment status for a flywheel run."""
        running_nims = list(
            self.db.nims.find(
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

        logger.info(f"Found {len(running_nims)} running NIMs for flywheel run {flywheel_run_id}")
        return running_nims

    def find_customization_jobs(self, nim_id: ObjectId) -> list[dict[str, Any]]:
        """Find all customization jobs associated with a NIM."""
        customizations = list(self.db.customizations.find({"nim_id": nim_id}))

        logger.info(f"Found {len(customizations)} customizations for NIM {nim_id}")
        return customizations

    def find_evaluation_jobs(self, nim_id: ObjectId) -> list[dict[str, Any]]:
        """Find all evaluation jobs associated with a NIM."""
        evaluations = list(self.db.evaluations.find({"nim_id": nim_id}))

        logger.info(f"Found {len(evaluations)} evaluations for NIM {nim_id}")
        return evaluations

    def cancel_customization_jobs(self, customizations: list[dict[str, Any]]):
        """Cancel all running customization jobs."""
        if not customizations:
            return

        if not self.customizer:
            self.customizer = self._customizer_class()

        for customization in customizations:
            if "job_id" in customization:
                try:
                    self.customizer.cancel_job(customization["job_id"])
                    logger.info(f"Cancelled customization job {customization['job_id']}")
                except Exception as e:
                    error_msg = f"Failed to cancel customization job {customization['job_id']}: {e}"
                    logger.warning(error_msg)
                    self.cleanup_errors.append(error_msg)

    def shutdown_nim(self, nim: dict[str, Any]):
        """Shutdown a NIM deployment."""
        model_name = nim["model_name"]

        # Find the NIM config in settings
        nim_config = None
        for config in self.settings.nims:
            if config.model_name == model_name:
                nim_config = config
                break

        if not nim_config:
            logger.warning(f"NIM config for {model_name} not found in settings")
            return

        try:
            dms_client = self._dms_client_class(nmp_config=self.settings.nmp_config, nim=nim_config)
            dms_client.shutdown_deployment()
            logger.info(f"Shutdown NIM deployment for {model_name}")
        except Exception as e:
            error_msg = f"Failed to shutdown NIM {model_name}: {e}"
            logger.warning(error_msg)
            self.cleanup_errors.append(error_msg)

    def shutdown_llm_judge(self):
        """Shutdown LLM judge if it's running locally."""
        try:
            llm_judge_config = self.settings.llm_judge_config

            if llm_judge_config.is_remote:
                logger.info("LLM judge is remote, no shutdown needed")
                return

            dms_client = self._dms_client_class(
                nmp_config=self.settings.nmp_config, nim=llm_judge_config
            )
            dms_client.shutdown_deployment()
            logger.info(f"Shutdown LLM judge deployment for {llm_judge_config.model_name}")
        except Exception as e:
            error_msg = f"Failed to shutdown LLM judge: {e}"
            logger.warning(error_msg)
            self.cleanup_errors.append(error_msg)

    def mark_resources_as_cancelled(self, flywheel_run_id: ObjectId):
        """Mark all resources as cancelled in the database."""
        try:
            # Mark flywheel run as cancelled
            self.db.flywheel_runs.update_one(
                {"_id": flywheel_run_id},
                {
                    "$set": {
                        "status": FlywheelRunStatus.CANCELLED.value,
                        "finished_at": datetime.utcnow(),
                        "error": "Cancelled by cleanup script",
                    }
                },
            )

            # Mark all NIMs as cancelled
            self.db.nims.update_many(
                {"flywheel_run_id": flywheel_run_id},
                {
                    "$set": {
                        "status": NIMRunStatus.CANCELLED.value,
                        "deployment_status": DeploymentStatus.CANCELLED.value,
                        "finished_at": datetime.utcnow(),
                        "error": "Cancelled by cleanup script",
                    }
                },
            )

            # Mark LLM judge as cancelled
            self.db.llm_judge_runs.update_one(
                {"flywheel_run_id": flywheel_run_id},
                {
                    "$set": {
                        "deployment_status": DeploymentStatus.CANCELLED.value,
                        "error": "Cancelled by cleanup script",
                    }
                },
            )

            logger.info(f"Marked all resources as cancelled for flywheel run {flywheel_run_id}")

        except Exception as e:
            error_msg = f"Failed to mark resources as cancelled: {e}"
            logger.warning(error_msg)
            self.cleanup_errors.append(error_msg)

    def cleanup_flywheel_run(self, flywheel_run: dict[str, Any]):
        """Clean up all resources for a single flywheel run."""
        flywheel_run_id = flywheel_run["_id"]
        logger.info(f"Cleaning up flywheel run {flywheel_run_id}")

        # Find running NIMs
        running_nims = self.find_running_nims(flywheel_run_id)

        for nim in running_nims:
            nim_id = nim["_id"]
            logger.info(f"Processing NIM {nim['model_name']} (ID: {nim_id})")

            # Find and clean up customization jobs
            customizations = self.find_customization_jobs(nim_id)
            if customizations:
                self.cancel_customization_jobs(customizations)

            # TODO: Find and cancel evaluation jobs once available from NMP

            # Shutdown the NIM
            self.shutdown_nim(nim)

        # Mark all resources as cancelled in the database
        self.mark_resources_as_cancelled(flywheel_run_id)

    def run_cleanup(self):
        """Main cleanup procedure."""
        logger.info("Starting cleanup of running resources...")

        # Check if docker compose is already running
        if self.check_docker_compose_status():
            logger.error("Docker compose services are still running. Please stop them first with:")
            logger.error("cd deploy && docker compose down")
            sys.exit(1)

        try:
            # Start service temporarily to get information
            self.start_service_temporarily()

            # Find all running flywheel runs
            running_flywheel_runs = self.find_running_flywheel_runs()

            if not running_flywheel_runs:
                logger.info("No running flywheel runs found. Nothing to clean up.")
                return

            # Clean up each flywheel run
            for flywheel_run in running_flywheel_runs:
                try:
                    self.cleanup_flywheel_run(flywheel_run)
                except Exception as e:
                    error_msg = f"Failed to clean up flywheel run {flywheel_run['_id']}: {e}"
                    logger.error(error_msg)
                    self.cleanup_errors.append(error_msg)

            # Shutdown LLM judge
            self.shutdown_llm_judge()

            # Report results
            if self.cleanup_errors:
                logger.warning(f"Cleanup completed with {len(self.cleanup_errors)} errors:")
                for error in self.cleanup_errors:
                    logger.warning(f"  - {error}")
            else:
                logger.info("Cleanup completed successfully with no errors!")

        finally:
            # Always stop the service
            self.stop_service()


def main():
    """Main entry point."""
    # Change to the project root directory
    os.chdir(project_root)

    print("=" * 60)
    print("Data Flywheel Blueprint - Cleanup Running Resources")
    print("=" * 60)
    print()

    # Confirm action
    response = input("This will clean up all running resources. Continue? (y/N): ")
    if response.lower() != "y":
        print("Cleanup cancelled.")
        sys.exit(0)

    cleanup_manager = CleanupManager()

    try:
        cleanup_manager.run_cleanup()
        print("\nCleanup process completed!")

    except KeyboardInterrupt:
        print("\nCleanup interrupted by user")
        cleanup_manager.stop_service()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        cleanup_manager.stop_service()
        sys.exit(1)


if __name__ == "__main__":
    main()
