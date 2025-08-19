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
import shutil
import zipfile
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import requests

from src.api.models import WorkloadClassification
from src.config import MLflowConfig
from src.log_utils import setup_logging

logger = setup_logging("data_flywheel.mlflow")


class MLflowClient:
    """Client for MLflow integration with Data Flywheel Blueprint."""

    def __init__(self, mlflow_config: MLflowConfig):
        """Initialize the MLflow client.

        Args:
            mlflow_config: MLflow configuration
        """
        self.config = mlflow_config
        if self.config.enabled:
            mlflow.set_tracking_uri(self.config.tracking_uri)

    def download_and_process_eval(
        self, eval_id: str, nmp_eval_uri: str, save_dir: Path, model: str, eval_type: str
    ) -> Path | None:
        """
        Downloads evaluation results and extracts ZIP contents to save_dir.

        Args:
            eval_id: Evaluation ID
            nmp_eval_uri: NMP evaluation URI
            save_dir: Directory to save results
            model: Model name
            eval_type: Evaluation type

        Returns:
            Path to the processed results file or None if failed
        """
        try:
            # Create output directory
            save_dir.mkdir(parents=True, exist_ok=True)

            # Download results
            url = f"{nmp_eval_uri}/download-results"
            response = requests.get(url, headers={"accept": "application/json"}, timeout=30)

            if not response.ok:
                logger.error(f"Download failed [{response.status_code}]: {response.text}")
                return None

            # Save ZIP file
            zip_path = save_dir / f"result_{eval_id}.zip"
            zip_path.write_bytes(response.content)
            logger.info(f"Downloaded results to {zip_path}")

            # Extract ZIP contents
            with zipfile.ZipFile(zip_path) as zip_file:
                zip_file.extractall(save_dir)
                logger.info(f"Extracted {len(zip_file.namelist())} files")

            # Cleanup ZIP
            zip_path.unlink()

            # Find and rename results.json
            results_json = next(save_dir.glob("**/results.json"), None)
            if results_json:
                # Sanitize model name to avoid directory separators in filename
                safe_model_name = model.replace("/", "_").replace("\\", "_")
                new_name = save_dir / f"{safe_model_name}_{eval_type}.json"
                results_json.rename(new_name)
                logger.info(f"Renamed results to {new_name}")
                logger.info(f"Successfully processed {eval_id}")
                return new_name

            logger.error("Error: results.json not found in extracted files")
            logger.error(f"Failed to process {eval_id}")
            return None

        except Exception as e:
            logger.error(f"Error processing evaluation {eval_id}: {e!s}")
            return None

    def _load_results(self, file_path: Path) -> dict[str, Any]:
        """Load results from JSON file."""
        try:
            with open(file_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading results from {file_path}: {e!s}")
            raise

    def _extract_metrics(
        self, item: dict[str, Any], workload_type: WorkloadClassification
    ) -> dict[str, Any]:
        """Extract metrics from evaluation item."""
        try:
            metrics = item.get("metrics", {})
            sample = item.get("sample", {})
            response = sample.get("response", {})
            usage = response.get("usage", {})

            if workload_type == WorkloadClassification.TOOL_CALLING:
                # Tool calling metrics
                return {
                    "function_name_accuracy": metrics.get("tool-calling-accuracy", {})
                    .get("scores", {})
                    .get("function_name_accuracy", {})
                    .get("value"),
                    "function_name_and_args_accuracy": metrics.get("tool-calling-accuracy", {})
                    .get("scores", {})
                    .get("function_name_and_args_accuracy", {})
                    .get("value"),
                    "correctness_rating": metrics.get("correctness", {})
                    .get("scores", {})
                    .get("rating", {})
                    .get("value"),
                    "total_tokens": usage.get("total_tokens"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                }
            else:
                # Generic metrics (llm-as-judge)
                return {
                    "similarity": metrics.get("llm-judge", {})
                    .get("scores", {})
                    .get("similarity", {})
                    .get("value"),
                    "total_tokens": usage.get("total_tokens"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                }
        except Exception as e:
            logger.error(f"Error extracting metrics: {e!s}")
            return {}

    def _extract_metadata(self, item: dict[str, Any]) -> dict[str, Any]:
        """Extract metadata from evaluation item."""
        try:
            sample = item.get("sample", {})
            response = sample.get("response", {})
            return {
                "model": response.get("model"),
                "workload_id": item.get("item", {}).get("workload_id"),
                "client_id": item.get("item", {}).get("client_id"),
                "timestamp": item.get("item", {}).get("timestamp"),
            }
        except Exception as e:
            logger.error(f"Error extracting metadata: {e!s}")
            return {}

    def upload_result(
        self,
        results_path: Path,
        flywheel_run_id: str,
        model_name: str,
        eval_type: str,
        workload_type: WorkloadClassification = WorkloadClassification.GENERIC,
    ) -> str | None:
        """
        Upload evaluation results to MLflow.

        Args:
            results_path: Path to results file
            flywheel_run_id: Flywheel run ID
            model_name: Model name
            eval_type: Evaluation type

        Returns:
            MLflow run URL or None if failed
        """
        if not self.config.enabled:
            logger.debug("MLflow integration is disabled")
            return None

        try:
            # Parse experiment name and run name
            experiment_name = f"{self.config.experiment_name_prefix}-{flywheel_run_id}"
            run_name = f"{model_name.replace('/', '_')}_{eval_type}"

            # Get or create experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    experiment_name, artifact_location=self.config.artifact_location
                )
                logger.info(f"Created new experiment: {experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")

            # Load results
            results = self._load_results(results_path)

            # Group by model (if needed)
            # custom-tool-calling if TOOL_CALLING else llm-as-judge if generic
            model_evaluations = {}
            task_key = (
                "custom-tool-calling"
                if workload_type == WorkloadClassification.TOOL_CALLING
                else "llm-as-judge"
            )
            for item in results.get(task_key, []):
                metadata = self._extract_metadata(item)
                model_name_from_item = metadata.get("model", model_name)
                model_evaluations.setdefault(model_name_from_item, []).append(item)

            # Print summary of what will be uploaded
            logger.info("📊 Uploading evaluation results to MLflow:")
            logger.info(f"   📁 Experiment: {experiment_name}")
            logger.info(f"   📄 Results file: {results_path.name}")
            logger.info(f"   🤖 Models found: {len(model_evaluations)}")
            for model_name_in_item, evaluations in model_evaluations.items():
                logger.info(f"      - {model_name_in_item}: {len(evaluations)} evaluation samples")

            # Upload each model's results as a separate run (if multiple models)
            mlflow_run_url = None
            for model_name_in_item, evaluations in model_evaluations.items():
                this_run_name = (
                    f"{run_name}_{model_name_in_item.replace('/', '_')}"
                    if len(model_evaluations) > 1
                    else run_name
                )

                with mlflow.start_run(experiment_id=experiment_id, run_name=this_run_name):
                    mlflow.log_param("model", model_name_in_item)
                    mlflow.log_param("run_name", run_name)
                    mlflow.log_param("experiment_name", experiment_name)
                    mlflow.log_param("flywheel_run_id", flywheel_run_id)
                    mlflow.log_param("eval_type", eval_type)

                    metrics_data = []
                    for item in evaluations:
                        metrics = self._extract_metrics(item, workload_type)
                        metadata = self._extract_metadata(item)
                        metrics_data.append({"timestamp": metadata.get("timestamp"), **metrics})

                    metrics_df = pd.DataFrame(metrics_data).sort_values("timestamp")

                    # Log step-by-step metrics
                    for step, (_, row) in enumerate(metrics_df.iterrows()):
                        step_metrics = row.drop("timestamp").to_dict()
                        # Filter out None values
                        step_metrics = {k: v for k, v in step_metrics.items() if v is not None}
                        if step_metrics:
                            mlflow.log_metrics(step_metrics, step=step)

                    # Log summary statistics
                    summary_metrics = {}
                    if workload_type == WorkloadClassification.TOOL_CALLING:
                        metric_columns = [
                            "function_name_accuracy",
                            "function_name_and_args_accuracy",
                            "correctness_rating",
                            "total_tokens",
                            "prompt_tokens",
                            "completion_tokens",
                        ]
                    else:
                        metric_columns = [
                            "similarity",
                            "total_tokens",
                            "prompt_tokens",
                            "completion_tokens",
                        ]

                    for col in metric_columns:
                        if col in metrics_df.columns:
                            mean_val = metrics_df[col].mean()
                            if pd.notna(mean_val):
                                summary_metrics[f"mean_{col}"] = mean_val

                    summary_metrics["total_evaluations"] = len(evaluations)
                    mlflow.log_metrics(summary_metrics)

                    # Store the experiment URL for the first model (or only model)
                    if mlflow_run_url is None:
                        # Construct the experiment URL from tracking URI and experiment ID
                        tracking_uri = mlflow.get_tracking_uri()
                        if tracking_uri.endswith("/"):
                            tracking_uri = tracking_uri[:-1]
                        mlflow_run_url = f"{tracking_uri}/#/experiments/{experiment_id}"

                logger.info(f"✅ Uploaded run '{this_run_name}' to experiment '{experiment_name}'")

            return mlflow_run_url

        except Exception as e:
            logger.error(f"Error uploading results to MLflow: {e!s}")
            return None

    def upload_evaluation_results(
        self,
        job_id: str,
        nmp_eval_uri: str,
        flywheel_run_id: str,
        model_name: str,
        eval_type: str,
        workload_type: WorkloadClassification = WorkloadClassification.GENERIC,
    ) -> str | None:
        """
        Complete workflow to download and upload evaluation results to MLflow.

        Args:
            job_id: Job ID for the evaluation
            nmp_eval_uri: NMP evaluation URI
            flywheel_run_id: Flywheel run ID
            model_name: Model name
            eval_type: Evaluation type

        Returns:
            MLflow experiment URL or None if failed
        """
        if not self.config.enabled:
            logger.debug("MLflow integration is disabled")
            return None

        temp_dir = None
        try:
            # Create temporary directory for results using artifact_location
            temp_dir = Path(self.config.artifact_location) / f"temp_mlflow_results_{job_id}"
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Download and process evaluation results
            results_file = self.download_and_process_eval(
                eval_id=job_id,
                nmp_eval_uri=nmp_eval_uri,
                save_dir=temp_dir,
                model=model_name,
                eval_type=eval_type,
            )

            if results_file:
                # Upload to MLflow
                mlflow_uri = self.upload_result(
                    results_path=results_file,
                    flywheel_run_id=flywheel_run_id,
                    model_name=model_name,
                    eval_type=eval_type,
                    workload_type=workload_type,
                )
                logger.info(f"MLflow upload completed for {eval_type} evaluation")
                return mlflow_uri
            else:
                logger.warning(f"Failed to download results for MLflow upload: {job_id}")
                return None

        except Exception as e:
            logger.error(f"MLflow upload failed for {eval_type} evaluation: {e!s}")
            return None
        finally:
            # Clean up temporary directory
            if temp_dir and temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e!s}")

    def cleanup_experiment(self, experiment_name: str) -> bool:
        """
        Clean up MLflow experiment and all its runs, including artifacts.

        Args:
            experiment_name: Name of the experiment to clean up

        Returns:
            True if cleanup was successful, False otherwise
        """
        if not self.config.enabled:
            logger.debug("MLflow integration is disabled, skipping cleanup")
            return True

        try:
            # Get experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                logger.info(f"Experiment {experiment_name} not found, nothing to clean up")
                return True

            # Clean up artifacts first
            artifact_location = Path(self.config.artifact_location)
            if artifact_location.exists():
                # Find and remove experiment artifacts
                experiment_artifacts = artifact_location / experiment.experiment_id
                if experiment_artifacts.exists():
                    try:
                        shutil.rmtree(experiment_artifacts)
                        logger.info(f"Cleaned up experiment artifacts: {experiment_artifacts}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to clean up experiment artifacts {experiment_artifacts}: {e!s}"
                        )

            # Delete experiment (this will delete all runs in the experiment)
            mlflow.delete_experiment(experiment.experiment_id)
            logger.info(f"Successfully deleted experiment: {experiment_name}")
            return True

        except Exception as e:
            logger.error(f"Error cleaning up MLflow experiment {experiment_name}: {e!s}")
            return False
