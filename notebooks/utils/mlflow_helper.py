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

import requests
import zipfile
from pathlib import Path
import mlflow
import pandas as pd
import json

def download_and_process_eval(eval_id: str, nmp_eval_uri: str, save_dir: Path, 
                             model: str, eval_type: str) -> bool:
    """Downloads evaluation results and extracts ZIP contents to save_dir."""
    # Create output directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Download results
    url = f"{nmp_eval_uri}/download-results"
    response = requests.get(url, headers={'accept': 'application/json'})
    
    if not response.ok:
        print(f"Download failed [{response.status_code}]: {response.text}")
        return False

    # Save ZIP file
    zip_path = save_dir / f"result_{eval_id}.zip"
    zip_path.write_bytes(response.content)
    print(f"Downloaded results to {zip_path}")

    # Extract ZIP contents
    with zipfile.ZipFile(zip_path) as zip_file:
        zip_file.extractall(save_dir)
        print(f"Extracted {len(zip_file.namelist())} files")

    # Cleanup ZIP
    zip_path.unlink()
    
    # Find and rename results.json
    results_json = next(save_dir.glob("**/results.json"), None)
    if results_json:
        # Sanitize model name to avoid directory separators in filename
        safe_model_name = model.replace('/', '_').replace('\\', '_')
        new_name = save_dir / f"{safe_model_name}_{eval_type}.json"
        results_json.rename(new_name)
        print(f"Renamed results to {new_name}")
        print(f"Successfully processed {eval_id}")
        return new_name
        
    print("Error: results.json not found in extracted files")
    print(f"Failed to process {eval_id}")
    return False


def load_results(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_metrics(item):
    metrics = item.get('metrics', {})
    sample = item.get('sample', {})
    usage = sample.get('response', {}).get('usage', {})
    return {
        'function_name_accuracy': metrics.get('tool-calling-accuracy', {}).get('scores', {}).get('function_name_accuracy', {}).get('value'),
        'function_name_and_args_accuracy': metrics.get('tool-calling-accuracy', {}).get('scores', {}).get('function_name_and_args_accuracy', {}).get('value'),
        'correctness_rating': metrics.get('correctness', {}).get('scores', {}).get('rating', {}).get('value'),
        'total_tokens': usage.get('total_tokens'),
        'prompt_tokens': usage.get('prompt_tokens'),
        'completion_tokens': usage.get('completion_tokens')
    }

def extract_metadata(item):
    sample = item.get('sample', {})
    response = sample.get('response', {})
    return {
        'model': response.get('model'),
        'workload_id': item.get('item', {}).get('workload_id'),
        'client_id': item.get('item', {}).get('client_id'),
        'timestamp': item.get('item', {}).get('timestamp')
    }

def upload_result_to_mlflow(
    results_path: Path,
    tracking_uri: str = "http://0.0.0.0:5000"
):
    # Parse experiment name and run name from path
    experiment_name = results_path.parent.name  # flywheel job id
    run_name = results_path.stem  # file stem

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(tracking_uri)

    # Get or create experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new experiment: {experiment_name}")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")

    # Load results
    results = load_results(results_path)

    # Group by model (if needed)
    model_evaluations = {}
    for item in results.get('custom-tool-calling', []):
        metadata = extract_metadata(item)
        model_name = metadata['model']
        model_evaluations.setdefault(model_name, []).append(item)

    # Print summary of what will be uploaded
    print(f"\n📊 Uploading evaluation results to MLflow:")
    print(f"   📁 Experiment: {experiment_name}")
    print(f"   📄 Results file: {results_path.name}")
    print(f"   🤖 Models found: {len(model_evaluations)}")
    for model_name, evaluations in model_evaluations.items():
        print(f"      - {model_name}: {len(evaluations)} evaluation samples")

    # Upload each model's results as a separate run (if multiple models)
    for model_name, evaluations in model_evaluations.items():
        this_run_name = f"{run_name}_{model_name.replace('/', '_')}" if len(model_evaluations) > 1 else run_name
        with mlflow.start_run(experiment_id=experiment_id, run_name=this_run_name):
            mlflow.log_param("model", model_name)
            mlflow.log_param("run_name", run_name)
            mlflow.log_param("experiment_name", experiment_name)
            metrics_data = []
            for item in evaluations:
                metrics = extract_metrics(item)
                metadata = extract_metadata(item)
                metrics_data.append({'timestamp': metadata['timestamp'], **metrics})
            metrics_df = pd.DataFrame(metrics_data).sort_values('timestamp')
            for step, (_, row) in enumerate(metrics_df.iterrows()):
                mlflow.log_metrics(row.drop('timestamp').to_dict(), step=step)
            # Log summary statistics
            summary_metrics = {
                f"mean_{col}": metrics_df[col].mean()
                for col in ['function_name_accuracy', 'function_name_and_args_accuracy', 'correctness_rating', 'total_tokens', 'prompt_tokens', 'completion_tokens']
                if col in metrics_df
            }
            summary_metrics['total_evaluations'] = len(evaluations)
            mlflow.log_metrics(summary_metrics)
        print(f"✅ Uploaded run '{this_run_name}' to experiment '{experiment_name}'")
