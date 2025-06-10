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

import matplotlib.pyplot as plt
import pandas as pd
import time
import requests
from datetime import datetime
from IPython.display import clear_output
from pathlib import Path
from notebooks.utils.mlflow_helper import download_and_process_eval, upload_result_to_mlflow

def format_runtime(seconds):
    """Format runtime in seconds to a human-readable string."""
    if seconds is None:
        return "-"
    minutes, seconds = divmod(seconds, 60)
    if minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    return f"{int(seconds)}s"

def create_results_table(job_data):
    """Create a pandas DataFrame from job data."""
    rows = []
    for nim in job_data["nims"]:
        model_name = nim["model_name"]
        for eval in nim["evaluations"]:
            all_scores = eval["scores"]
            
            row = {
                "Model": model_name,
                "Eval Type": eval["eval_type"].upper(),
                "Percent Done": eval["progress"],
                "Runtime": format_runtime(eval["runtime_seconds"]),
                "Status": "Completed" if eval["finished_at"] else "Running",
                "Started": datetime.fromisoformat(eval["started_at"]).strftime("%H:%M:%S"),
                "Finished": datetime.fromisoformat(eval["finished_at"]).strftime("%H:%M:%S") if eval["finished_at"] else "-"
            }
            
            if "function_name" in all_scores:
                row["Function name accuracy"] = all_scores["function_name"]
            
            if "function_name_and_args_accuracy" in all_scores:
                row["Function name + args accuracy (exact-match)"] = all_scores["function_name_and_args_accuracy"]
                
            if "tool_calling_correctness" in all_scores:
                row["Function name + args accuracy (LLM-judge)"] = all_scores["tool_calling_correctness"]
            
            # Add any other scores with formatted names
            for score_name, score_value in all_scores.items():
                if score_name not in ["function_name", "tool_calling_correctness", "similarity", "function_name_and_args_accuracy"]:
                    formatted_name = score_name.replace("_", " ").title()
                    row[formatted_name] = score_value
            
            rows.append(row)
    
    if not rows:
        return pd.DataFrame(columns=["Model", "Eval Type", "Function Name Accuracy", "Tool Calling Correctness (LLM-Judge)", "Similarity (LLM-Judge)", "Percent Done", "Runtime", "Status", "Started", "Finished"])
    
    df = pd.DataFrame(rows)
    return df.sort_values(["Model", "Eval Type"])

def create_customization_table(job_data):
    """Create a pandas DataFrame from customization data."""
    customizations = []
    for nim in job_data["nims"]:
        model_name = nim["model_name"]
        for custom in nim["customizations"]:
            customizations.append({
                "Model": model_name,
                "Started": datetime.fromisoformat(custom["started_at"]).strftime("%H:%M:%S"),
                "Epochs Completed": custom["epochs_completed"],
                "Steps Completed": custom["steps_completed"],
                "Finished": datetime.fromisoformat(custom["finished_at"]).strftime("%H:%M:%S") if custom["finished_at"] else "-",
                "Status": "Completed" if custom["finished_at"] else "Running",
                "Runtime": format_runtime(custom["runtime_seconds"]),
                "Percent Done": custom["progress"],
            })
   
    if not customizations:
        customizations = pd.DataFrame(columns=["Model", "Started", "Epochs Completed", "Steps Completed", "Finished", "Runtime", "Percent Done"])
    customizations = pd.DataFrame(customizations)
    return customizations.sort_values(["Model"])

def get_job_status(api_base_url, job_id):
    """Get the current status of a job."""
    response = requests.get(f"{api_base_url}/api/jobs/{job_id}")
    response.raise_for_status()
    return response.json()

def monitor_job(api_base_url, job_id, poll_interval, mlflow_uri=None, enable_mlflow_upload=False):
    """Monitor a job and display its progress in a table.
    
    Args:
        api_base_url: Base URL for the API
        job_id: Job ID to monitor
        poll_interval: Polling interval in seconds
        mlflow_uri: MLflow tracking URI (required if enable_mlflow_upload=True)
        enable_mlflow_upload: Whether to enable automatic MLflow upload for completed evaluations
    """
    print(f"Monitoring job {job_id}...")
    print("Press Ctrl+C to stop monitoring")
    
    if enable_mlflow_upload:
        if not mlflow_uri:
            raise ValueError("mlflow_uri is required when enable_mlflow_upload=True")
        print(f"MLflow integration enabled - will upload completed evaluations to {mlflow_uri}")
    
    # Track completed evaluations to avoid duplicate uploads
    completed_evaluations = set()
    
    while True:
        try:
            clear_output(wait=True)
            job_data = get_job_status(api_base_url, job_id)
            
            # Check for newly completed evaluations and trigger MLflow upload
            if enable_mlflow_upload:
                print("Uploading completed evaluations to MLflow...")
                for nim in job_data["nims"]:
                    model_name = nim["model_name"]
                    for eval_data in nim["evaluations"]:
                        # Extract eval_id from nmp_uri
                        nmp_eval_uri = eval_data.get('nmp_uri', '')
                        eval_id = nmp_eval_uri.split('/')[-1] if nmp_eval_uri else None
                        
                        # Create unique key for this evaluation
                        eval_key = f"{model_name}_{eval_data['eval_type']}_{eval_id or 'unknown'}"
                        
                        # Check if evaluation is completed and not already processed
                        if eval_data.get("finished_at") and eval_key not in completed_evaluations:
                            completed_evaluations.add(eval_key)
                            
                            if nmp_eval_uri:
                                print(f"\n🔄 Processing completed evaluation: {eval_key}")
                                try:
                                    # Download results
                                    results_json = download_and_process_eval(
                                        eval_id=eval_id,
                                        nmp_eval_uri=nmp_eval_uri,
                                        save_dir=Path("results") / job_id,
                                        model=model_name,
                                        eval_type=eval_data['eval_type']
                                    )
                                    
                                    # Upload to MLflow
                                    if results_json:
                                        upload_result_to_mlflow(
                                            results_json,
                                            tracking_uri=mlflow_uri
                                        )
                                        print(f"✅ Successfully uploaded {eval_key} to MLflow")
                                    else:
                                        print(f"❌ Failed to download results for {eval_key}")
                                        
                                except Exception as e:
                                    print(f"❌ Error processing {eval_key}: {str(e)}")
                            else:
                                print(f"⚠️ No nmp_uri or eval_id found for completed evaluation: {eval_key}")
            
            results_df = create_results_table(job_data)
            customizations_df = create_customization_table(job_data)
            clear_output(wait=True)
            print(f"Job Status: {job_data['status']}")
            print(f"Total Records: {job_data['num_records']}")
            print(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}")
            print("\nResults:")
            display(results_df)
            print("\nCustomizations:")
            display(customizations_df)
            display(job_data)

            # Plot 1: Evaluation Scores
            if not results_df.empty:
                metrics = [
                    "Function name accuracy",
                    "Function name + args accuracy (exact-match)",
                    "Function name + args accuracy (LLM-judge)"
                ]
            
                models = results_df["Model"].unique()
            
                for model in models:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    model_df = results_df[results_df["Model"] == model]
                    if not all(metric in model_df.columns for metric in metrics):
                        continue  # skip this model for now                 
                    plot_df = model_df.set_index("Eval Type")[metrics].T
            
                    # Plot bar chart for this model
                    plot_df.plot(kind="bar", ax=ax)
                    ax.set_title(f"Evaluation results for {model}", fontsize=12)
                    ax.set_ylabel("Score")
                    ax.set_ylim(0, 1)
                    ax.legend(title="Eval Type")
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.xticks(rotation=30)
                    plt.tight_layout()
                    plt.show()
            else:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.text(0.5, 0.5, "No Evaluation Data", ha='center', va='center')
                ax.set_axis_off()
                plt.tight_layout()
                plt.show()

            plt.tight_layout()
            plt.show()                        
            time.sleep(poll_interval)

            # Check if job is completed or failed
            if job_data['status'] in ['completed', 'failed']:
                # print(f"\nJob monitoring complete! Final status: {job_data['status']}")
                if job_data['status'] == 'failed':
                    print("Job failed - check error details above")
                    if job_data.get('error'):
                        print(f"Error: {job_data['error']}")
                else:
                    print("Job completed successfully!")
                break

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            break
