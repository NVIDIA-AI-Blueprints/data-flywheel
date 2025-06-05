import wandb
from src.api.db_manager import TaskDBManager
from src.log_utils import setup_logging

logger = setup_logging("data_flywheel.wandb_callback")


# Create W&B progress callback if enabled
def make_wandb_progress_callback(manager: TaskDBManager, run, eval_instance, previous_result):
    
    def callback(update_data):
        """Update evaluation document with progress and log to W&B"""
        # Update MongoDB
        manager.update_evaluation(eval_instance.id, update_data)
        
        # Prepare W&B logging data with proper namespacing
        wandb_data = {
            # Core metrics
            "progress": eval_instance.progress,
            "runtime_seconds": eval_instance.runtime_seconds,
            "started_at": eval_instance.started_at,
            "finished_at": eval_instance.finished_at,
            "nmp_uri": eval_instance.nmp_uri,
        }
        
        # Update with any new values from update_data
        for key, value in update_data.items():
            if value is None:
                continue
                
            if key == 'scores':
                # Log each score individually with proper namespacing
                for score_name, score_value in value.items():
                    wandb_data[f"scores/{score_name}"] = score_value
                logger.info(f"Evaluation scores: {value}")
                
            elif key == 'progress':
                wandb_data["progress"] = value
                # Add step if available
                step = int(update_data.get('steps_completed', 0))
                run.log({"progress": value}, step=step)
                logger.info(f"Evaluation progress: {value}%")
                
            elif key == 'error':
                wandb_data["error"] = value
                logger.error(f"Evaluation error: {value}")
                
            else:
                wandb_data[key] = value
                logger.debug(f"Logged field {key}: {value}")
        
        # Log all data to W&B
        if wandb_data:
            run.log(wandb_data)
            
        # If this is a completion update, log final summary
        if update_data.get('finished_at'):
            summary_data = {
                "summary": {
                    "total_runtime": eval_instance.runtime_seconds,
                    "final_scores": eval_instance.scores,
                    "status": "completed" if not eval_instance.error else "failed",
                    "error": eval_instance.error
                }
            }
            
            # Add complete results if available
            if 'complete_results' in update_data:
                summary_data["complete_results"] = update_data['complete_results']
            
            run.log(summary_data)
            logger.info(f"Logged evaluation summary to W&B for {eval_instance.eval_type}")
    
    return callback