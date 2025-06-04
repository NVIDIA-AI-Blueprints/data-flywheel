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
            "evaluation": {
                "progress": eval_instance.progress,
                "runtime_seconds": eval_instance.runtime_seconds,
                "started_at": eval_instance.started_at,
                "finished_at": eval_instance.finished_at,
                "nmp_uri": eval_instance.nmp_uri,
            }
        }
        
        # Update wandb_data with any new values from update_data
        for key, value in update_data.items():
            if value is None:
                continue
                
            if key == 'scores':
                # Log each score individually with proper namespacing
                for score_name, score_value in value.items():
                    wandb_data[f"evaluation/scores/{score_name}"] = score_value
                logger.info(f"Evaluation scores: {value}")
                
            elif key == 'progress':
                wandb_data["evaluation/progress"] = value
                # Add step if available
                step = int(update_data.get('steps_completed', 0))
                run.log({"evaluation/progress": value}, step=step)
                logger.info(f"Evaluation progress: {value}%")
                
            elif key == 'error':
                wandb_data["evaluation/error"] = value
                logger.error(f"Evaluation error: {value}")
                
            else:
                # Handle any other fields by adding them to the evaluation namespace
                wandb_data[f"evaluation/{key}"] = value
                logger.debug(f"Logged evaluation field {key}: {value}")
        
        # Log all data to W&B in a single call
        #TODO: Fix wandb: WARNING Tried to log to step 0 that is less than the current step 25. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
        if wandb_data:
            run.log(wandb_data)
            
        # If this is a completion update, log final summary
        if update_data.get('finished_at'):
            run.log({
                "evaluation/summary": {
                    "total_runtime": eval_instance.runtime_seconds,
                    "final_scores": eval_instance.scores,
                    "status": "completed" if not eval_instance.error else "failed",
                    "error": eval_instance.error
                }
            })
            logger.info(f"Logged evaluation summary to W&B for {eval_instance.eval_type}")
    
    return callback