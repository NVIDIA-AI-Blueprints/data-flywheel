from datetime import datetime
from typing import Optional, Dict, Any
import requests
from src.log_utils import setup_logging
import wandb
from src.config import settings

logger = setup_logging("data_flywheel.model_manager")

class ModelManager:
    """Manages model information and W&B model registry integration."""
    
    def __init__(self, nmp_config):
        self.nmp_config = nmp_config
        self.base_url = nmp_config.nemo_base_url
        
    def get_model_info(self, namespace: str, model_name: str) -> Dict[str, Any]:
        """Retrieve model information from the NMP API.
        
        Args:
            namespace: The namespace of the model
            model_name: The name of the model. If it already contains the namespace (e.g. "namespace/model_name"),
                       the namespace parameter will be ignored.
            
        Returns:
            Dict containing model information
        """
        # Check if model_name already contains namespace
        if '/' in model_name:
            namespace, model_name = model_name.split('/', 1)
            
        url = f"{self.base_url}/v1/models/{namespace}/{model_name}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error retrieving model info for {model_name}: {str(e)}")
            raise
            
    def register_model_to_wandb(
        self,
        model_name: str,
        namespace: str,
        run: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a model to W&B model registry.
        
        Args:
            model_name: Name of the model. If it already contains the namespace (e.g. "namespace/model_name"),
                       the namespace parameter will be ignored.
            namespace: Namespace of the model
            run: W&B run to associate with the model
            metadata: Additional metadata to store with the model
        """
        try:
            # Check if model_name already contains namespace
            if '/' in model_name:
                namespace, model_name = model_name.split('/', 1)
                
            # Get model info from NMP
            model_info = self.get_model_info(namespace, model_name)
            
            # Sanitize model name for W&B artifact (replace @ with _)
            sanitized_model_name = model_name.replace('@', '_')
            
            # Prepare model metadata
            model_metadata = {
                "name": model_name,  # Keep original name in metadata
                "namespace": namespace,
                "created_at": model_info.get("created_at"),
                "updated_at": model_info.get("updated_at"),
                "description": model_info.get("description"),
                "base_model": model_info.get("base_model"),
                "spec": model_info.get("spec", {}),
                "artifact": model_info.get("artifact", {}),
                "api_endpoint": model_info.get("api_endpoint", {}),
                "peft": model_info.get("peft", {}),
                "prompt": model_info.get("prompt", {}),
                "guardrails": model_info.get("guardrails", {}),
            }
            
            # Add any additional metadata
            if metadata:
                model_metadata.update(metadata)
                
            # Create model in W&B with sanitized name
            model = wandb.Artifact(
                name=f"model-{sanitized_model_name}",
                type="model",
                description=f"Model {model_name} from namespace {namespace}",
                metadata=model_metadata
            )
            
            # Add model to run
            #TODO: Grab the hf file from the model_info and add it to the model artifact
            #Example: 'files_url': 'hf://dfwbp/customized-meta-llama-3.2-1b-instruct@cust-Cc8ETHRdZg6iGFwdEAmjeL'},
            run.log_artifact(model)
            logger.info(f"Registered model {model_name} to W&B model registry")
            
        except Exception as e:
            logger.error(f"Error registering model {model_name} to W&B: {str(e)}")
            raise
            
    def link_evals_to_model(
        self,
        model_name: str,
        namespace: str,
        eval_run: Any
    ) -> None:
        """Link evaluation runs to a model artifact in W&B.

        Args:
            model_name: Name of the model. If it already contains the namespace (e.g. "namespace/model_name"),
                       the namespace parameter will be ignored.
            namespace: Namespace of the model
            eval_run: W&B run for evaluation
        """
        try:
            # Check if model_name already contains namespace
            if '/' in model_name:
                namespace, model_name = model_name.split('/', 1)

            # Sanitize model name for W&B artifact
            sanitized_model_name = model_name.replace('@', '_')

            # Reference the model artifact
            artifact_name = f"model-{sanitized_model_name}:latest"
            api = wandb.Api()
            artifact = api.artifact(
                f"{settings.wandb_config.entity}/{settings.wandb_config.project}/{artifact_name}",
                type="model"
            )

            # Link evaluation runs
            eval_run.use_artifact(artifact)

            logger.info(f"Linked eval runs to model {model_name} in W&B")

        except Exception as e:
            logger.error(f"Error linking eval runs to model {model_name}: {str(e)}")
            raise
        