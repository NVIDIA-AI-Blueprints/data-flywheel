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
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings


class NMPConfig(BaseModel):
    """Configuration for NMP"""

    datastore_base_url: str = Field(..., frozen=True)
    nemo_base_url: str = Field(..., frozen=True)
    nim_base_url: str = Field(..., frozen=True)
    nmp_namespace: str = Field("dwfbp", frozen=True)


class DataSplitConfig(BaseModel):
    """Configuration for data split"""

    eval_size: int = Field(default=20, description="Size of evaluation set")
    val_ratio: float = Field(
        default=0.1,
        description="Validation ratio",
        ge=0,
        lt=1,
    )
    min_total_records: int = Field(default=50, description="Minimum total records")
    random_seed: int | None = Field(None, description="Random seed")
    limit: int = Field(default=10000, description="Limit on number of records to evaluate")
    parse_function_arguments: bool = Field(
        default=True, description="Data Validation: Parse function arguments to JSON"
    )


class SimilarityConfig(BaseModel):
    """Configuration for semantic similarity-based ICL example selection"""

    relevance_ratio: float | None = Field(
        default=0.7,
        description="Ratio of examples selected by pure relevance (0.7 = 70% relevance, 30% coverage). Only used with semantic_similarity.",
        ge=0.0,
        le=1.0,
    )
    embedding_nim_config: "EmbeddingConfig | None" = Field(
        default=None, description="Configuration for embedding NIM when using semantic_similarity"
    )


class ICLConfig(BaseModel):
    """Configuration for ICL"""

    max_context_length: int = Field(default=8192, description="Maximum context length for ICL")
    reserved_tokens: int = Field(default=2048, description="Reserved tokens for ICL")
    max_examples: int = Field(default=3, description="Maximum examples for ICL")
    min_examples: int = Field(default=1, description="Minimum examples for ICL")
    example_selection: Literal["uniform_distribution", "semantic_similarity"] = Field(
        default="uniform_distribution", description="ICL example selection method"
    )
    similarity_config: "SimilarityConfig | None" = Field(
        default=None, description="Configuration for semantic similarity example selection"
    )

    @model_validator(mode="after")
    def validate_semantic_similarity_config(self) -> "ICLConfig":
        """Validate that similarity_config is provided when semantic_similarity is selected."""
        if self.example_selection == "semantic_similarity":
            if self.similarity_config is None:
                raise ValueError(
                    "similarity_config is required when example_selection is set to 'semantic_similarity'. "
                    "Please provide similarity_config in your ICL configuration."
                )
            if self.similarity_config.embedding_nim_config is None:
                raise ValueError(
                    "embedding_nim_config is required within similarity_config when example_selection is set to 'semantic_similarity'. "
                    "Please provide embedding_nim_config in your similarity_config."
                )
        return self

    def validate_examples_limit(self, eval_size: int) -> None:
        """Validate that max_examples doesn't exceed the evaluation dataset size."""
        if self.max_examples > eval_size:
            raise ValueError(
                f"ICL max_examples ({self.max_examples}) cannot exceed "
                f"data split eval_size ({eval_size}). "
            )


class LoRAConfig(BaseModel):
    adapter_dim: int = Field(default=32, description="Adapter dimension")
    adapter_dropout: float = Field(default=0.1, description="Adapter dropout")


class TrainingConfig(BaseModel):
    training_type: str = Field(default="sft", description="Training type")
    finetuning_type: str = Field(default="lora", description="Finetuning type")
    epochs: int = Field(default=2, description="Number of epochs")
    batch_size: int = Field(default=16, description="Batch size")
    learning_rate: float = Field(default=0.0001, description="Learning rate")
    lora: LoRAConfig = Field(default_factory=LoRAConfig)


class LoggingConfig(BaseModel):
    """Configuration for logging"""

    level: str = "INFO"


class MLflowConfig(BaseModel):
    """Configuration for MLflow integration"""

    enabled: bool = Field(
        default_factory=lambda: "mlflow" in os.getenv("COMPOSE_PROFILES", "").split(","),
        description="Enable MLflow integration based on COMPOSE_PROFILES environment variable",
    )
    tracking_uri: str = Field(default="http://0.0.0.0:5000", description="MLflow tracking URI")
    experiment_name_prefix: str = Field(
        default="data-flywheel", description="Prefix for experiment names"
    )
    artifact_location: str = Field(default="./mlruns", description="Location for MLflow artifacts")


class CustomizerConfig(BaseModel):
    """Configuration for model customization"""

    target: str = Field(..., description="Target model for customization")
    gpus: int = Field(..., description="Number of GPUs for customization")
    num_nodes: int = Field(default=1, description="Number of nodes for customization")
    tensor_parallel_size: int = Field(
        default=1, description="Tensor parallel size for customization"
    )
    data_parallel_size: int = Field(default=1, description="Data parallel size for customization")
    use_sequence_parallel: bool = Field(
        default=False, description="Whether to use sequence parallel"
    )
    micro_batch_size: int = Field(default=1, description="Micro batch size for customization")
    training_precision: str = Field(default="bf16-mixed", description="Training precision")
    max_seq_length: int = Field(default=4096, description="Maximum sequence length")


class NIMConfig(BaseModel):
    """Configuration for a NIM (Neural Information Model)"""

    model_name: str = Field(..., description="Name of the model")
    tag: str | None = Field(None, description="Container tag for the NIM")
    context_length: int = Field(..., description="Context length for ICL evaluations")
    gpus: int | None = Field(None, description="Number of GPUs for deployment")
    pvc_size: str | None = Field(None, description="Size of PVC for deployment")
    registry_base: str = Field(default="nvcr.io/nim", frozen=True)
    customization_enabled: bool = Field(default=False, description="Enable customization")
    customizer_configs: CustomizerConfig | None = Field(
        default=None, description="Customizer configuration"
    )
    model_type: Literal["llm", "embedding"] = Field(
        default="llm", description="Type of NIM - llm or embedding"
    )

    @model_validator(mode="after")
    def validate_customization_config(self) -> "NIMConfig":
        """Validate that customizer_configs is provided when customization is enabled."""
        if self.customization_enabled and self.customizer_configs is None:
            raise ValueError(
                "customizer_configs is required when customization_enabled is set to True. "
                "Please provide customizer_configs in your NIM configuration."
            )
        return self

    def nmp_model_name(self) -> str:
        """Models names in NMP cannot have slashes, so we have to replace them with dashes."""
        return self.model_name.replace("/", "-")

    @staticmethod
    def generate_config_name(base_model: str) -> str:
        """
        Generate consistent config name from base model.

        Args:
            base_model: Base model name in format "namespace/model"

        Returns:
            Configuration name in format "model@v1.0.0+dfw"

        Raises:
            ValueError: If base_model format is invalid
        """
        if base_model is None:
            raise ValueError("Invalid base model format")

        model_parts = base_model.split("/")
        if len(model_parts) != 2:
            raise ValueError(f"Invalid base model format: {base_model}")

        _, model = model_parts
        return f"{model}@v1.0.0+dfw"

    def to_dms_config(self) -> dict[str, Any]:
        """Convert NIMConfig to DMS deployment configuration."""
        return {
            "name": self.nmp_model_name(),
            "namespace": settings.nmp_config.nmp_namespace,
            "config": {
                "model": self.model_name,
                "nim_deployment": {
                    "image_name": f"{self.registry_base}/{self.model_name}",
                    "image_tag": self.tag,
                    "pvc_size": self.pvc_size,
                    "gpu": self.gpus,
                    "additional_envs": {
                        # NIMs can have different default
                        # GD backends. `outlines` is the
                        # best for tasks that utilize
                        # structured responses.
                        "NIM_GUIDED_DECODING_BACKEND": "outlines",
                    },
                },
            },
        }

    def target_model_for_evaluation(self) -> str | dict[str, Any]:
        """Get the model name for evaluation"""
        return self.model_name


class EmbeddingConfig(NIMConfig):
    """Configuration for embedding models"""

    # Override the nim type to be embedding
    model_type: Literal["llm", "embedding"] = Field(
        default="embedding", description="Type of NIM - always embedding for EmbeddingConfig"
    )

    # Deployment type for embedding models
    deployment_type: Literal["remote", "local"]

    # Remote fields
    url: str | None = None
    api_key: str | None = None
    context_length: int | None = None  # overwrite NIMConfig to be optional
    model_name: str | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )

    @property
    def is_remote(self) -> bool:
        return self.deployment_type == "remote"

    @classmethod
    def remote_config(cls, data: dict, api_key: str) -> "EmbeddingConfig":
        """Get configuration based on type"""
        return cls(
            url=data.get("url"),
            deployment_type="remote",
            model_name=data.get("model_name"),
            api_key=api_key,
        )

    @classmethod
    def local_config(cls, data: dict) -> "EmbeddingConfig":
        return cls(
            model_name=data.get("model_name"),
            deployment_type="local",
            tag=data.get("tag"),
            context_length=data.get("context_length"),
            gpus=data.get("gpus"),
            pvc_size=data.get("pvc_size"),
            registry_base=data.get("registry_base") or "nvcr.io/nim",
            customization_enabled=False,  # customization should be disabled for embedding
            url=data.get("url"),  # Remove hardcoded localhost URL
        )

    def get_endpoint_url(self) -> str:
        """Get the endpoint URL for the embedding service"""
        if self.is_remote:
            return self.url
        else:
            return getattr(self, "url", "http://localhost:8000/v1/embeddings")

    @classmethod
    def from_json(cls, data: dict) -> "EmbeddingConfig":
        # Read API key directly from EMB_API_KEY environment variable, default to NVIDIA_API_KEY
        api_key = os.environ.get("EMB_API_KEY") or os.environ.get("NVIDIA_API_KEY")
        is_remote = data.get("deployment_type") == "remote"
        return cls.remote_config(data, api_key) if is_remote else cls.local_config(data)


class LLMJudgeConfig(NIMConfig):
    deployment_type: Literal["remote", "local"]
    # Remote fields
    url: str | None = None
    api_key: str | None = None
    context_length: int | None = None  # overwrite NIMConfig to be optional
    model_name: str | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )

    @property
    def is_remote(self) -> bool:
        return self.deployment_type == "remote"

    @classmethod
    def remote_config(cls, data: dict, api_key: str) -> "LLMJudgeConfig":
        """Get configuration based on type"""
        return cls(
            url=data.get("url"),
            deployment_type="remote",
            model_name=data.get("model_name"),
            api_key=api_key,
        )

    @classmethod
    def local_config(cls, data: dict) -> "LLMJudgeConfig":
        return cls(
            model_name=data.get("model_name"),
            deployment_type="local",
            tag=data.get("tag"),
            context_length=data.get("context_length"),
            gpus=data.get("gpus"),
            pvc_size=data.get("pvc_size"),
            registry_base=data.get("registry_base") or "nvcr.io/nim",
            customization_enabled=False,  # customization should be disabled for local LLM judge
        )

    def judge_model_config(self) -> dict[str, Any]:
        if self.is_remote:
            return {
                "api_endpoint": {
                    "url": self.url,
                    "model_id": self.model_name,
                    "api_key": self.api_key,
                },
            }
        else:
            return self.model_name

    @classmethod
    def from_json(cls, data: dict) -> "LLMJudgeConfig":
        # Read API key directly from LLM_JUDGE_API_KEY environment variable, default to NVIDIA_API_KEY
        api_key = os.environ.get("LLM_JUDGE_API_KEY") or os.environ.get("NVIDIA_API_KEY")
        is_remote = data.get("deployment_type") == "remote"
        return cls.remote_config(data, api_key) if is_remote else cls.local_config(data)


class Settings(BaseSettings):
    """Application settings loaded from environment variables and config file."""

    nmp_config: NMPConfig
    nims: list[NIMConfig]
    llm_judge_config: LLMJudgeConfig
    training_config: TrainingConfig
    data_split_config: DataSplitConfig
    icl_config: ICLConfig
    logging_config: LoggingConfig = Field(default_factory=LoggingConfig)
    mlflow_config: MLflowConfig = Field(default_factory=MLflowConfig)

    @model_validator(mode="after")
    def validate_icl_and_data_split_consistency(self) -> "Settings":
        """Validate that max_examples in ICL config doesn't exceed eval_size in data split config."""
        self.icl_config.validate_examples_limit(self.data_split_config.eval_size)
        return self

    @model_validator(mode="after")
    def validate_nims_not_empty(self) -> "Settings":
        """Validate that at least one NIM is configured."""
        if not self.nims or len(self.nims) == 0:
            raise ValueError(
                "At least one NIM must be configured. "
                "Please add at least one NIM to the 'nims' list in your configuration."
            )
        return self

    def get_api_key(self, env_var: str) -> str | None:
        """Get API key from environment variable."""
        return os.getenv(env_var)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "Settings":
        """Load settings from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            config_data = yaml.safe_load(f)
            lora_config = LoRAConfig(**config_data["lora_config"])
            training_config = TrainingConfig(**config_data["training_config"], lora=lora_config)
            llm_judge_config = LLMJudgeConfig.from_json(config_data["llm_judge_config"])
            logging_config = (
                LoggingConfig(**config_data.get("logging_config", {}))
                if "logging_config" in config_data
                else LoggingConfig()
            )
            mlflow_config = (
                MLflowConfig(**config_data.get("mlflow_config", {}))
                if "mlflow_config" in config_data
                else MLflowConfig()
            )

            # Deduplicate NIMs by model_name
            # we should have only unique NIMs in the config
            # will pick up the first one if there are duplicates
            seen_models = set()
            unique_nims = []
            for nim in config_data["nims"]:
                if nim["model_name"] not in seen_models:
                    unique_nims.append(nim)
                    seen_models.add(nim["model_name"])

            # Handle ICL config with similarity config
            icl_data = config_data["icl_config"]
            if icl_data.get("similarity_config") and icl_data["similarity_config"].get(
                "embedding_nim_config"
            ):
                icl_data["similarity_config"]["embedding_nim_config"] = EmbeddingConfig.from_json(
                    icl_data["similarity_config"]["embedding_nim_config"]
                )
                icl_data["similarity_config"] = SimilarityConfig(**icl_data["similarity_config"])

            return cls(
                nmp_config=NMPConfig(**config_data["nmp_config"]),
                nims=[NIMConfig(**nim) for nim in unique_nims],
                llm_judge_config=llm_judge_config,
                training_config=training_config,
                data_split_config=DataSplitConfig(**config_data["data_split_config"]),
                icl_config=ICLConfig(**icl_data),
                logging_config=logging_config,
                mlflow_config=mlflow_config,
            )


# Load settings from config file
settings = Settings.from_yaml(Path(__file__).parent.parent / "config" / "config.yaml")
