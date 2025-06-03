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
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class NMPConfig(BaseModel):
    """Configuration for NMP"""

    datastore_base_url: str = Field(..., frozen=True)
    nemo_base_url: str = Field(..., frozen=True)
    nim_base_url: str = Field(..., frozen=True)
    nmp_namespace: str = Field("dwfbp", frozen=True)


class DataSplitConfig(BaseModel):
    """Configuration for data split"""

    eval_size: int = Field(default=20, description="Size of evaluation set")
    val_ratio: float = Field(default=0.1, description="Validation ratio")
    min_total_records: int = Field(default=50, description="Minimum total records")
    random_seed: int | None = Field(None, description="Random seed")
    limit: int = Field(default=10000, description="Limit on number of records to evaluate")


class ICLConfig(BaseModel):
    """Configuration for ICL"""

    max_context_length: int = Field(default=8192, description="Maximum context length for ICL")
    reserved_tokens: int = Field(default=2048, description="Reserved tokens for ICL")
    max_examples: int = Field(default=3, description="Maximum examples for ICL")
    min_examples: int = Field(default=1, description="Minimum examples for ICL")


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


class NIMConfig(BaseModel):
    """Configuration for a NIM (Neural Information Model)"""

    model_name: str = Field(..., description="Name of the model")
    tag: str | None = Field(None, description="Container tag for the NIM")
    context_length: int = Field(..., description="Context length for ICL evaluations")
    gpus: int | None = Field(None, description="Number of GPUs for deployment")
    pvc_size: str | None = Field(None, description="Size of PVC for deployment")
    registry_base: str = Field(default="nvcr.io/nim", frozen=True)
    customization_enabled: bool = Field(default=False, description="Enable customization")

    def nmp_model_name(self) -> str:
        """Models names in NMP cannot have slashes, so we have to replace them with dashes."""
        return self.model_name.replace("/", "-")

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


class LLMJudgeConfig(BaseModel):
    type: str  # "remote" or "local"
    # Remote fields
    url: str | None = None
    model_id: str | None = None
    api_key_env: str | None = None
    api_key: str | None = None
    # Local fields
    model_name: str | None = None
    tag: str | None = None
    context_length: int | None = None
    gpus: int | None = None
    pvc_size: str | None = None
    registry_base: str | None = None
    customization_enabled: bool = False

    def is_remote(self) -> bool:
        return self.type == "remote"

    def is_local(self) -> bool:
        return self.type == "local"

    def get_remote_config(self) -> dict:
        if not self.is_remote():
            raise ValueError("Not a remote judge config")
        return {
            "api_endpoint": {
                "url": self.url,
                "model_id": self.model_id,
                "api_key": self.api_key,
            }
        }

    def get_local_nim_config(self) -> "NIMConfig":
        if not self.is_local():
            raise ValueError("Not a local judge config")
        return NIMConfig(
            model_name=self.model_name,
            tag=self.tag,
            context_length=self.context_length,
            gpus=self.gpus,
            pvc_size=self.pvc_size,
            registry_base=self.registry_base or "nvcr.io/nim",
            customization_enabled=self.customization_enabled,
        )

    @classmethod
    def from_yaml(cls, data: dict) -> "LLMJudgeConfig":
        api_key_env = data.get("api_key_env")
        api_key = os.environ.get(api_key_env) if api_key_env else None
        return cls(**data, api_key=api_key)


class WandbConfig(BaseModel):
    """Configuration for Weights & Biases"""
    api_key: str | None = Field(None, description="Weights & Biases API key")
    enabled: bool = Field(default=False, description="Whether WANDB logging is enabled")

    @classmethod
    def from_env(cls) -> "WandbConfig":
        """Create WandbConfig from environment variables"""
        api_key = os.getenv('WANDB_API_KEY', '').strip()
        return cls(
            api_key=api_key if api_key else None,
            enabled=bool(api_key)
        )


class Settings(BaseSettings):
    """Application settings loaded from environment variables and config file."""

    nmp_config: NMPConfig
    nims: list[NIMConfig]
    llm_judge_config: LLMJudgeConfig
    training_config: TrainingConfig
    data_split_config: DataSplitConfig
    icl_config: ICLConfig
    logging_config: LoggingConfig = Field(default_factory=LoggingConfig)
    wandb_config: WandbConfig = Field(default_factory=WandbConfig.from_env)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=True,
        extra="allow"
    )

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
            llm_judge_config = LLMJudgeConfig.from_yaml(config_data["llm_judge_config"])
            logging_config = (
                LoggingConfig(**config_data.get("logging_config", {}))
                if "logging_config" in config_data
                else LoggingConfig()
            )
            wandb_config = WandbConfig.from_env()

            return cls(
                nmp_config=NMPConfig(**config_data["nmp_config"]),
                nims=[NIMConfig(**nim) for nim in config_data["nims"]],
                llm_judge_config=llm_judge_config,
                training_config=training_config,
                data_split_config=DataSplitConfig(**config_data["data_split_config"]),
                icl_config=ICLConfig(**config_data["icl_config"]),
                logging_config=logging_config,
                wandb_config=wandb_config,
            )


# Load settings from config file
settings = Settings.from_yaml(Path(__file__).parent.parent / "config" / "config.yaml")
