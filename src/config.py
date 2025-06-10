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


class LLMJudgeConfig(NIMConfig):
    type: Literal["remote", "local"]
    # Remote fields
    url: str | None = None
    api_key_env: str | None = None
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
        return self.type == "remote"

    @classmethod
    def remote_config(cls, data: dict, api_key: str) -> "LLMJudgeConfig":
        """Get configuration based on type"""
        return cls(
            url=data.get("url"),
            type="remote",
            model_name=data.get("model_name"),
            api_key=api_key,
        )

    @classmethod
    def local_config(cls, data: dict) -> "LLMJudgeConfig":
        return cls(
            model_name=data.get("model_name"),
            type="local",
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
        api_key_env = data.get("api_key_env")
        api_key = os.environ.get(api_key_env) if api_key_env else None
        is_remote = data.get("type") == "remote"
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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=True,
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
            llm_judge_config = LLMJudgeConfig.from_json(config_data["llm_judge_config"])
            logging_config = (
                LoggingConfig(**config_data.get("logging_config", {}))
                if "logging_config" in config_data
                else LoggingConfig()
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

            return cls(
                nmp_config=NMPConfig(**config_data["nmp_config"]),
                nims=[NIMConfig(**nim) for nim in unique_nims],
                llm_judge_config=llm_judge_config,
                training_config=training_config,
                data_split_config=DataSplitConfig(**config_data["data_split_config"]),
                icl_config=ICLConfig(**config_data["icl_config"]),
                logging_config=logging_config,
            )


# Load settings from config file
settings = Settings.from_yaml(Path(__file__).parent.parent / "config" / "config.yaml")
