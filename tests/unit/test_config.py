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

import pytest
from pydantic import ValidationError

from src.config import (
    DataSplitConfig,
    EmbeddingConfig,
    ICLConfig,
    LLMJudgeConfig,
    LoggingConfig,
    LoRAConfig,
    MLflowConfig,
    NIMConfig,
    NMPConfig,
    Settings,
    SimilarityConfig,
    TrainingConfig,
)


@pytest.fixture
def base_configs():
    """Base configurations for Settings creation."""
    return {
        "nmp_config": NMPConfig(
            datastore_base_url="http://test-datastore",
            nemo_base_url="http://test-nemo",
            nim_base_url="http://test-nim",
        ),
        "nims": [
            NIMConfig(
                model_name="test/model",
                context_length=8192,
                model_type="llm",
            )
        ],
        "llm_judge_config": LLMJudgeConfig(
            model_name="test-judge",
            context_length=8192,
            deployment_type="local",
        ),
        "training_config": TrainingConfig(lora=LoRAConfig()),
        "logging_config": LoggingConfig(),
        "mlflow_config": MLflowConfig(),
    }


class TestICLConfig:
    """Test ICLConfig validation."""

    def test_semantic_similarity_requires_embedding_config(self):
        """Test that semantic similarity selection requires embedding config."""
        with pytest.raises(ValidationError, match="similarity_config is required"):
            ICLConfig(example_selection="semantic_similarity", similarity_config=None)

    def test_semantic_similarity_with_embedding_config_succeeds(self):
        """Test that semantic similarity works with embedding config provided."""
        embedding_config = EmbeddingConfig(
            deployment_type="local",
            model_name="test-model",
            context_length=1024,
            gpus=1,
            pvc_size="10Gi",
            tag="latest",
        )

        similarity_config = SimilarityConfig(embedding_nim_config=embedding_config)

        config = ICLConfig(
            example_selection="semantic_similarity", similarity_config=similarity_config
        )

        assert config.example_selection == "semantic_similarity"
        assert config.similarity_config.embedding_nim_config is not None

    def test_uniform_distribution_no_embedding_config_required(self):
        """Test that uniform distribution doesn't require embedding config."""
        config = ICLConfig(example_selection="uniform_distribution")
        assert config.example_selection == "uniform_distribution"
        assert config.similarity_config is None

    @pytest.mark.parametrize(
        "max_examples,eval_size,should_pass",
        [
            (3, 10, True),  # Valid: max_examples < eval_size
            (5, 5, True),  # Valid: max_examples == eval_size
            (10, 5, False),  # Invalid: max_examples > eval_size
            (0, 5, True),  # Valid: zero max_examples
            (1, 0, False),  # Invalid: max_examples > zero eval_size
        ],
    )
    def test_validate_examples_limit(self, max_examples, eval_size, should_pass):
        """Test examples limit validation with different scenarios."""
        config = ICLConfig(max_examples=max_examples)

        if should_pass:
            config.validate_examples_limit(eval_size)  # Should not raise
        else:
            with pytest.raises(ValueError, match="cannot exceed"):
                config.validate_examples_limit(eval_size)


class TestSettings:
    """Test Settings validation."""

    @pytest.mark.parametrize(
        "max_examples,eval_size,should_pass",
        [
            (3, 20, True),  # Default valid case
            (5, 5, True),  # Equal values
            (10, 5, False),  # Invalid case
            (100, 3, False),  # Large difference
        ],
    )
    def test_icl_examples_vs_eval_size(self, base_configs, max_examples, eval_size, should_pass):
        """Test ICL max_examples validation against eval_size."""
        data_split_config = DataSplitConfig(eval_size=eval_size)
        icl_config = ICLConfig(max_examples=max_examples)

        if should_pass:
            Settings(data_split_config=data_split_config, icl_config=icl_config, **base_configs)
        else:
            with pytest.raises(ValidationError, match="cannot exceed"):
                Settings(data_split_config=data_split_config, icl_config=icl_config, **base_configs)

    def test_default_config_values_are_valid(self, base_configs):
        """Test that default configuration values pass validation."""
        settings = Settings(
            data_split_config=DataSplitConfig(),  # eval_size=20
            icl_config=ICLConfig(),  # max_examples=3
            **base_configs,
        )

        assert settings.icl_config.max_examples == 3
        assert settings.data_split_config.eval_size == 20

    def test_settings_initialization_all_params(self):
        """Test Settings initialization with all parameters."""
        settings = Settings(
            nmp_config=NMPConfig(
                datastore_base_url="http://test",
                nemo_base_url="http://test",
                nim_base_url="http://test",
            ),
            nims=[
                NIMConfig(
                    model_name="test/model",
                    context_length=1024,
                    model_type="llm",
                )
            ],
            llm_judge_config=LLMJudgeConfig(
                model_name="test", context_length=1024, deployment_type="local"
            ),
            training_config=TrainingConfig(lora=LoRAConfig()),
            data_split_config=DataSplitConfig(eval_size=50),
            icl_config=ICLConfig(max_examples=5),
            logging_config=LoggingConfig(),
            mlflow_config=MLflowConfig(),
        )

        assert settings.data_split_config.eval_size == 50
        assert settings.icl_config.max_examples == 5


class TestConfigIntegration:
    """Test integration scenarios."""

    def test_semantic_similarity_with_examples_limit(self, base_configs):
        """Test semantic similarity config with examples limit validation."""
        embedding_config = EmbeddingConfig(
            deployment_type="remote",
            model_name="test-embedding",
            url="http://test-url",
            api_key="test-key",
        )
        similarity_config = SimilarityConfig(embedding_nim_config=embedding_config)

        settings = Settings(
            data_split_config=DataSplitConfig(eval_size=10),
            icl_config=ICLConfig(
                max_examples=5,
                example_selection="semantic_similarity",
                similarity_config=similarity_config,
            ),
            **base_configs,
        )

        assert settings.icl_config.example_selection == "semantic_similarity"
        assert settings.icl_config.max_examples == 5
        assert settings.data_split_config.eval_size == 10

    def test_multiple_validation_errors(self, base_configs):
        """Test that multiple validation errors are handled properly."""
        # This should fail both semantic similarity and examples limit validation
        with pytest.raises(ValidationError):
            Settings(
                data_split_config=DataSplitConfig(eval_size=2),
                icl_config=ICLConfig(
                    max_examples=10,  # Too high
                    example_selection="semantic_similarity",
                    similarity_config=None,  # Missing
                ),
                **base_configs,
            )
