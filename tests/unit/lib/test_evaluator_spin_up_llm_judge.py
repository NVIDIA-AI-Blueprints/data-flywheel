# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``Evaluator.spin_up_llm_judge``.

These tests ensure the Evaluator correctly orchestrates deployment of a *local*
LLM judge through the ``DMSClient``:

1. When the model is **not yet deployed** the Evaluator should invoke
   ``deploy_model``.
2. When the model is **already deployed** ``deploy_model`` must *not* be
   called.
3. If ``deploy_model`` raises an exception the Evaluator should propagate it
   unchanged so that the hosting process can fail fast.
"""

from unittest.mock import MagicMock

import pytest

from src.lib.nemo.evaluator import Evaluator

# ---------------------------------------------------------------------------
# Helper patch builders
# ---------------------------------------------------------------------------


def _patch_llm_judge_config(monkeypatch):
    """Patch ``settings.llm_judge_config`` with a minimal stub that provides the
    ``get_local_nim_config`` helper used by ``spin_up_llm_judge``.
    """

    dummy_nim_cfg = MagicMock()
    dummy_nim_cfg.model_name = "stub-model"

    dummy_judge_cfg = MagicMock()
    dummy_judge_cfg.get_local_nim_config.return_value = dummy_nim_cfg
    dummy_judge_cfg.is_remote.return_value = False

    monkeypatch.setattr("src.config.settings.llm_judge_config", dummy_judge_cfg)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_spin_up_llm_judge_deploys_when_not_deployed(monkeypatch):
    """If the judge is *not* deployed, ``deploy_model`` should be triggered."""

    _patch_llm_judge_config(monkeypatch)

    # Build a dummy DMS client that reports *not* deployed
    dummy_client = MagicMock()
    dummy_client.is_deployed.return_value = False
    dummy_client.deploy_model = MagicMock()

    # Patch the DMSClient constructor inside the evaluator so every instantiation
    # returns our dummy instance
    monkeypatch.setattr("src.lib.nemo.evaluator.DMSClient", lambda *_, **__: dummy_client)

    # Act
    result = Evaluator().spin_up_llm_judge()

    # Assert
    dummy_client.is_deployed.assert_called_once()
    dummy_client.deploy_model.assert_called_once()
    assert result is True


def test_spin_up_llm_judge_skips_when_already_deployed(monkeypatch):
    """If the judge is already deployed the Evaluator must *not* redeploy."""

    _patch_llm_judge_config(monkeypatch)

    dummy_client = MagicMock()
    dummy_client.is_deployed.return_value = True
    dummy_client.deploy_model = MagicMock()

    monkeypatch.setattr("src.lib.nemo.evaluator.DMSClient", lambda *_, **__: dummy_client)

    result = Evaluator().spin_up_llm_judge()

    dummy_client.is_deployed.assert_called_once()
    dummy_client.deploy_model.assert_not_called()
    assert result is True


def test_spin_up_llm_judge_propagates_deployment_errors(monkeypatch):
    """Any exception raised by ``deploy_model`` must bubble up unchanged."""

    _patch_llm_judge_config(monkeypatch)

    dummy_client = MagicMock()
    dummy_client.is_deployed.return_value = False
    dummy_client.deploy_model.side_effect = RuntimeError("boom")

    monkeypatch.setattr("src.lib.nemo.evaluator.DMSClient", lambda *_, **__: dummy_client)

    with pytest.raises(RuntimeError, match="boom"):
        Evaluator().spin_up_llm_judge()

    dummy_client.is_deployed.assert_called_once()
    dummy_client.deploy_model.assert_called_once()
