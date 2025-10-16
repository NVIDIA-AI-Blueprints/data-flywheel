#!/bin/bash

helm upgrade --install data-flywheel . \
  --set secrets.ngcApiKey=$NGC_API_KEY \
  --set secrets.nvidiaApiKey=$NVIDIA_API_KEY \
  --set secrets.hfToken=$HF_TOKEN \
  --set secrets.llmJudgeApiKey=$LLM_JUDGE_API_KEY \
  --set secrets.embApiKey=$EMB_API_KEY \
  --namespace nv-nvidia-blueprint-data-flywheel \
  --create-namespace \
  --timeout 20m \
  --debug

