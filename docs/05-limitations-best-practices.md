# Limitations & Best Practices

## Operational Scope

The Data Flywheel Foundational Blueprint provides flywheel service that evaluates candidate models on defined workloads using automated metrics and benchmarks. Models that demonstrate strong performance according to these criteria are surfaced in the evaluation results as promising candidates for further review.

### Example Evaluation Results

The following simplified example results surface the `llama-3.2-1b-instruct-ft-1` model as a promising candidate for your use case.

```json
[
  {
    "model_name": "llama-3.2-1b-instruct-ft-1",
    "evaluations": [
      {
        "eval_type": "accuracy",
        "scores": { "score": 0.92 },
        "progress": 100.0
      }
    ]
  },
  {
    "model_name": "llama-3.2-1b-instruct-ft-2",
    "evaluations": [
      {
        "eval_type": "accuracy",
        "scores": { "score": 0.78 },
        "progress": 100.0
      }
    ]
  }
]
```

This flywheel doesn't automatically promote or deploy any model. You must also validate surfaced models using your own acceptance tests and guardrails before considering them for production.

## Evaluation Types and Metrics

For a detailed explanation of evaluation types and metrics (such as base-eval, icl-eval, similarity, and tool-calling metrics), see [Evaluation Types and Metrics](./06-evaluation-metrics.md).

## Common Pitfalls

### Assuming User Feedback Is Needed

This version of the flywheel exclusively relies on production logs and automated teacher/judge models for evaluation. There is no ingestion or use of explicit user feedback (positive or negative) in any part of the evaluation, training, or promotion process. Attempts to influence model selection or improvement via user thumbs-up/down or similar feedback mechanisms will have no effect.

### Over-trusting a High Accuracy %

A high accuracy score (e.g., 95%) indicates strong overall alignment with a reference model, but the remaining 5% of cases may include critical or policy-violating errors. The system's automated metrics (such as similarity or function-calling accuracy) do not guarantee safety or correctness in all scenarios. Always review divergent cases and consider additional guardrails or human review for high-risk workloads.

### Ignoring Test-Time Compute

Test-time compute is not just a function of model size; smaller models may generate longer outputs or require more tokens to achieve similar results, potentially increasing latency and cost. The blueprint does not automatically optimize for or report on total compute cost. When comparing models, always consider both latency and total tokens generated per request, and validate under realistic production loads.

### Data Leakage

This flywheel includes deduplication and validation steps to reduce the risk of data leakage between evaluation, training, and reference datasets. However, it is your responsibility to ensure that no overlap exists, especially if datasets are updated or re-used. Overlapping data can lead to inflated evaluation scores and misleading results.

### Un-tagged Workloads

Workload IDs are essential for correct data partitioning, evaluation, and reporting. The system expects all records to be tagged with a unique workload ID; missing or inconsistent IDs will result in improper dataset splits and unreliable comparisons. Always ensure that your data ingestion and logging pipelines assign and preserve workload IDs.

### Misinterpreting the `arguments` Field in AIVA Datasets

Datasets produced by AIVA (e.g., `aiva-final.jsonl`) store the `arguments` field as a parsed JSON object rather than the JSON-encoded string returned by the raw OpenAI Chat API. This is intentionally done because the instrumentation layer captured tool-call inputs after they were parsed by the application, and the NeMo customizer expects `arguments` to be an object. If your downstream tools require the original OpenAI representation, stringify this field before use (for example, `record["arguments"] = json.dumps(record["arguments"])`).

## Recommended Verification Steps Before Promotion

| Step | Purpose |
|------|---------|
| Human spot-check of divergent answers | Detect hidden flaws or policy violations |
| Run domain-specific eval suite | Confirm task-level metrics (BLEU, ROUGE, code tests, etc.) |
| Load test under production traffic | Ensure latency and capacity targets |
| Security & compliance review | Verify no new data handling risks |

## Operational Tips

### Start with a Small Subset of Traffic to Shorten Iteration Cycles

Begin by routing only a small portion of your production or evaluation traffic to new models or configurations. This approach allows you to:

- Quickly validate changes and catch issues early, reducing risk.
- Use the `eval_size` and `val_ratio` parameters in your configuration to control the number of records used for evaluation and validation.
- Iterate faster by focusing on a representative sample before scaling up to the full workload.
- Roll back or adjust based on early feedback without impacting the majority of users or data.

### Schedule Jobs During Off-Peak Hours if Using Shared Infrastructure

If your environment shares compute resources with other teams or workloads, consider running intensive jobs (such as training, evaluation, or large-scale data processing) during off-peak hours. This can:

- Reduce contention for resources, leading to faster and more reliable job completion.
- Minimize the impact on other critical workloads.
- Be automated using workflow schedulers, cron jobs, or cloud-native orchestration tools.
- Leverage the system's support for asynchronous job execution (e.g., Celery tasks) to queue jobs for later execution.

### Keep Historical Job Results; Improvements Aren't Always Monotonic

Always retain the results of previous jobs, including evaluation metrics and model performance data. This practice is important because:

- Model improvements may not be linear; sometimes, changes can degrade performance on certain workloads or metrics.
- Historical results allow you to compare new outcomes against established baselines and detect regressions.
- The system tracks job runs and results in persistent storage (e.g., MongoDB collections like `flywheel_runs` and `llm_judge_runs`).
- Keeping a record of past results supports root cause analysis and informed rollback decisions if needed.

### Tune `data_split_config` and `icl_config` as You Learn Workload Characteristics

This flywheel's blueprint provides configuration options to control how data is partitioned and how in-context learning (ICL) is performed:

- `data_split_config` controls the partitioning of data into evaluation, training, and validation sets. Adjust `eval_size`, `val_ratio`, and `min_total_records` to match your workload's size and diversity.
- `icl_config` manages ICL parameters such as `max_context_length`, `reserved_tokens`, and the number of ICL examples. Tune these to optimize for your model's capabilities and the complexity of your tasks.
- Monitor the impact of these settings on evaluation outcomes and iterate as you gather more data about your workloads.
- Refer to the configuration files and code (`src/config.py`, `src/tasks/tasks.py`, and `src/lib/flywheel/util.py`) for details on how these parameters are used in practice.

