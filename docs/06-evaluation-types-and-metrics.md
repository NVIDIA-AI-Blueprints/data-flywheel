# Evaluation Types and Metrics

The Flywheel supports the following evaluation types and metrics out-of-the-box.

## Evaluation Types

### Evaluation Types Matrix

| Evaluation Type      | Dataset                                 | Model                        | Context                | Purpose                                                                                 |
|---------------------|-----------------------------------------|------------------------------|------------------------|-----------------------------------------------------------------------------------------|
| Base Evaluation     | Base (held-out, production-like)         | Out-of-the-box (no fine-tuning) | No ICL examples        | Establishes a baseline for model performance on real-world data, before any adaptation.  |
| ICL Evaluation      | ICL-augmented (with ICL examples)        | Out-of-the-box (no fine-tuning) | Few-shot (ICL) examples | Measures how well the model adapts when given a few relevant examples (prompt engineering). |
| Customized Evaluation | Base (held-out, production-like)         | Fine-tuned/customized         | No ICL examples        | Quantifies improvement (or regression) from fine-tuning a model on your specific data.   |

### Base Evaluation (`base-eval`)

Base evaluation tests a model on a standard, held-out dataset sampled from production logs, **without** any in-context learning (ICL) examples or fine-tuning. This is the default evaluation of a model's out-of-the-box performance.

- **dataset:** Base (held-out, production-like)
- **model:** Out-of-the-box (no fine-tuning)
- **context:** No ICL examples
- **purpose:** Establishes a baseline for model performance on real-world data, before any adaptation or context is provided.

**Example:**

```json
{
  "messages": [
    {"role": "user", "content": "How do I reset my password?"},
    {"role": "assistant", "content": "Click 'Forgot password' on the login page."}
  ]
}
```

### ICL Evaluation (`icl-eval`)

ICL (In-Context Learning) evaluation tests a model's ability to leverage a few-shot context—that is, it prepends a small number of example Q&A pairs to each prompt, simulating a "few-shot" learning scenario.

- **dataset:** ICL-augmented (each test example is preceded by a configurable number of ICL examples)
- **model:** Out-of-the-box (no fine-tuning)
- **context:** Few-shot (ICL) examples, controlled by `icl_config` in `config.yaml`
- **purpose:** Measures how well the model can generalize or adapt when given a few relevant examples, simulating real-world prompt engineering or agentic use cases.

**Example:**

```json
{
  "messages": [
    {"role": "user", "content": "How do I reset my password?"},
    {"role": "assistant", "content": "Click 'Forgot password' on the login page."},
    {"role": "user", "content": "How do I change my email address?"},
    {"role": "assistant", "content": "Go to account settings and update your email."},
    {"role": "user", "content": "How do I delete my account?"}
  ]
}
```

> **Note** 
>
> The last user message is the test query; the previous pairs are ICL examples.

### Customized Evaluation (`customized-eval`)

Customized evaluation tests a *fine-tuned* or *customized* version of the model on the same base dataset. This measures the effect of supervised fine-tuning (for example, LoRA, SFT) on model performance.

- **dataset:** Base (held-out, production-like)
- **model:** Fine-tuned/customized (for example, via LoRA, SFT)
- **context:** No ICL examples
- **purpose:** Quantifies the improvement (or regression) from fine-tuning a model on your specific data, and enables direct comparison to the base model.

**Example:**
```json
{
  "messages": [
    {"role": "user", "content": "How do I reset my password?"},
    {"role": "assistant", "content": "Click 'Forgot password' on the login page."}
  ]
  // Model being evaluated is a fine-tuned/customized version
}
```

### Tool-Calling Evaluation Example

The Flywheel Blueprint is designed for agentic workloads, especially those using tool-calling. Tool-calling metrics are central to evaluating how well models can interact with external functions or APIs.

**Tool-calling Example:**
```json
{
  "request": {
    "model": "meta/llama-3.1-70b-instruct",
    "messages": [
      {"role": "system", "content": "You are a chatbot..."},
      {"role": "user", "content": "Has my bill been processed?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "structured_rag",
          "parameters": {
            "properties": {
              "query": {"type": "string"},
              "user_id": {"type": "string"}
            },
            "required": ["query", "user_id"]
          }
        }
      }
    ]
  },
  "response": {
    "choices": [{
      "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
          "type": "function",
          "function": {
            "name": "structured_rag",
            "arguments": {"query": "account bill processed", "user_id": "4165"}
          }
        }]
      }
    }]
  }
}
```
> **Note**
>
> This structure is used for both evaluation and fine-tuning. The model is expected to select the correct function and provide the correct arguments in its tool call output.

## Metrics

Metrics appear in the `scores` field of evaluation results.

| Metric Name | Type/Range | Computed By| Description |
|-------------------------------|--------------------|----------------------|----------------------------------------------------------------------------------------------|
| similarity | Integer (1–10) | LLM judge | For non tool-calling datasets; LLM-judged similarity between candidate and reference answers. Higher is better. |
| function_name / function_name_accuracy | Float (0–1) | Programmatic | Accuracy of function name prediction in tool-calling tasks. |
| function_name_and_args_accuracy| Float (0–1) | Programmatic | Accuracy of both function name and arguments in tool-calling tasks. |
| tool_calling_correctness | Integer (0 or 1) | LLM judge | LLM-judged correctness of tool call output (1=correct, 0=incorrect). |

### Custom Metrics and Evaluations

The Flywheel evaluation framework is designed to be extensible. Advanced users can define custom evaluation metrics or scoring logic by:

- **Custom LLM Judge Prompts:**  
  Supply your own prompts and criteria for LLM-based judging, enabling new types of qualitative or task-specific metrics.
- **Programmatic Metrics:**  
  Add new programmatic checks (e.g., for specific function arguments, output structure, or other behaviors) by extending the evaluation configuration and pipeline.
- **Flexible Configuration:**  
  The evaluation configuration (see `src/lib/nemo/evaluator.py`) allows you to specify additional metrics, scoring rules, or even new evaluation tasks.

> **Note:**
> If you need to track a new metric, you can add it to your evaluation config and update the result parsing logic. The results will include your custom metric in the `scores` dictionary, alongside the built-in metrics.

## Evaluation Results Format

Evaluation results are returned in a consistent structure, regardless of evaluation type. Each result includes metadata (such as evaluation type, timestamps, and progress) and a `scores` dictionary containing the relevant metrics for that evaluation.

### General Structure

```json
{
  "eval_type": "base-eval",
  "scores": { /* metric values here */ },
  "started_at": "...",
  "finished_at": "...",
  "runtime_seconds": 123.4,
  "progress": 100.0,
  "nmp_uri": "...",
  "error": null
}
```

The **keys in the `scores` dictionary vary** depending on the evaluation and workload type:

| Eval Type         | Workload Type      | scores keys (examples)                                                |
|-------------------|-------------------|-----------------------------------------------------------------------|
| base-eval         | Q&A               | similarity                                                           |
| icl-eval          | Q&A               | similarity                                                           |
| customized-eval   | Q&A               | similarity                                                           |
| base-eval         | Tool-calling      | function_name, function_name_and_args_accuracy, tool_calling_correctness |
| icl-eval          | Tool-calling      | function_name, function_name_and_args_accuracy, tool_calling_correctness |
| customized-eval   | Tool-calling      | function_name, function_name_and_args_accuracy, tool_calling_correctness |


### Examples 


**Q&A (no tool-calling)**

```json
{
  "eval_type": "base-eval",
  "scores": {
    "similarity": 9
  },
  "started_at": "...",
  "finished_at": "...",
  "runtime_seconds": 123.4,
  "progress": 100.0,
  "nmp_uri": "...",
  "error": null
}
```

**Tool-calling**

```json
{
  "eval_type": "base-eval",
  "scores": {
    "function_name": 1.0,
    "function_name_and_args_accuracy": 0.95,
    "tool_calling_correctness": 1
  },
  "started_at": "...",
  "finished_at": "...",
  "runtime_seconds": 123.4,
  "progress": 100.0,
  "nmp_uri": "...",
  "error": null
}
```