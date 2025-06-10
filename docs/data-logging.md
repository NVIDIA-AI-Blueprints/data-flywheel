# Data Logging for AI Apps

Instrumenting your AI application to log interactions is a critical step in implementing the Data Flywheel. This guide explains how to enable data logging for any AI app, providing a general approach and best practices. A working example using an [AI Virtual Assistant (AIVA)](https://github.com/NVIDIA-AI-Blueprints/ai-virtual-assistant) is included for reference.

## General Approach and Requirements

### Supported Logging Backends

- **Elasticsearch** (default, recommended)
- (Extendable to other backends as needed)

### Environment Variables

To enable data logging, set the following environment variables:

```sh
ELASTICSEARCH_URL=http://your-elasticsearch-host:9200
ELASTICSEARCH_INDEX=llm-logs  # Default index name
ELASTICSEARCH_CLIENT_ID=your-app-name  # Client identifier
```

### Data Schema

Log entries should include:

```json
{
  "request": { ... },
  "response": { ... },
  "timestamp": "...",
  "client_id": "...",
  "workload_id": "..."
}
```

## Implementing Data Logging in Any App

### Generic Logging Handler (Example)

```python
class GenericLogHandler(AsyncCallbackHandler):
    def __init__(self, workload_id):
        self.workload_id = workload_id
        self.elasticsearch_url = os.getenv("ELASTICSEARCH_URL", "")
    async def on_chat_model_start(self, serialized, messages, run_id, **kwargs):
        # Capture request data
        self.request_data = {...}
    async def on_llm_end(self, output, run_id, **kwargs):
        # Create log entry and send to backend
        log_entry = {...}
        await self._log_to_backend(log_entry)
```

### Integration Steps

1. Initialize the logging handler with the appropriate `workload_id`.
2. Attach the handler to your large language model (LLM) or agent workflow.
3. Make sure environment variables are set for your logging backend.
4. Log each request/response interaction.

## Example: Instrumenting AIVA

This section provides a practical example of instrumenting an [AI Virtual Assistant (AIVA)](https://github.com/NVIDIA-AI-Blueprints/ai-virtual-assistant) application to log data for the Data Flywheel. It extends the general guidelines presented in the ["Instrumenting an application"](../README.md#2instrumenting-an-application) section of the main README. Instrumenting your application to log LLM interactions is a critical step in implementing the Data Flywheel. This example demonstrates how to integrate Elasticsearch logging into AIVA to capture comprehensive data about LLM interactions.

### Configuration

To enable data logging to Elasticsearch for AIVA, configure the following environment variables:

```sh
ELASTICSEARCH_URL=http://your-elasticsearch-host:9200
ELASTICSEARCH_INDEX=aiva-llm-logs  # Default index name
ELASTICSEARCH_CLIENT_ID=aiva       # Client identifier
```

### Data Schema

The log entries stored in Elasticsearch contain the following structure:

```json
{
  "request": {
    "model": "model_name",
    "messages": [{"role": "user", "content": "..."}],
    "temperature": 0.2,
    "max_tokens": 1024,
    "tools": []
  },
  "response": {
    "id": "run_id",
    "object": "chat.completion",
    "model": "model_name",
    "usage": {"prompt_tokens": 50, "completion_tokens": 120, "total_tokens": 170}
  },
  "timestamp": "2024-05-15T12:34:56.789Z",
  "client_id": "aiva",
  "workload_id": "session_id"
}
```

### Implementation Architecture

The AIVA logging system consists of three main components:

1. **ElasticsearchLogHandler**: A custom LangChain callback handler that captures request and response data.
2. **Integration with AIVA Agent System**: Seamless incorporation into the agent architecture.
3. **Docker Compose/Environment Configuration**: Use environment variables to connect to Elasticsearch.

- **Request data**: Prompts, messages, model parameters
- **Response data**: Completions, tokens, usage statistics
- **Metadata**: Timestamps, client identifiers, workload IDs

#### Code Implementation (AIVA Example)

```python
class ElasticsearchLogHandler(AsyncCallbackHandler):
    def __init__(self, workload_id):
        self.workload_id = workload_id
        self.elasticsearch_url = os.getenv("ELASTICSEARCH_URL", "")
    async def on_chat_model_start(self, serialized, messages, run_id, **kwargs):
        self.request_data = {
            "model": metadata.get("ls_model_name", serialized.get("model")),
            "messages": convert_to_openai_messages(messages[0]),
            "temperature": metadata.get("ls_temperature"),
            "max_tokens": metadata.get("ls_max_tokens"),
            "tools": kwargs.get("invocation_params", {}).get("tools")
        }
    async def on_llm_end(self, output, run_id, **kwargs):
        log_entry = {
            "request": self.request_data,
            "response": {
                "id": str(run_id),
                "object": "chat.completion",
                "model": output.llm_output.pop("model_name"),
                **output.llm_output
            },
            "timestamp": datetime.utcnow().isoformat(),
            "client_id": os.getenv("ELASTICSEARCH_CLIENT_ID", "aiva"),
            "workload_id": self.workload_id
        }
        await self._log_to_elasticsearch(log_entry)
```

#### Integration with AIVA

```python
class Assistant:
    def __init__(self, prompt, tools, workload_id):
        self.prompt = prompt
        self.tools = tools
        self.workload_id = workload_id
    async def __call__(self, state, config):
        llm = get_llm(**config.get('configurable', {}).get("llm_settings"))
        elasticsearch_url = os.getenv("ELASTICSEARCH_URL", "")
        if elasticsearch_url:
            callbacks = await create_langchain_callbacks(workload_id=self.workload_id)
            runnable = runnable.with_config(callbacks=callbacks)
            config = {**config, "callbacks": callbacks}
        result = await runnable.ainvoke(state)
        return {"messages": result}
```

#### Dependencies

- `elasticsearch==8.17.2`

## Best Practices

- Use consistent `workload_id` values for accurate workload identification.
- Make sure you include error handling in logging routines.
- Be mindful of privacy and personally identifiable information (PII)—consider redacting or anonymizing as needed.
- Log only what's necessary for model improvement and debugging.

## Additional Resources

- [Instrumenting an application (README)](../README.md#2instrumenting-an-application)
- [Elasticsearch Python client](https://www.elastic.co/guide/en/elasticsearch/client/python-api/current/index.html)
- [LangChain Callbacks](https://python.langchain.com/docs/modules/callbacks/) 