# Migrating AI Virtual Assistant to NAT (NEMO Agent Toolkit)

## Overview

This guide shows how to migrate the AI Virtual Assistant from the original LangGraph implementation (main branch) to the NAT implementation (nat-dfw-integration branch).

### Why Migrate to NAT?

The primary goal of this migration is to enable **native NVIDIA Data Flywheel Blueprint integration** for automatic observability and model optimization.

## ⭐ Data Flywheel Blueprint Integration - The Key Benefit

One of the main objectives of this migration is to enable native Data Flywheel support. Once your functions are decorated with NAT's `@register_function`, all you need to do is add a few lines to your config file:

```yaml
general:
  telemetry:
    tracing:
      dfw_elasticsearch:
        _type: data_flywheel_elasticsearch
        endpoint: ${DATA_FLYWHEEL_ENDPOINT}
        client_id: ${DATA_FLYWHEEL_CLIENT_ID}
        index: flywheel
```

**That's it!** Your traces will automatically export to Data Flywheel for:
- LLM distillation and optimization
- Performance analysis and monitoring
- Training smaller, more efficient models
- Runtime optimization insights

This is extremely user-friendly - you get Data Flywheel capabilities "for free" with minimal configuration.

📚 **Learn more**: [NAT Data Flywheel Plugin Documentation](https://docs.nvidia.com/nemo/agent-toolkit/latest/workflows/observe/observe-workflow-with-data-flywheel.html)

---

## Migration by Example: `get_purchase_history`

Let's walk through migrating one function from start to finish. We'll use `get_purchase_history` as our example - a function that queries the database for user purchase history.

### Original Implementation (Main Branch)

**File: `src/agent/tools.py`**

```python
import os
from langchain_core.tools import tool
from functools import lru_cache
import psycopg2
import psycopg2.extras
from urllib.parse import urlparse
from src.common.utils import get_config

@tool
@lru_cache
def get_purchase_history(user_id: str) -> str:
    """Retrieves the recent return and order details for a user,
    including order ID, product name, status, relevant dates, quantity, and amount."""

    SQL_QUERY = f"""
    SELECT order_id, product_name, order_date, order_status, quantity, order_amount, return_status,
    return_start_date, return_received_date, return_completed_date, return_reason, notes
    FROM public.customer_data
    WHERE customer_id={user_id}
    ORDER BY order_date DESC
    LIMIT 15;
    """

    app_database_url = get_config().database.url
    parsed_url = urlparse(f"//{app_database_url}", scheme='postgres')
    
    host = parsed_url.hostname
    port = parsed_url.port

    db_params = {
        'dbname': os.getenv("CUSTOMER_DATA_DB", 'customer_data'),
        'user': os.getenv('POSTGRES_USER_READONLY', None),
        'password': os.getenv('POSTGRES_PASSWORD_READONLY', None),
        'host': host,
        'port': port
    }

    with psycopg2.connect(**db_params) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(SQL_QUERY)
            result = cur.fetchall()

    return [dict(row) for row in result]
```

**Usage in graph:**
```python
# In main.py
from tools import get_purchase_history

# Used directly as a tool
order_status_tools = [structured_rag, get_purchase_history, ProductValidation]
```

**Configuration:**
```python
# Hardcoded environment variables
database_url = os.getenv("APP_DATABASE_URL")
db_user = os.getenv("POSTGRES_USER")
```

---

### NAT Implementation (NAT Branch)

#### Step 1: Create the Function File

**File: `src/nat_agent/aiva_agent/src/aiva_agent/functions/get_purchase_history_fn.py`**

```python
import logging
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class GetPurchaseHistoryConfig(FunctionBaseConfig, name="get_purchase_history"):
    """Configuration for get_purchase_history function."""
    database_url: str = Field(..., description="Database connection URL")
    user: str = Field(..., description="Database user")
    password: str = Field(..., description="Database password")
    dbname: str = Field(default="customer_data", description="Database name")


@register_function(config_type=GetPurchaseHistoryConfig)
async def get_purchase_history_fn(
    config: GetPurchaseHistoryConfig, 
    builder: Builder
):
    """Retrieves the recent return and order details for a user."""
    
    import psycopg2
    import psycopg2.extras
    from urllib.parse import urlparse
    
    # Parse connection parameters from config
    parsed_url = urlparse(f"//{config.database_url}", scheme='postgres')
    host = parsed_url.hostname
    port = parsed_url.port
    
    db_params = {
        'dbname': config.dbname,
        'user': config.user,
        'password': config.password,
        'host': host,
        'port': port
    }
    
    async def _response_fn(user_id: str) -> list:
        """Retrieves the recent return and order details for a user,
        including order ID, product name, status, relevant dates, quantity, and amount."""
        
        SQL_QUERY = f"""
        SELECT order_id, product_name, order_date, order_status, quantity, order_amount, return_status,
        return_start_date, return_received_date, return_completed_date, return_reason, notes
        FROM public.customer_data
        WHERE customer_id={user_id}
        ORDER BY order_date DESC
        LIMIT 15;
        """
        
        with psycopg2.connect(**db_params) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(SQL_QUERY)
                result = cur.fetchall()
        
        def _serialize_for_json(row_dict):
            """Convert date and decimal objects to JSON-serializable types."""
            from decimal import Decimal
            for key, value in row_dict.items():
                if hasattr(value, 'strftime'):  # date/datetime objects
                    row_dict[key] = value.strftime('%Y-%m-%d')
                elif isinstance(value, Decimal):  # decimal objects
                    row_dict[key] = float(value)
            return row_dict
        
        return [_serialize_for_json(dict(row)) for row in result]
    
    yield FunctionInfo.create(
        single_fn=_response_fn,
        description="Retrieves the recent return and order details for a user, including order ID, product name, status, relevant dates, quantity, and amount."
    )
```

#### Step 2: Register the Function

**File: `src/nat_agent/aiva_agent/src/aiva_agent/register.py`**

```python
# Import the function to register it
from aiva_agent.functions import get_purchase_history_fn

# Import all other functions...
```

#### Step 3: Add Configuration

**File: `src/nat_agent/aiva_agent/configs/config.yml`**

```yaml
general:
  use_uvloop: true
  
  # 🎯 DATA FLYWHEEL INTEGRATION - Just add these lines!
  telemetry:
    tracing:
      dfw_elasticsearch:
        _type: data_flywheel_elasticsearch
        endpoint: ${DATA_FLYWHEEL_ENDPOINT}
        client_id: ${DATA_FLYWHEEL_CLIENT_ID}
        index: flywheel

functions:
  get_purchase_history:
    _type: get_purchase_history
    database_url: ${APP_DATABASE_URL}
    user: ${POSTGRES_USER}
    password: ${POSTGRES_PASSWORD}
    dbname: ${CUSTOMER_DATA_DB}
```

#### Step 4: Use in Workflow

**File: `src/nat_agent/aiva_agent/src/aiva_agent/functions/aiva_agent_fn.py`**

```python
@register_function(config_type=AivaAgentFunctionConfig)
async def aiva_agent_function(config: AivaAgentFunctionConfig, builder: Builder):

    # Get as a tool for LangChain
    return_processing_tools = await builder.get_tools(
        ["get_purchase_history", "return_window_validation"],
        wrapper_type=LLMFrameworkEnum.LANGCHAIN
    )
    
    # Use in your graph
    graph_builder.add_node(
        "return_processing_safe_tools",
        create_tool_node_with_fallback(return_processing_tools)
    )
```

---

## Key Differences Explained

### 1. Function Structure

| Original | NAT |
|----------|-----|
| `@tool` decorator | `@register_function` decorator |
| Direct implementation | Config class + generator pattern |
| Environment variables in code | Configuration via YAML |
| Global imports | Dependency injection via builder |

### 2. Configuration Class

The config class defines all parameters needed by the function:

```python
class GetPurchaseHistoryConfig(FunctionBaseConfig, name="get_purchase_history"):
    database_url: str = Field(..., description="Database connection URL")
    user: str = Field(..., description="Database user")
    password: str = Field(..., description="Database password")
    dbname: str = Field(default="customer_data", description="Database name")
```

- **`name="get_purchase_history"`**: Matches the identifier in config.yml
- **Field descriptions**: Used for documentation and validation
- **Defaults**: Can provide fallback values
- **Type hints**: Enable automatic validation

### 3. Generator Pattern

```python
@register_function(config_type=GetPurchaseHistoryConfig)
async def get_purchase_history_fn(config, builder):
    # Setup code (runs once during initialization)
    
    async def _response_fn(user_id: str) -> list:
        # Business logic (runs on each invocation)
        pass
    
    yield FunctionInfo.create(single_fn=_response_fn)
```

- **Outer function**: Runs once during registration, sets up dependencies
- **Inner `_response_fn`**: The actual function that gets called
- **`yield FunctionInfo`**: Returns the function info to NAT, allowing across frameworks

### 4. Builder Pattern (Dependency Injection)

Instead of direct imports, use the builder to get dependencies:

```python
# Get a function
helper_fn = await builder.get_function("helper_function")
result = await helper_fn.ainvoke(input)

# Get an LLM
llm = await builder.get_llm("chat_llm", wrapper_type=LLMFrameworkEnum.LANGCHAIN)
response = await llm.ainvoke(prompt)

# Get tools
tools = await builder.get_tools(["tool1", "tool2"], wrapper_type=LLMFrameworkEnum.LANGCHAIN)
```

Benefits:
- ✅ Easy to test (mock the builder)
- ✅ Clear dependencies
- ✅ Loose coupling

---

## Complete Migration Checklist

### For Each Function:

- [ ] **Create function file** in `functions/` directory
  - Naming: `{function_name}_fn.py`

- [ ] **Define config class**
  ```python
  class MyFunctionConfig(FunctionBaseConfig, name="my_function"):
      param1: str = Field(..., description="...")
  ```

- [ ] **Add `@register_function` decorator**
  ```python
  @register_function(config_type=MyFunctionConfig)
  async def my_function_fn(config, builder):
  ```

- [ ] **Implement as async generator**
  ```python
  async def _response_fn(...):
      # Your logic
      pass
  
  yield FunctionInfo.create(single_fn=_response_fn)
  ```

- [ ] **Import in `register.py`**
  ```python
  from aiva_agent.functions import my_function_fn
  ```

- [ ] **Add to `config.yml`**
  ```yaml
  functions:
    my_function:
      _type: my_function
      param1: ${ENV_VAR}
  ```

- [ ] **Enable Data Flywheel** (one-time setup)
  ```yaml
  general:
    telemetry:
      tracing:
        dfw_elasticsearch:
          _type: data_flywheel_elasticsearch
          endpoint: ${DATA_FLYWHEEL_ENDPOINT}
          client_id: ${DATA_FLYWHEEL_CLIENT_ID}
          index: flywheel
  ```

---

## Advanced: Function with LLM Dependency

Let's look at a more complex example that uses an LLM.

### Original Implementation

```python
# In main.py or tools.py
from langchain_core.prompts import ChatPromptTemplate
from src.common.utils import get_prompts

prompts = get_prompts()

def handle_other_talk(state: State, config: RunnableConfig):
    """Handles greetings and other queries."""
    
    # LLM is somehow obtained (global, passed in, etc.)
    chat_llm = get_llm()
    
    base_prompt = prompts.get("other_talk_template")
    prompt = ChatPromptTemplate.from_messages([
        ("system", base_prompt),
        ("placeholder", "{messages}"),
    ])
    
    chain = prompt | chat_llm
    response = await chain.ainvoke(state, config)
    
    return {"messages": [response]}
```

### NAT Implementation

**File: `functions/handle_other_talk_fn.py`**

```python
import logging
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.builder.framework_enum import LLMFrameworkEnum

logger = logging.getLogger(__name__)


class HandleOtherTalkConfig(FunctionBaseConfig, name="handle_other_talk"):
    """Configuration for handle_other_talk function."""
    llm_name: str = Field(..., description="Name of the LLM to use")
    llm_tags: list[str] = Field(
        default=["should_stream"],
        description="Tags for the LLM"
    )
    prompt_config_file: str = Field(
        default="prompt.yaml",
        description="Path to prompt configuration file"
    )


@register_function(
    config_type=HandleOtherTalkConfig,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN]
)
async def handle_other_talk_fn(
    config: HandleOtherTalkConfig,
    builder: Builder
):
    """Handle other talk function implementation."""
    
    from langchain_core.prompts.chat import ChatPromptTemplate
    from aiva_agent.utils import get_prompts
    
    # Get the LLM from builder (dependency injection)
    llm = await builder.get_llm(
        config.llm_name,
        wrapper_type=LLMFrameworkEnum.LANGCHAIN
    )
    llm.disable_streaming = True
    llm = llm.with_config(tags=config.llm_tags)
    
    # Get prompts
    prompts = get_prompts(prompt_config_file=config.prompt_config_file)
    base_prompt = prompts.get("other_talk_template", "")
    prompt = ChatPromptTemplate.from_messages([
        ("system", base_prompt),
        ("placeholder", "{messages}"),
    ])
    
    # Create the chain
    chain = prompt | llm
    
    async def _response_fn(state: dict, config: dict) -> dict:
        """Handle other talk response."""
        response = await chain.ainvoke(state, config)
        return {"messages": [response]}
    
    yield FunctionInfo.create(
        single_fn=_response_fn,
        description="Handles greetings and other queries."
    )
```

**Configuration in `config.yml`:**

```yaml
llms:
  chat_llm:
    _type: nim
    model_name: ${APP_CHAT_LLM_MODELNAME}
    base_url: ${APP_CHAT_LLM_SERVERURL}
    temperature: 0.2
    top_p: 0.7
    max_tokens: 1024
    api_key: ${NVIDIA_API_KEY}

functions:
  handle_other_talk:
    _type: handle_other_talk
    llm_name: chat_llm
    prompt_config_file: ${APP_PROMPT_CONFIG_FILE}
```

**Key points:**
1. LLM configuration is in YAML, not code
2. Function gets LLM by name via builder
3. Easy to swap LLMs by changing config
4. Framework wrapper enables LangChain compatibility

---

## Configuration File Structure

### Complete `config.yml` Structure

```yaml
general:
  use_uvloop: true
  
  # 🎯 DATA FLYWHEEL - Automatic observability!
  telemetry:
    tracing:
      dfw_elasticsearch:
        _type: data_flywheel_elasticsearch
        endpoint: ${DATA_FLYWHEEL_ENDPOINT}
        client_id: ${DATA_FLYWHEEL_CLIENT_ID}
        index: flywheel
  
  # Optional: Define FastAPI endpoints
  front_end:
    _type: fastapi
    endpoints:
      - path: /health
        method: GET
        description: Health check endpoint
        function_name: health_check
      - path: /create_session
        method: GET
        description: Generate a new session id
        function_name: create_session

# Define your LLMs once, reference everywhere
llms:
  tool_call_llm:
    _type: nim
    model_name: ${APP_TOOLCALL_LLM_MODELNAME}
    base_url: ${APP_TOOLCALL_LLM_SERVERURL}
    temperature: 0.2
    top_p: 0.7
    max_tokens: 1024
    api_key: ${NVIDIA_API_KEY}
    
  chat_llm:
    _type: nim
    model_name: ${APP_CHAT_LLM_MODELNAME}
    base_url: ${APP_CHAT_LLM_SERVERURL}
    temperature: 0.2
    api_key: ${NVIDIA_API_KEY}

# Define all your functions
functions:
  get_purchase_history:
    _type: get_purchase_history
    database_url: ${APP_DATABASE_URL}
    user: ${POSTGRES_USER}
    password: ${POSTGRES_PASSWORD}
    dbname: ${CUSTOMER_DATA_DB}
  
  handle_other_talk:
    _type: handle_other_talk
    llm_name: chat_llm
    prompt_config_file: ${APP_PROMPT_CONFIG_FILE}
  
  # ... more functions

# Define your main workflow
workflow:
  _type: aiva_agent
  tool_call_llm_name: tool_call_llm
  chat_llm_name: chat_llm
  checkpointer_type: ${APP_CHECKPOINTER_NAME}
  cache_type: ${APP_CACHE_NAME}
```

### Environment Variable Interpolation

NAT automatically replaces `${VAR_NAME}` with environment variables:

```yaml
functions:
  my_function:
    api_key: ${OPENAI_API_KEY}      # From environment
    database_url: ${DATABASE_URL}   # From environment
    timeout: 30                     # Literal value
```

---

## Data Flywheel: Advanced Usage

### Custom Workload Scoping

For fine-grained optimization, create custom workload scopes:

```python
from nat.profiler.decorators.function_tracking import track_unregistered_function

@track_unregistered_function(
    name="document_summarizer",
    metadata={"task_type": "summarization"}
)
async def summarize_document(document: str) -> str:
    """This will create a separate workload_id for Data Flywheel."""
    return await llm_client.complete(f"Summarize: {document}")

@track_unregistered_function(name="question_answerer")
async def answer_question(context: str, question: str) -> str:
    """This will create another workload_id."""
    return await llm_client.complete(f"Context: {context}\nQuestion: {question}")
```

This allows Data Flywheel to:
- Train separate optimized models for each workload
- Target specific tasks for distillation
- Track performance per workload type

### Monitoring Trace Export

As your workflow runs, check logs for Data Flywheel exports:

```
INFO - Exporting batch of 10 traces to Data Flywheel
INFO - Successfully exported traces to Elasticsearch index: flywheel
```

### Configuration Parameters

| Parameter | Description | Required | Example |
|-----------|-------------|----------|---------|
| `client_id` | Identifier for your application | Yes | `"aiva-agent"` |
| `index` | Elasticsearch index name | Yes | `"flywheel"` |
| `endpoint` | Elasticsearch endpoint URL | Yes | `"https://es.example.com:9200"` |
| `username` | Elasticsearch username | No | `"elastic"` |
| `password` | Elasticsearch password | No | `"elastic"` |
| `batch_size` | Batch size before exporting | No | `10` |

📚 **Full documentation**: [Data Flywheel Integration Guide](https://docs.nvidia.com/nemo/agent-toolkit/latest/workflows/observe/observe-workflow-with-data-flywheel.html)

---

## Summary

### Migration Steps

1. ✅ Create function file with config class
2. ✅ Add `@register_function` decorator
3. ✅ Implement async generator and pass into `FunctionInfo` constructor
4. ✅ Import in `register.py`
5. ✅ Add configuration to `config.yml`
6. ✅ **Enable Data Flywheel** (just 5 lines in config!)
7. ✅ Test with mocks
8. ✅ Run and verify

### Key Benefits Achieved

1. **🎯 Automatic Data Flywheel Export**
   - Minimal configuration (5 lines!)
   - Automatic trace collection
   - Model optimization ready

2. **📊 Built-in Observability**
   - Function tracking
   - Performance metrics
   - Error tracking

3. **⚙️ Better Configuration**
   - YAML-based, not scattered in code
   - Environment variable interpolation
   - Easy to change without code changes

4. **🧪 Easier Testing**
   - Dependency injection
   - Easy mocking
   - Clear interfaces

5. **📁 Better Organization**
   - One function per file
   - Clear structure
   - Easy to navigate

### Resources

- **NAT Documentation**: https://docs.nvidia.com/nemo/agent-toolkit/latest/
- **NAT Data Flywheel Documentation**: https://docs.nvidia.com/nemo/agent-toolkit/latest/workflows/observe/observe-workflow-with-data-flywheel.html

