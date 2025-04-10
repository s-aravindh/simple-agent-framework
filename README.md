# Simple Agent Framework

A lightweight agent framework combining the best aspects of Pydantic-AI and OpenAI Agents SDK.

> **⚠️ WARNING:** This is an educational and learning framework inspired by OpenAI Agents SDK and Pydantic-AI. It is designed for learning and experimentation purposes only and is **not recommended for production usage**. The implementation focuses on clarity and simplicity rather than performance, security, or scalability needed in production environments.

## Features

- Simple, minimal API inspired by OpenAI Agents SDK
- Strong typing with Pydantic for I/O validation
- Support for both OpenAI and AWS Bedrock models
- Easy tool definition and execution
- Flexible agent configurations

## Installation

```bash
pip install simple-agent
```

## Quick Start

```python
from simple_agent import Agent
from simple_agent.models import OpenAIModel
from simple_agent.tools import function_tool

# Define a simple tool
@function_tool
def get_weather(location: str) -> str:
    """Get the weather for a location."""
    # Implement weather lookup logic here
    return f"It's sunny in {location}."

# Create an agent with OpenAI model
agent = Agent(
    name="WeatherAssistant",
    instructions="Help users with weather questions.",
    model=OpenAIModel(model="gpt-4o"),
    tools=[get_weather]
)

# Run the agent
result = agent.run("What's the weather like in San Francisco?")
print(result)
```

## Bedrock Support

```python
from simple_agent import Agent
from simple_agent.models import BedrockModel

agent = Agent(
    name="BedrockAssistant",
    instructions="Help users with their questions.",
    model=BedrockModel(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        region_name="us-west-2"
    )
)

result = agent.run("Tell me about machine learning.")
print(result)
```

## License

MIT