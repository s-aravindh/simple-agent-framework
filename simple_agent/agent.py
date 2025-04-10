import asyncio
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field

from .models.base import ModelBase, ModelResponse
from .models.openai import OpenAIModel
from .stream import ContentChunkEvent, DoneEvent, ErrorEvent, StreamEvent, ToolCallEvent, ToolResultEvent
from .tools.base import Tool


@dataclass
class AgentConfig:
    """Configuration options for an agent."""
    
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    max_iterations: int = 10


class Agent:
    """A simple agent that can use tools and generate responses.
    
    Combines the best aspects of pydantic-ai and OpenAI Agents SDK for a clean,
    simple interface with strong typing.
    """
    
    def __init__(
        self,
        name: str,
        instructions: str,
        model: Optional[ModelBase] = None,
        tools: List[Union[Tool, Callable]] = None,
        config: Optional[AgentConfig] = None,
        output_type: Optional[Type[BaseModel]] = None,
    ):
        """Initialize a new agent.
        
        Args:
            name: Name of the agent.
            instructions: System instructions for the agent.
            model: Model provider to use. Defaults to OpenAI GPT-4o if not provided.
            tools: List of tools the agent can use.
            config: Configuration options.
            output_type: Optional Pydantic model for structured output.
        """
        self.name = name
        self.instructions = instructions
        self.model = model or OpenAIModel()
        self.tools = []
        self.config = config or AgentConfig()
        self.output_type = output_type
        
        # Process tools
        if tools:
            for tool in tools:
                if callable(tool) and hasattr(tool, "_tool"):
                    # This is a function decorated with @function_tool
                    self.tools.append(tool._tool)
                elif isinstance(tool, Tool):
                    self.tools.append(tool)
                else:
                    raise ValueError(f"Tool must be a Tool instance or decorated function, got {type(tool)}")
    
    async def arun(self, input_text: str) -> Any:
        """Run the agent asynchronously.
        
        Args:
            input_text: The input text from the user.
            
        Returns:
            The agent's response, which may be a string or a structured object
            if output_type was specified.
        """
        # Initialize conversation history
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": input_text}
        ]
        
        # Convert tools to the format expected by the model
        tool_dicts = [tool.to_dict() for tool in self.tools]
        
        # Run the agent loop
        iterations = 0
        
        while iterations < self.config.max_iterations:
            iterations += 1
            
            # Generate a response from the model
            response = await self.model.generate(
                messages=messages,
                tools=tool_dicts if self.tools else None,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            
            # If there are tool calls, execute them
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    # Find the matching tool
                    tool_name = tool_call["function"]["name"]
                    tool = next((t for t in self.tools if t.name == tool_name), None)
                    
                    if not tool:
                        error_msg = f"Tool '{tool_name}' not found"
                        tool_response = {"error": error_msg}
                    else:
                        try:
                            # Execute the tool with the provided arguments
                            args = tool_call["function"]["arguments"]
                            tool_result = await tool.execute(**args)
                            tool_response = tool_result
                        except Exception as e:
                            tool_response = {"error": str(e)}
                    
                    # Add the tool call and result to the conversation
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call]
                    })
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": str(tool_response),
                    })
            else:
                # No tool calls, agent has completed its task
                if response.content is None:
                    response.content = ""
                    
                # Parse structured output if requested
                if self.output_type and response.content:
                    try:
                        # Try to parse as JSON if it looks like JSON
                        content = response.content.strip()
                        if content.startswith("{") and content.endswith("}"):
                            import json
                            parsed = json.loads(content)
                            return self.output_type(**parsed)
                        else:
                            # Fall back to returning the raw content
                            return response.content
                    except Exception as e:
                        # If parsing fails, return the raw content
                        return response.content
                else:
                    # Return raw string output
                    return response.content
        
        # If we hit the iteration limit without completion
        return "Maximum iterations reached without completing the task."
    
    async def astream(self, input_text: str) -> AsyncGenerator[StreamEvent, None]:
        """Run the agent asynchronously with streaming responses.
        
        Args:
            input_text: The input text from the user.
            
        Returns:
            An async generator that yields StreamEvent objects.
        """
        # Initialize conversation history
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": input_text}
        ]
        
        # Convert tools to the format expected by the model
        tool_dicts = [tool.to_dict() for tool in self.tools]
        
        # Run the agent loop
        iterations = 0
        final_content = ""
        
        while iterations < self.config.max_iterations:
            iterations += 1
            
            # Generate a streaming response from the model
            content_buffer = ""
            async for event in self.model.generate_stream(
                messages=messages,
                tools=tool_dicts if self.tools else None,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            ):
                # Pass through content chunk events
                if isinstance(event, ContentChunkEvent):
                    content_buffer += event.content
                    yield event
                
                # Handle tool call events
                elif isinstance(event, ToolCallEvent):
                    # Find the matching tool
                    tool_name = event.tool_name
                    tool = next((t for t in self.tools if t.name == tool_name), None)
                    
                    # Forward the tool call event
                    yield event
                    
                    if not tool:
                        error_msg = f"Tool '{tool_name}' not found"
                        yield ErrorEvent(message=error_msg)
                        tool_response = {"error": error_msg}
                    else:
                        try:
                            # Execute the tool with the provided arguments
                            tool_result = await tool.execute(**event.arguments)
                            
                            # Yield tool result event
                            yield ToolResultEvent(
                                tool_name=tool_name,
                                tool_id=event.tool_id,
                                result=tool_result
                            )
                            
                            tool_response = tool_result
                        except Exception as e:
                            error_message = str(e)
                            yield ErrorEvent(message=f"Error executing tool '{tool_name}': {error_message}")
                            tool_response = {"error": error_message}
                    
                    # Add the tool call and result to the conversation
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": event.tool_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(event.arguments)
                            }
                        }]
                    })
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": event.tool_id,
                        "content": str(tool_response),
                    })
                
                # Handle error events
                elif isinstance(event, ErrorEvent):
                    yield event
                
                # Handle done events
                elif isinstance(event, DoneEvent):
                    if event.final_content:
                        final_content = event.final_content
                    
                    # Check if there were tool calls by looking at the last message
                    if messages and messages[-1]["role"] == "tool":
                        # If there were tool calls, continue the conversation
                        break
                    else:
                        # No tool calls, we're done
                        yield DoneEvent(final_content=final_content)
                        
                        # Parse structured output if requested
                        if self.output_type and final_content:
                            try:
                                # Try to parse as JSON if it looks like JSON
                                content = final_content.strip()
                                if content.startswith("{") and content.endswith("}"):
                                    parsed = json.loads(content)
                                    structured_output = self.output_type(**parsed)
                                    # We're done here, return
                                    return
                            except Exception as e:
                                # If parsing fails, just continue
                                pass
                        
                        # We're done, return from the generator
                        return
            
            # If we've reached here, we've processed a complete model response
            # and potentially executed tools. If no tools were called, we're done.
            if not messages or messages[-1]["role"] != "tool":
                yield DoneEvent(final_content=final_content)
                return
        
        # If we hit the iteration limit without completion
        yield ErrorEvent(message="Maximum iterations reached without completing the task.")
        yield DoneEvent(final_content=final_content)
    
    def run(self, input_text: str) -> Any:
        """Run the agent synchronously.
        
        Args:
            input_text: The input text from the user.
            
        Returns:
            The agent's response.
        """
        return asyncio.run(self.arun(input_text))
    
    def stream(self, input_text: str) -> List[StreamEvent]:
        """Run the agent synchronously with streaming, collecting all events.
        
        Args:
            input_text: The input text from the user.
            
        Returns:
            A list of all stream events generated during execution.
        """
        events = []
        
        async def collect_events():
            async for event in self.astream(input_text):
                events.append(event)
                
        asyncio.run(collect_events())
        return events