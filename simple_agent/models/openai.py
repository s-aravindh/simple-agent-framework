import json
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from .base import ModelBase, ModelResponse
from ..stream import StreamEvent, ContentChunkEvent, ToolCallEvent, DoneEvent, ErrorEvent


class OpenAIModel(ModelBase):
    """OpenAI model provider implementation."""
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Initialize the OpenAI model.
        
        Args:
            model: The OpenAI model name to use.
            api_key: Optional API key (will use env var if not provided).
            organization: Optional organization ID.
            base_url: Optional base URL for the API.
        """
        self.model = model
        self.client = AsyncOpenAI(
            api_key=api_key,
            organization=organization,
            base_url=base_url
        )
        
    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = None,
    ) -> ModelResponse:
        """Generate a response from OpenAI."""
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
            
        if tools:
            kwargs["tools"] = tools
            
        response = await self.client.chat.completions.create(**kwargs)
        
        # Extract the completion message
        message = response.choices[0].message
        
        # Parse into our standard format
        model_response = ModelResponse(
            content=message.content,
            tool_calls=[
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments)
                    }
                }
                for tc in message.tool_calls or []
            ]
        )
        
        return model_response
    
    async def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Generate a streaming response from OpenAI."""
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,  # Enable streaming
        }
        
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
            
        if tools:
            kwargs["tools"] = tools
            
        try:
            # Get the streaming response
            stream = await self.client.chat.completions.create(**kwargs)
            
            tool_calls_buffer = {}  # Buffer to accumulate tool call chunks
            content_buffer = ""  # Buffer to accumulate content chunks
            
            async for chunk in stream:
                delta = chunk.choices[0].delta
                
                # Process content chunks
                if delta.content:
                    content_buffer += delta.content
                    yield ContentChunkEvent(content=delta.content)
                
                # Process tool calls
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        tc_id = tc_delta.id or tc_delta.index
                        
                        # Initialize the tool call in the buffer if it's new
                        if tc_id not in tool_calls_buffer:
                            tool_calls_buffer[tc_id] = {
                                "id": tc_id,
                                "type": "function",
                                "function": {
                                    "name": "",
                                    "arguments": ""
                                }
                            }
                        
                        # Update the tool call data with this chunk
                        if tc_delta.function.name:
                            tool_calls_buffer[tc_id]["function"]["name"] = tc_delta.function.name
                            
                        if tc_delta.function.arguments:
                            tool_calls_buffer[tc_id]["function"]["arguments"] += tc_delta.function.arguments
                            
                        # If we have a complete tool call, emit an event
                        if (chunk.choices[0].finish_reason == "tool_calls" and
                            tool_calls_buffer[tc_id]["function"]["name"] and
                            tool_calls_buffer[tc_id]["function"]["arguments"]):
                            
                            try:
                                args = json.loads(tool_calls_buffer[tc_id]["function"]["arguments"])
                                
                                yield ToolCallEvent(
                                    tool_name=tool_calls_buffer[tc_id]["function"]["name"],
                                    tool_id=tc_id,
                                    arguments=args
                                )
                            except json.JSONDecodeError:
                                # Arguments might be incomplete JSON
                                pass
            
            # Yield final event when stream is complete
            yield DoneEvent(final_content=content_buffer)
            
        except Exception as e:
            yield ErrorEvent(message=f"Error during streaming: {str(e)}")
            # Yield a done event to signal the end of the stream
            yield DoneEvent(final_content=None)