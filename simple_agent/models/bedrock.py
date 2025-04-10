import json
from typing import Any, AsyncGenerator, Dict, List, Optional

import boto3
from pydantic import BaseModel, Field

from .base import ModelBase, ModelResponse
from ..stream import StreamEvent, ContentChunkEvent, ToolCallEvent, DoneEvent, ErrorEvent


class BedrockModel(ModelBase):
    """AWS Bedrock model provider implementation."""
    
    def __init__(
        self,
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        region_name: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        """Initialize the Bedrock model.
        
        Args:
            model_id: The Bedrock model ID to use.
            region_name: AWS region name.
            aws_access_key_id: Optional AWS access key ID.
            aws_secret_access_key: Optional AWS secret access key.
        """
        self.model_id = model_id
        
        # Create Bedrock runtime client
        session_kwargs = {"region_name": region_name}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs.update({
                "aws_access_key_id": aws_access_key_id,
                "aws_secret_access_key": aws_secret_access_key,
            })
            
        session = boto3.Session(**session_kwargs)
        self.client = session.client("bedrock-runtime")
        
        # Determine model provider from model_id for appropriate formatting
        self.provider = model_id.split(".")[0] if "." in model_id else "anthropic"
        
    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = None,
    ) -> ModelResponse:
        """Generate a response from Bedrock."""
        
        # Format request body based on provider
        if self.provider == "anthropic":
            body = self._format_anthropic_request(messages, tools, temperature, max_tokens)
        elif self.provider == "amazon":
            body = self._format_titan_request(messages, tools, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported model provider: {self.provider}")
        
        # Invoke model
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body)
        )
        
        response_body = json.loads(response.get("body").read().decode("utf-8"))
        
        return self._parse_response(response_body)
    
    async def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Generate a streaming response from Bedrock."""
        
        try:
            # Format request body based on provider with streaming enabled
            if self.provider == "anthropic":
                body = self._format_anthropic_request(messages, tools, temperature, max_tokens)
                body["stream"] = True
            elif self.provider == "amazon":
                body = self._format_titan_request(messages, tools, temperature, max_tokens)
                # Amazon Titan uses a different streaming parameter
                body["streamEnabled"] = True
            else:
                raise ValueError(f"Unsupported model provider: {self.provider}")
            
            # Invoke model with streaming
            response_stream = self.client.invoke_model_with_response_stream(
                modelId=self.model_id,
                body=json.dumps(body)
            )
            
            content_buffer = ""
            tool_calls_buffer = {}
            
            # Process the streaming response based on provider
            if self.provider == "anthropic":
                # Process Claude streaming response
                for event in response_stream.get("body"):
                    if "chunk" in event:
                        chunk_data = json.loads(event["chunk"]["bytes"].decode("utf-8"))
                        
                        # Handle content chunks
                        if "type" in chunk_data and chunk_data["type"] == "content_block_delta":
                            if "delta" in chunk_data and "text" in chunk_data["delta"]:
                                text = chunk_data["delta"]["text"]
                                content_buffer += text
                                yield ContentChunkEvent(content=text)
                        
                        # Handle tool use (Claude's tool calling)
                        elif "type" in chunk_data and chunk_data["type"] == "tool_use":
                            tool_name = chunk_data.get("name", "")
                            tool_id = chunk_data.get("id", f"tool_{len(tool_calls_buffer)}")
                            arguments = chunk_data.get("input", {})
                            
                            yield ToolCallEvent(
                                tool_name=tool_name,
                                tool_id=tool_id,
                                arguments=arguments
                            )
                            
                        # Handle completion events
                        elif "type" in chunk_data and chunk_data["type"] == "message_stop":
                            yield DoneEvent(final_content=content_buffer)
                
            elif self.provider == "amazon":
                # Process Titan streaming response
                for event in response_stream.get("body"):
                    if "chunk" in event:
                        chunk_data = json.loads(event["chunk"]["bytes"].decode("utf-8"))
                        
                        if "outputText" in chunk_data:
                            text = chunk_data["outputText"]
                            content_buffer += text
                            yield ContentChunkEvent(content=text)
                            
                # Yield final event when stream is complete
                yield DoneEvent(final_content=content_buffer)
                
        except Exception as e:
            yield ErrorEvent(message=f"Error during streaming: {str(e)}")
            yield DoneEvent(final_content=None)
    
    def _format_anthropic_request(
        self, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]], 
        temperature: Optional[float],
        max_tokens: Optional[int]
    ) -> Dict[str, Any]:
        """Format request for Claude models."""
        
        # Extract system message if present
        system = None
        formatted_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                formatted_messages.append({
                    "role": "user" if msg["role"] == "user" else "assistant",
                    "content": msg["content"]
                })
        
        request = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": formatted_messages,
            "temperature": temperature or 0.7,
        }
        
        if system:
            request["system"] = system
            
        if max_tokens:
            request["max_tokens"] = max_tokens
            
        if tools:
            request["tools"] = tools
            
        return request
    
    def _format_titan_request(
        self, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]], 
        temperature: Optional[float],
        max_tokens: Optional[int]
    ) -> Dict[str, Any]:
        """Format request for Amazon Titan models."""
        
        # Combine messages into a formatted prompt
        prompt = ""
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            prompt += f"{role}: {content}\n"
        
        prompt += "Assistant: "
        
        request = {
            "inputText": prompt,
            "textGenerationConfig": {
                "temperature": temperature or 0.7,
            }
        }
        
        if max_tokens:
            request["textGenerationConfig"]["maxTokenCount"] = max_tokens
            
        # Note: As of my knowledge, Titan models have limited tool use capabilities
        # This implementation may need to be updated as Bedrock adds features
        
        return request
    
    def _parse_response(self, response_body: Dict[str, Any]) -> ModelResponse:
        """Parse response from different model providers."""
        
        if self.provider == "anthropic":
            content = response_body.get("content", [{}])[0].get("text", "")
            tool_calls = []
            
            # Extract tool calls if present
            if "tool_use" in response_body:
                tool_uses = response_body.get("tool_use", [])
                for i, tool_use in enumerate(tool_uses):
                    tool_calls.append({
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": tool_use.get("name", ""),
                            "arguments": tool_use.get("input", {})
                        }
                    })
                    
            return ModelResponse(content=content, tool_calls=tool_calls)
            
        elif self.provider == "amazon":
            return ModelResponse(
                content=response_body.get("results", [{}])[0].get("outputText", ""),
                tool_calls=[]  # Titan has limited tool support as of now
            )
            
        else:
            raise ValueError(f"Unsupported model provider: {self.provider}")