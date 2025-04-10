from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
from pydantic import BaseModel as PydanticBaseModel, Field

from ..stream import StreamEvent


class ModelResponse(PydanticBaseModel):
    """A standardized response format from any model provider."""
    
    content: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    

class ModelBase(ABC):
    """Base interface for all model providers."""
    
    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ModelResponse:
        """Generate a response from the model.
        
        Args:
            messages: List of message objects (system, user, assistant).
            tools: Optional list of tools available to the model.
            temperature: Optional temperature setting for randomness.
            max_tokens: Optional maximum tokens to generate.
            
        Returns:
            ModelResponse object containing the model's response.
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Generate a streaming response from the model.
        
        Args:
            messages: List of message objects (system, user, assistant).
            tools: Optional list of tools available to the model.
            temperature: Optional temperature setting for randomness.
            max_tokens: Optional maximum tokens to generate.
            
        Returns:
            An async generator yielding StreamEvent objects.
        """
        pass