"""
Stream event definitions for the simple agent framework.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Types of stream events."""
    
    CONTENT_CHUNK = "content_chunk"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    DONE = "done"


class StreamEvent(BaseModel):
    """Base class for all stream events."""
    
    type: EventType
    

class ContentChunkEvent(StreamEvent):
    """Event for a chunk of content from the model."""
    
    type: EventType = EventType.CONTENT_CHUNK
    content: str
    

class ToolCallEvent(StreamEvent):
    """Event for a tool call from the model."""
    
    type: EventType = EventType.TOOL_CALL
    tool_name: str
    tool_id: str
    arguments: Dict[str, Any]
    

class ToolResultEvent(StreamEvent):
    """Event for a tool execution result."""
    
    type: EventType = EventType.TOOL_RESULT
    tool_name: str
    tool_id: str
    result: Any
    

class ErrorEvent(StreamEvent):
    """Event for an error during streaming."""
    
    type: EventType = EventType.ERROR
    message: str
    

class DoneEvent(StreamEvent):
    """Event indicating the stream is complete."""
    
    type: EventType = EventType.DONE
    final_content: Optional[str] = None