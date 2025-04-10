import inspect
import json
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, Union, get_type_hints

from pydantic import BaseModel, create_model

from pydantic.json_schema import JsonSchemaMode, model_json_schema


class ToolType(str, Enum):
    """Types of tools supported by the framework."""
    
    FUNCTION = "function"


class Tool:
    """Base class for tools that can be used by agents."""
    
    def __init__(
        self,
        name: str,
        description: str,
        function: Callable,
        parameters_schema: Dict[str, Any],
    ):
        self.name = name
        self.description = description
        self.function = function
        self.parameters_schema = parameters_schema
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to a dictionary format compatible with model APIs."""
        return {
            "type": ToolType.FUNCTION,
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            }
        }
        
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with the given arguments."""
        if inspect.iscoroutinefunction(self.function):
            return await self.function(**kwargs)
        return self.function(**kwargs)


def _get_param_schema(func: Callable) -> Dict[str, Any]:
    """Extract parameter schema from a function's type hints."""
    hints = get_type_hints(func)
    
    # Remove return type from hints
    if "return" in hints:
        del hints["return"]
    
    # Get function signature
    sig = inspect.signature(func)
    
    # Build field info
    properties = {}
    required = []
    
    for name, param in sig.parameters.items():
        param_type = hints.get(name, Any)
        
        # Handle optional parameters
        is_optional = param.default != inspect.Parameter.empty
        
        field_info = {
            "type": _python_type_to_json_type(param_type),
        }
        
        # Add description from docstring if available
        if func.__doc__:
            param_doc = _extract_param_doc(func.__doc__, name)
            if param_doc:
                field_info["description"] = param_doc
                
        properties[name] = field_info
        
        if not is_optional:
            required.append(name)
    
    # Create the schema
    schema = {
        "type": "object",
        "properties": properties,
    }
    
    if required:
        schema["required"] = required
        
    return schema


def _extract_param_doc(docstring: str, param_name: str) -> Optional[str]:
    """Extract parameter documentation from function docstring."""
    lines = docstring.split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{param_name}:"):
            return line.strip().split(":", 1)[1].strip()
    return None


def _python_type_to_json_type(py_type: Type) -> str:
    """Map Python types to JSON schema types."""
    if py_type == str:
        return "string"
    elif py_type == int:
        return "integer"
    elif py_type == float:
        return "number"
    elif py_type == bool:
        return "boolean"
    elif py_type == list or py_type == List:
        return "array"
    elif py_type == dict or py_type == Dict:
        return "object"
    else:
        return "string"  # Default to string for complex types


def function_tool(func=None, *, name=None, description=None):
    """Decorator to convert a function into a tool."""
    
    def decorator(f):
        tool_name = name or f.__name__
        tool_description = description or (f.__doc__ and f.__doc__.strip().split("\n")[0]) or ""
        
        # Get JSON schema for function parameters
        param_schema = _get_param_schema(f)
        
        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        
        # Attach tool metadata
        wrapper._tool = Tool(
            name=tool_name,
            description=tool_description,
            function=f,
            parameters_schema=param_schema,
        )
        
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)


def pydantic_tool(func=None, *, input_model: Type[BaseModel] = None, name=None, description=None):
    """Create a tool from a function with a Pydantic model as input."""
    
    def decorator(f):
        tool_name = name or f.__name__
        tool_description = description or (f.__doc__ and f.__doc__.strip().split("\n")[0]) or ""
        
        if input_model is None:
            # Infer input model from function signature
            hints = get_type_hints(f)
            sig = inspect.signature(f)
            
            fields = {}
            for param_name, param in sig.parameters.items():
                param_type = hints.get(param_name, Any)
                default = ... if param.default == inspect.Parameter.empty else param.default
                fields[param_name] = (param_type, default)
                
            # Create a dynamic Pydantic model for the function parameters
            dynamic_model = create_model(f'{f.__name__}Model', **fields)
            param_model = dynamic_model
        else:
            param_model = input_model
            
        # Get JSON schema for the Pydantic model
        schema = model_json_schema(
            param_model,
            mode=JsonSchemaMode.OPENAPI_3_1
        )
        
        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        
        # Attach tool metadata
        wrapper._tool = Tool(
            name=tool_name,
            description=tool_description,
            function=f,
            parameters_schema=schema,
        )
        
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)