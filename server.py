from fastapi import FastAPI, Request, HTTPException
import uvicorn
import logging
import json
import re
import asyncio
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Union, Literal, Set
import os
from fastapi.responses import JSONResponse, StreamingResponse
import litellm
import uuid
import time
from dotenv import load_dotenv
from datetime import datetime
import sys

# Load environment variables early
load_dotenv()

# Basic LiteLLM Configuration - conservative settings to avoid hanging
litellm.drop_params = True
litellm.set_verbose = False
litellm.request_timeout = 90

# Constants for better maintainability  
class Constants:
    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"
    ROLE_SYSTEM = "system"
    ROLE_TOOL = "tool"
    
    CONTENT_TEXT = "text"
    CONTENT_IMAGE = "image"
    CONTENT_TOOL_USE = "tool_use"
    CONTENT_TOOL_RESULT = "tool_result"
    
    TOOL_FUNCTION = "function"
    
    STOP_END_TURN = "end_turn"
    STOP_MAX_TOKENS = "max_tokens"
    STOP_TOOL_USE = "tool_use"
    STOP_ERROR = "error"
    
    EVENT_MESSAGE_START = "message_start"
    EVENT_MESSAGE_STOP = "message_stop"
    EVENT_MESSAGE_DELTA = "message_delta"
    EVENT_CONTENT_BLOCK_START = "content_block_start"
    EVENT_CONTENT_BLOCK_STOP = "content_block_stop"
    EVENT_CONTENT_BLOCK_DELTA = "content_block_delta"
    EVENT_PING = "ping"
    
    DELTA_TEXT = "text_delta"
    DELTA_INPUT_JSON = "input_json_delta"

# Enhanced Configuration focused on Gemini and Vertex AI
class Config:
    def __init__(self):
        # Gemini configuration (required)
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Vertex AI configuration
        self.vertex_project_id = os.environ.get("VERTEX_PROJECT_ID")
        self.vertex_location = os.environ.get("VERTEX_LOCATION", "us-central1")
        self.google_application_credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        
        # Model configuration
        self.big_model = os.environ.get("BIG_MODEL", "gemini-1.5-pro-latest")
        self.small_model = os.environ.get("SMALL_MODEL", "gemini-1.5-flash-latest")
        
        # Provider preference (gemini or vertex)
        self.preferred_provider = os.environ.get("PREFERRED_PROVIDER", "gemini").lower()
        
        # Server configuration
        self.host = os.environ.get("HOST", "0.0.0.0")
        self.port = int(os.environ.get("PORT", "8082"))
        self.log_level = os.environ.get("LOG_LEVEL", "WARNING")
        self.max_tokens_limit = int(os.environ.get("MAX_TOKENS_LIMIT", "8192"))
        
        # Connection settings - conservative defaults
        self.request_timeout = int(os.environ.get("REQUEST_TIMEOUT", "90"))
        self.max_retries = int(os.environ.get("MAX_RETRIES", "2"))
        
        # Streaming settings
        self.max_streaming_retries = int(os.environ.get("MAX_STREAMING_RETRIES", "12"))
        self.force_disable_streaming = os.environ.get("FORCE_DISABLE_STREAMING", "false").lower() == "true"
        self.emergency_disable_streaming = os.environ.get("EMERGENCY_DISABLE_STREAMING", "false").lower() == "true"
        
        # Log Vertex AI configuration
        if self.vertex_project_id:
            print(f"🔵 Vertex AI Project ID configured: {self.vertex_project_id}")
            print(f"🔵 Vertex AI Location configured: {self.vertex_location}")
            if self.google_application_credentials:
                print(f"🔵 Vertex AI using service account credentials from: {self.google_application_credentials}")
            else:
                print("🔵 Vertex AI will use Application Default Credentials (ADC)")
        else:
            print("⚠️ VERTEX_PROJECT_ID not set. Vertex AI models will not work correctly without it.")
            
        print(f"🔧 Preferred provider set to: {self.preferred_provider}")
        
    def validate_api_key(self):
        """Basic API key validation"""
        if not self.gemini_api_key:
            return False
        # Basic format check for Google API keys
        if not (self.gemini_api_key.startswith('AIza') and len(self.gemini_api_key) == 39):
            return False
        return True

try:
    config = Config()
    print(f"✅ Configuration loaded: API_KEY={'*' * 20}..., BIG_MODEL='{config.big_model}', SMALL_MODEL='{config.small_model}'")
except Exception as e:
    print(f"🔴 Configuration Error: {e}")
    sys.exit(1)

# Apply connection settings to LiteLLM
litellm.request_timeout = config.request_timeout
litellm.num_retries = config.max_retries

# Model Management focused on Gemini and Vertex AI
class ModelManager:
    def __init__(self, config):
        self.config = config
        
        # Base Gemini models (Google AI Studio)
        self.base_gemini_models = [
            "gemini-1.5-pro-latest",
            "gemini-1.5-pro-preview-0514",
            "gemini-1.5-flash-latest", 
            "gemini-1.5-flash-preview-0514",
            "gemini-pro",
            "gemini-2.5-pro-preview-05-06",
            "gemini-2.5-flash-preview-04-17",
            "gemini-2.0-flash-exp",
            "gemini-exp-1206"
        ]
        
        # Vertex AI models
        self.vertex_ai_models = [
            "gemini-2.5-pro-preview-03-25",
            "gemini-2.0-flash",
            "gemini-1.5-flash-preview-0514",
            "gemini-1.5-pro-preview-0514",
            "gemini-2.5-flash-preview-04-17"
        ]
        
        self._gemini_models = set(self.base_gemini_models)
        self._add_env_models()
        
        print(f"🔵 Vertex AI models available: {', '.join(self.vertex_ai_models)}")
    
    def _add_env_models(self):
        for model in [self.config.big_model, self.config.small_model]:
            if model.startswith("gemini") and model not in self._gemini_models:
                self._gemini_models.add(model)
    
    @property
    def gemini_models(self) -> List[str]:
        return sorted(list(self._gemini_models))
    
    def validate_and_map_model(self, original_model: str) -> tuple[str, bool]:
        clean_model = self._clean_model_name(original_model)
        mapped_model = self._map_model_alias(clean_model)
        
        if mapped_model != clean_model:
            # Check which provider the mapped model should use
            if self.config.preferred_provider == "vertex" and mapped_model in self.vertex_ai_models:
                return f"vertex_ai/{mapped_model}", True
            else:
                return f"gemini/{mapped_model}", True
        elif clean_model in self.vertex_ai_models and not original_model.startswith('vertex_ai/'):
            return f"vertex_ai/{clean_model}", True
        elif clean_model in self._gemini_models:
            return f"gemini/{clean_model}", True
        elif not original_model.startswith(('gemini/', 'vertex_ai/')):
            return f"gemini/{original_model}", False
        else:
            return original_model, False
    
    def _clean_model_name(self, model: str) -> str:
        if model.startswith('gemini/'):
            return model[7:]
        elif model.startswith('vertex_ai/'):
            return model[10:]
        elif model.startswith('anthropic/'):
            return model[10:]
        return model
    
    def _map_model_alias(self, clean_model: str) -> str:
        model_lower = clean_model.lower()
        
        if 'haiku' in model_lower:
            return self.config.small_model
        elif 'sonnet' in model_lower or 'opus' in model_lower:
            return self.config.big_model
        
        return clean_model

model_manager = ModelManager(config)

# Logging Configuration
logging.basicConfig(
    level=getattr(logging, config.log_level.upper()),
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Simple message filter
class SimpleMessageFilter(logging.Filter):
    def filter(self, record):
        blocked_phrases = [
            "LiteLLM completion()",
            "HTTP Request:",
            "cost_calculator"
        ]
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            return not any(phrase in record.msg for phrase in blocked_phrases)
        return True

root_logger = logging.getLogger()
root_logger.addFilter(SimpleMessageFilter())

# Configure uvicorn to be quieter
for uvicorn_logger in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
    logging.getLogger(uvicorn_logger).setLevel(logging.WARNING)

app = FastAPI(title="Gemini-Vertex AI Proxy", version="2.7.0")

# Enhanced error classification
def classify_gemini_error(error_msg: str) -> str:
    """Provide specific error guidance for common Gemini issues."""
    error_lower = error_msg.lower()
    
    # Streaming/parsing errors
    if "error parsing chunk" in error_lower and "expecting property name" in error_lower:
        return "Gemini streaming parsing error (malformed JSON chunk). This is a known intermittent Gemini API issue. Please try again or disable streaming by setting stream=false."
    
    # Tool schema validation errors
    if "function_declarations" in error_lower and "format" in error_lower:
        if "only 'enum' and 'date-time' are supported" in error_lower:
            return "Tool schema error: Gemini only supports 'enum' and 'date-time' formats for string parameters. Remove other format types like 'url', 'email', 'uri', etc."
        else:
            return "Tool schema validation error. Check your tool parameter definitions for unsupported format types or properties."
    
    # Rate limiting
    elif "rate limit" in error_lower or "quota" in error_lower:
        return "Rate limit or quota exceeded. Please wait a moment and try again. Check your Google Cloud Console for quota limits."
    
    # Authentication issues
    elif "api key" in error_lower or "authentication" in error_lower or "unauthorized" in error_lower:
        return "API key error. Please check that your API key is valid and has the necessary permissions."
    
    # Parsing/streaming issues
    elif "parsing" in error_lower or "json" in error_lower or "malformed" in error_lower:
        return "Response parsing error. This is often a temporary API issue - please retry your request."
    
    # Connection issues
    elif "connection" in error_lower or "timeout" in error_lower:
        return "Connection or timeout error. Please check your internet connection and try again."
    
    # Safety/content filtering
    elif "safety" in error_lower or "content" in error_lower and "filter" in error_lower:
        return "Content filtered by safety systems. Please modify your request to comply with content policies."
    
    # Token/length issues
    elif "token" in error_lower and ("limit" in error_lower or "exceed" in error_lower):
        return "Token limit exceeded. Please reduce the length of your request or increase the max_tokens parameter."
    
    # Default: return original message
    return error_msg

# Enhanced schema cleaner for Gemini/Vertex AI
def clean_gemini_schema(schema: Any) -> Any:
    """Recursively removes unsupported fields from a JSON schema for Gemini/Vertex compatibility."""
    if isinstance(schema, dict):
        # Remove fields unsupported by Gemini/Vertex
        schema.pop("additionalProperties", None)
        schema.pop("default", None)

        # Handle string format restrictions
        if schema.get("type") == "string" and "format" in schema:
            allowed_formats = {"enum", "date-time"}
            if schema["format"] not in allowed_formats:
                logger.debug(f"Removing unsupported format '{schema['format']}' for string type in Gemini/Vertex schema")
                schema.pop("format")

        # Recursively clean nested schemas
        for key, value in list(schema.items()):
            schema[key] = clean_gemini_schema(value)
                
    elif isinstance(schema, list):
        return [clean_gemini_schema(item) for item in schema]
            
    return schema

# Pydantic Models
class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str

class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]

class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]

class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any]]

class SystemContent(BaseModel):
    type: Literal["text"]
    text: str

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]]]

class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class ThinkingConfig(BaseModel):
    enabled: bool = True

class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    original_model: Optional[str] = None

    @field_validator('model')
    @classmethod
    def validate_model_field(cls, v, info):
        original_model = v
        mapped_model, was_mapped = model_manager.validate_and_map_model(v)
        
        logger.debug(f"📋 MODEL VALIDATION: Original='{original_model}', Preferred='{config.preferred_provider}', Big='{config.big_model}', Small='{config.small_model}'")
        
        if was_mapped:
            # Enhanced logging with provider-specific emojis
            if mapped_model.startswith("vertex_ai/"):
                logger.info(f"🔵 VERTEX MODEL MAPPING: '{original_model}' ➡️ '{mapped_model}'")
            elif mapped_model.startswith("gemini/"):
                logger.info(f"🟡 GEMINI MODEL MAPPING: '{original_model}' ➡️ '{mapped_model}'")
            else:
                logger.debug(f"📌 MODEL MAPPING: '{original_model}' ➡️ '{mapped_model}'")
        
        if info and hasattr(info, 'data') and isinstance(info.data, dict):
            info.data['original_model'] = original_model
            
        return mapped_model

class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None

    @field_validator('model')
    @classmethod
    def validate_model_token_count(cls, v, info):
        mapped_model, _ = model_manager.validate_and_map_model(v)
        if info and hasattr(info, 'data') and isinstance(info.data, dict):
            info.data['original_model'] = v
        return mapped_model

class TokenCountResponse(BaseModel):
    input_tokens: int

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = Constants.ROLE_ASSISTANT
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "error"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage

# Tool result parsing
def parse_tool_result_content(content):
    """Parse and normalize tool result content into a string format."""
    if content is None:
        return "No content provided"

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        result_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == Constants.CONTENT_TEXT:
                result_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                result_parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    result_parts.append(item.get("text", ""))
                else:
                    try:
                        result_parts.append(json.dumps(item))
                    except:
                        result_parts.append(str(item))
        return "\n".join(result_parts).strip()

    if isinstance(content, dict):
        if content.get("type") == Constants.CONTENT_TEXT:
            return content.get("text", "")
        try:
            return json.dumps(content)
        except:
            return str(content)

    try:
        return str(content)
    except:
        return "Unparseable content"

# Enhanced message conversion
def convert_anthropic_to_litellm(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    """Convert Anthropic API request format to LiteLLM format."""
    litellm_messages = []
    
    # System message handling
    if anthropic_request.system:
        system_text = ""
        if isinstance(anthropic_request.system, str):
            system_text = anthropic_request.system
        elif isinstance(anthropic_request.system, list):
            text_parts = []
            for block in anthropic_request.system:
                if hasattr(block, 'type') and block.type == Constants.CONTENT_TEXT:
                    text_parts.append(block.text)
                elif isinstance(block, dict) and block.get("type") == Constants.CONTENT_TEXT:
                    text_parts.append(block.get("text", ""))
            system_text = "\n\n".join(text_parts)
        
        if system_text.strip():
            litellm_messages.append({"role": Constants.ROLE_SYSTEM, "content": system_text.strip()})

    # Process messages
    for msg in anthropic_request.messages:
        if isinstance(msg.content, str):
            litellm_messages.append({"role": msg.role, "content": msg.content})
            continue

        # Process content blocks - accumulate different types
        text_parts = []
        image_parts = []
        tool_calls = []
        pending_tool_messages = []

        for block in msg.content:
            if block.type == Constants.CONTENT_TEXT:
                text_parts.append(block.text)
            elif block.type == Constants.CONTENT_IMAGE:
                if (isinstance(block.source, dict) and 
                    block.source.get("type") == "base64" and
                    "media_type" in block.source and "data" in block.source):
                    image_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{block.source['media_type']};base64,{block.source['data']}"
                        }
                    })
            elif block.type == Constants.CONTENT_TOOL_USE and msg.role == Constants.ROLE_ASSISTANT:
                tool_calls.append({
                    "id": block.id,
                    "type": Constants.TOOL_FUNCTION,
                    Constants.TOOL_FUNCTION: {
                        "name": block.name,
                        "arguments": json.dumps(block.input)
                    }
                })
            elif block.type == Constants.CONTENT_TOOL_RESULT and msg.role == Constants.ROLE_USER:
                # CRITICAL: Split user message when tool_result is encountered
                if text_parts or image_parts:
                    content_parts = []
                    text_content = "".join(text_parts).strip()
                    if text_content:
                        content_parts.append({"type": Constants.CONTENT_TEXT, "text": text_content})
                    content_parts.extend(image_parts)
                    
                    litellm_messages.append({
                        "role": Constants.ROLE_USER,
                        "content": content_parts[0]["text"] if len(content_parts) == 1 and content_parts[0]["type"] == Constants.CONTENT_TEXT else content_parts
                    })
                    text_parts.clear()
                    image_parts.clear()

                # Add tool result as separate "tool" role message
                parsed_content = parse_tool_result_content(block.content)
                pending_tool_messages.append({
                    "role": Constants.ROLE_TOOL,
                    "tool_call_id": block.tool_use_id,
                    "content": parsed_content
                })

        # Finalize message based on role
        if msg.role == Constants.ROLE_USER:
            # Add any remaining text/image content
            if text_parts or image_parts:
                content_parts = []
                text_content = "".join(text_parts).strip()
                if text_content:
                    content_parts.append({"type": Constants.CONTENT_TEXT, "text": text_content})
                content_parts.extend(image_parts)
                
                litellm_messages.append({
                    "role": Constants.ROLE_USER,
                    "content": content_parts[0]["text"] if len(content_parts) == 1 and content_parts[0]["type"] == Constants.CONTENT_TEXT else content_parts
                })
            # Add any pending tool messages
            litellm_messages.extend(pending_tool_messages)
            
        elif msg.role == Constants.ROLE_ASSISTANT:
            assistant_msg = {"role": Constants.ROLE_ASSISTANT}
            
            # Handle content for assistant messages
            content_parts = []
            text_content = "".join(text_parts).strip()
            if text_content:
                content_parts.append({"type": Constants.CONTENT_TEXT, "text": text_content})
            content_parts.extend(image_parts)
            
            # FIXED: Don't set content to None - let LiteLLM handle missing content
            if content_parts:
                assistant_msg["content"] = content_parts[0]["text"] if len(content_parts) == 1 and content_parts[0]["type"] == Constants.CONTENT_TEXT else content_parts
            else: 
                assistant_msg["content"] = None
                
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
                
            # Only add message if it has actual content or tool calls
            if assistant_msg.get("content") or assistant_msg.get("tool_calls"):
                litellm_messages.append(assistant_msg)

    # Build final LiteLLM request
    litellm_request = {
        "model": anthropic_request.model,
        "messages": litellm_messages,
        "max_tokens": min(anthropic_request.max_tokens, config.max_tokens_limit),
        "temperature": anthropic_request.temperature,
        "stream": anthropic_request.stream,
        # Add timeout configuration for long conversations
        "timeout": 600,  # 10 minutes
        "request_timeout": 600,
    }

    # Add optional parameters
    if anthropic_request.stop_sequences:
        litellm_request["stop"] = anthropic_request.stop_sequences
    if anthropic_request.top_p is not None:
        litellm_request["top_p"] = anthropic_request.top_p
    if anthropic_request.top_k is not None:
        litellm_request["topK"] = anthropic_request.top_k

    # Add tools with schema cleaning for Gemini/Vertex
    if anthropic_request.tools:
        valid_tools = []
        is_gemini_vertex = anthropic_request.model.startswith(("gemini/", "vertex_ai/"))
        
        for tool in anthropic_request.tools:
            if tool.name and tool.name.strip():
                input_schema = tool.input_schema
                if is_gemini_vertex:
                    input_schema = clean_gemini_schema(tool.input_schema)
                    
                valid_tools.append({
                    "type": Constants.TOOL_FUNCTION,
                    Constants.TOOL_FUNCTION: {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": input_schema
                    }
                })
        if valid_tools:
            litellm_request["tools"] = valid_tools

    # Add tool choice configuration
    if anthropic_request.tool_choice:
        choice_type = anthropic_request.tool_choice.get("type")
        if choice_type == "auto":
            litellm_request["tool_choice"] = "auto"
        elif choice_type == "any":
            litellm_request["tool_choice"] = "auto"
        elif choice_type == "tool" and "name" in anthropic_request.tool_choice:
            litellm_request["tool_choice"] = {
                "type": Constants.TOOL_FUNCTION, 
                Constants.TOOL_FUNCTION: {"name": anthropic_request.tool_choice["name"]}
            }
        else:
            litellm_request["tool_choice"] = "auto"

    # Add thinking configuration (Gemini specific)
    if anthropic_request.thinking is not None:
        if anthropic_request.thinking.enabled:
            litellm_request["thinkingConfig"] = {"thinkingBudget": 24576}
        else:
            litellm_request["thinkingConfig"] = {"thinkingBudget": 0}

    # Add user metadata if provided
    if (anthropic_request.metadata and 
        "user_id" in anthropic_request.metadata and
        isinstance(anthropic_request.metadata["user_id"], str)):
        litellm_request["user"] = anthropic_request.metadata["user_id"]

    return litellm_request

# Response conversion
def convert_litellm_to_anthropic(litellm_response, original_request: MessagesRequest) -> MessagesResponse:
    """Convert LiteLLM response back to Anthropic API format."""
    try:
        # Extract response data safely
        response_id = f"msg_{uuid.uuid4()}"
        content_text = ""
        tool_calls = None
        finish_reason = "stop"
        prompt_tokens = 0
        completion_tokens = 0

        # Handle LiteLLM ModelResponse object format
        if hasattr(litellm_response, 'choices') and hasattr(litellm_response, 'usage'):
            choices = litellm_response.choices
            message = choices[0].message if choices else None
            content_text = getattr(message, 'content', "") or ""
            tool_calls = getattr(message, 'tool_calls', None)
            finish_reason = choices[0].finish_reason if choices else "stop"
            response_id = getattr(litellm_response, 'id', response_id)
            
            if hasattr(litellm_response, 'usage'):
                usage = litellm_response.usage
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                completion_tokens = getattr(usage, "completion_tokens", 0)
                
        # Handle dictionary response format
        elif isinstance(litellm_response, dict):
            choices = litellm_response.get("choices", [])
            message = choices[0].get("message", {}) if choices else {}
            content_text = message.get("content", "") or ""
            tool_calls = message.get("tool_calls")
            finish_reason = choices[0].get("finish_reason", "stop") if choices else "stop"
            usage = litellm_response.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            response_id = litellm_response.get("id", response_id)

        # Build content blocks
        content_blocks = []
        
        # Add text content if present
        if content_text:
            content_blocks.append(ContentBlockText(type=Constants.CONTENT_TEXT, text=content_text))

        # Process tool calls
        if tool_calls:
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]

            for tool_call in tool_calls:
                try:
                    # Extract tool call data from different formats
                    if isinstance(tool_call, dict):
                        tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
                        function_data = tool_call.get(Constants.TOOL_FUNCTION, {})
                        name = function_data.get("name", "")
                        arguments_str = function_data.get("arguments", "{}")
                    elif hasattr(tool_call, "id") and hasattr(tool_call, Constants.TOOL_FUNCTION):
                        tool_id = tool_call.id
                        name = tool_call.function.name
                        arguments_str = tool_call.function.arguments
                    else:
                        continue

                    if not name:
                        continue

                    # Parse tool arguments safely
                    try:
                        arguments_dict = json.loads(arguments_str)
                    except json.JSONDecodeError:
                        arguments_dict = {"raw_arguments": arguments_str}

                    content_blocks.append(ContentBlockToolUse(
                        type=Constants.CONTENT_TOOL_USE,
                        id=tool_id,
                        name=name,
                        input=arguments_dict
                    ))
                except Exception as e:
                    logger.warning(f"Error processing tool call: {e}")
                    continue

        # Ensure at least one content block
        if not content_blocks:
            content_blocks.append(ContentBlockText(type=Constants.CONTENT_TEXT, text=""))

        # Map finish reason to Anthropic format
        if finish_reason == "length":
            stop_reason = Constants.STOP_MAX_TOKENS
        elif finish_reason == "tool_calls":
            stop_reason = Constants.STOP_TOOL_USE
        elif finish_reason is None and tool_calls:
            stop_reason = Constants.STOP_TOOL_USE
        else:
            stop_reason = Constants.STOP_END_TURN

        return MessagesResponse(
            id=response_id,
            model=original_request.original_model or original_request.model,
            role=Constants.ROLE_ASSISTANT,
            content=content_blocks,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=Usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens
            )
        )
        
    except Exception as e:
        logger.error(f"Error converting response: {e}")
        return MessagesResponse(
            id=f"msg_error_{uuid.uuid4()}",
            model=original_request.original_model or original_request.model,
            role=Constants.ROLE_ASSISTANT, 
            content=[ContentBlockText(type=Constants.CONTENT_TEXT, text="Response conversion error")],
            stop_reason=Constants.STOP_ERROR,
            usage=Usage(input_tokens=0, output_tokens=0)
        )

# Enhanced streaming handler with timeout improvements
async def handle_streaming_with_recovery(response_generator, original_request: MessagesRequest):
    """Enhanced streaming handler with robust error recovery and timeout handling."""
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    
    # Send initial SSE events
    yield f"event: {Constants.EVENT_MESSAGE_START}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_START, 'message': {'id': message_id, 'type': 'message', 'role': Constants.ROLE_ASSISTANT, 'model': original_request.original_model or original_request.model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
    
    yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': 0, 'content_block': {'type': Constants.CONTENT_TEXT, 'text': ''}})}\n\n"
    
    yield f"event: {Constants.EVENT_PING}\ndata: {json.dumps({'type': Constants.EVENT_PING})}\n\n"

    # Streaming state management
    accumulated_text = ""
    text_block_index = 0
    tool_block_counter = 0
    current_tool_calls = {}
    input_tokens = 0
    output_tokens = 0
    final_stop_reason = Constants.STOP_END_TURN
    
    # Enhanced error recovery tracking
    consecutive_errors = 0
    max_consecutive_errors = 10
    stream_terminated_early = False
    malformed_chunks_count = 0
    max_malformed_chunks = 20
    
    # Timeout handling
    has_sent_stop_reason = False
    last_ping = time.time()  # Track last ping time
    PING_INTERVAL = 30  # Send ping every 30 seconds
    
    # Buffer for incomplete chunks
    chunk_buffer = ""
    
    def is_malformed_chunk(chunk_str: str) -> bool:
        """Enhanced malformed chunk detection."""
        if not chunk_str or not isinstance(chunk_str, str):
            return True
            
        chunk_stripped = chunk_str.strip()
        
        # Empty or whitespace
        if not chunk_stripped:
            return True
            
        # Single characters that indicate malformed JSON
        malformed_singles = ["{", "}", "[", "]", ",", ":", '"', "'"]
        if chunk_stripped in malformed_singles:
            return True
            
        # Common malformed patterns
        malformed_patterns = [
            '{"', '"}', "[{", "}]", "{}", "[]", 
            "null", '""', "''", " ", "",
            "{,", ",}", "[,", ",]"
        ]
        if chunk_stripped in malformed_patterns:
            return True
            
        # Incomplete JSON structures
        if chunk_stripped.startswith('{') and not chunk_stripped.endswith('}'):
            if len(chunk_stripped) < 15:
                return True
                
        if chunk_stripped.startswith('[') and not chunk_stripped.endswith(']'):
            if len(chunk_stripped) < 10:
                return True
        
        # Check for obviously broken JSON patterns
        if chunk_stripped.count('{') != chunk_stripped.count('}'):
            if len(chunk_stripped) < 20:
                return True
                
        if chunk_stripped.count('[') != chunk_stripped.count(']'):
            if len(chunk_stripped) < 20:
                return True
        
        return False
    
    def try_parse_buffered_chunk(buffer: str) -> tuple[dict, str]:
        """Try to parse buffered chunks, return parsed chunk and remaining buffer."""
        if not buffer.strip():
            return None, ""
            
        # Try to find complete JSON objects in the buffer
        brace_count = 0
        start_pos = -1
        
        for i, char in enumerate(buffer):
            if char == '{':
                if start_pos == -1:
                    start_pos = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_pos != -1:
                    # Found complete JSON object
                    json_str = buffer[start_pos:i+1]
                    try:
                        parsed = json.loads(json_str)
                        remaining_buffer = buffer[i+1:]
                        return parsed, remaining_buffer
                    except json.JSONDecodeError:
                        continue
        
        # No complete JSON found
        return None, buffer
    
    try:
        # Wrap the entire streaming process in comprehensive error handling
        stream_iterator = aiter(response_generator)
        
        while True:
            try:
                # Send periodic keep-alive pings to prevent timeout
                if time.time() - last_ping > PING_INTERVAL:
                    yield f"event: {Constants.EVENT_PING}\ndata: {json.dumps({'type': Constants.EVENT_PING})}\n\n"
                    last_ping = time.time()
                
                # Get next chunk with timeout
                try:
                    chunk = await asyncio.wait_for(anext(stream_iterator), timeout=90.0)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    logger.warning("Streaming timeout, terminating")
                    stream_terminated_early = True
                    break
                
                # Reset consecutive error counter on successful chunk retrieval
                consecutive_errors = 0
                
                # Check if this is the end of the response with usage data
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    if hasattr(chunk.usage, 'prompt_tokens'):
                        input_tokens = chunk.usage.prompt_tokens
                    if hasattr(chunk.usage, 'completion_tokens'):
                        output_tokens = chunk.usage.completion_tokens
                
                # Handle string chunks with enhanced validation
                if isinstance(chunk, str):
                    if chunk.strip() == "[DONE]":
                        break
                    
                    # Check for malformed chunks
                    if is_malformed_chunk(chunk):
                        malformed_chunks_count += 1
                        logger.debug(f"Skipping malformed chunk #{malformed_chunks_count}: '{chunk[:50]}{'...' if len(chunk) > 50 else ''}'")
                        
                        if malformed_chunks_count > max_malformed_chunks:
                            logger.error(f"Too many malformed chunks ({malformed_chunks_count}), terminating stream")
                            stream_terminated_early = True
                            break
                        continue
                    
                    # Add to buffer and try to parse
                    chunk_buffer += chunk
                    parsed_chunk, chunk_buffer = try_parse_buffered_chunk(chunk_buffer)
                    
                    if parsed_chunk is None:
                        # Keep buffering if we don't have a complete chunk yet
                        if len(chunk_buffer) > 10000:  # Prevent buffer from growing too large
                            logger.warning("Chunk buffer too large, clearing")
                            chunk_buffer = ""
                        continue
                    
                    chunk = parsed_chunk
                
                # If we have a dictionary at this point, process it
                if isinstance(chunk, dict):
                    # Process the chunk normally (existing logic)
                    pass
                elif hasattr(chunk, 'choices'):
                    # Process ModelResponse object normally (existing logic)
                    pass
                else:
                    # Try one more JSON parse attempt
                    try:
                        if isinstance(chunk, str):
                            chunk = json.loads(chunk)
                        else:
                            logger.debug(f"Skipping unprocessable chunk type: {type(chunk)}")
                            continue
                    except json.JSONDecodeError as parse_error:
                        logger.debug(f"Failed to parse chunk as JSON: {parse_error}")
                        continue

                # Extract chunk data
                delta_content_text = None
                delta_tool_calls = None
                chunk_finish_reason = None

                if hasattr(chunk, 'choices') and chunk.choices:
                    choice = chunk.choices[0]
                    if hasattr(choice, 'delta') and choice.delta:
                        delta = choice.delta
                        delta_content_text = getattr(delta, 'content', None)
                        if hasattr(delta, 'tool_calls'):
                            delta_tool_calls = delta.tool_calls
                    chunk_finish_reason = getattr(choice, 'finish_reason', None)
                elif isinstance(chunk, dict):
                    choices = chunk.get("choices", [])
                    if choices:
                        choice = choices[0]
                        delta = choice.get("delta", {})
                        delta_content_text = delta.get("content")
                        delta_tool_calls = delta.get("tool_calls")
                        chunk_finish_reason = choice.get("finish_reason")

                # Handle text delta
                if delta_content_text:
                    accumulated_text += delta_content_text
                    yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': text_block_index, 'delta': {'type': Constants.DELTA_TEXT, 'text': delta_content_text}})}\n\n"

                # Handle tool call deltas
                if delta_tool_calls:
                    for tc_chunk in delta_tool_calls:
                        if not (hasattr(tc_chunk, 'function') and tc_chunk.function and 
                               hasattr(tc_chunk.function, 'name') and tc_chunk.function.name):
                            continue
                            
                        tool_call_id = tc_chunk.id
                        
                        if tool_call_id not in current_tool_calls:
                            tool_block_counter += 1
                            tool_index = text_block_index + tool_block_counter
                            
                            current_tool_calls[tool_call_id] = {
                                "index": tool_index,
                                "name": tc_chunk.function.name or "",
                                "args_buffer": tc_chunk.function.arguments or ""
                            }
                            
                            yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': tool_index, 'content_block': {'type': Constants.CONTENT_TOOL_USE, 'id': tool_call_id, 'name': current_tool_calls[tool_call_id]['name'], 'input': {}}})}\n\n"
                        
                        if tc_chunk.function.arguments:
                            current_tool_calls[tool_call_id]["args_buffer"] += tc_chunk.function.arguments
                            yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': current_tool_calls[tool_call_id]['index'], 'delta': {'type': Constants.DELTA_INPUT_JSON, 'partial_json': tc_chunk.function.arguments}})}\n\n"

                # Handle finish reason
                if chunk_finish_reason and not has_sent_stop_reason:
                    has_sent_stop_reason = True
                    if chunk_finish_reason == "length":
                        final_stop_reason = Constants.STOP_MAX_TOKENS
                    elif chunk_finish_reason == "tool_calls":
                        final_stop_reason = Constants.STOP_TOOL_USE
                    elif chunk_finish_reason == "stop":
                        final_stop_reason = Constants.STOP_END_TURN
                    else:
                        final_stop_reason = Constants.STOP_END_TURN
                    break
                        
            except (json.JSONDecodeError, ValueError) as parse_error:
                consecutive_errors += 1
                logger.debug(f"JSON parsing error (attempt {consecutive_errors}/{max_consecutive_errors}): {parse_error}")
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"Too many consecutive parsing errors ({consecutive_errors}), terminating stream")
                    stream_terminated_early = True
                    break
                continue
                
            except (litellm.exceptions.APIConnectionError, RuntimeError) as api_error:
                consecutive_errors += 1
                error_msg = str(api_error)
                
                # Check for the specific malformed chunk error
                if ("Error parsing chunk" in error_msg and 
                    "Expecting property name enclosed in double quotes" in error_msg):
                    
                    logger.warning(f"API malformed chunk error (attempt {consecutive_errors}/{max_consecutive_errors})")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"Too many consecutive API errors ({consecutive_errors}), terminating stream")
                        stream_terminated_early = True
                        
                        # Send error info to client
                        error_text = f"\n⚠️ Streaming encountered repeated malformed chunks. This is a known API issue.\n"
                        yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': text_block_index, 'delta': {'type': Constants.DELTA_TEXT, 'text': error_text}})}\n\n"
                        break
                    
                    # Brief delay before continuing
                    await asyncio.sleep(0.1)
                    continue
                else:
                    # Other API errors - terminate immediately
                    logger.error(f"API error: {api_error}")
                    stream_terminated_early = True
                    break
                    
            except Exception as general_error:
                consecutive_errors += 1
                logger.error(f"Unexpected streaming error (attempt {consecutive_errors}/{max_consecutive_errors}): {general_error}")
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"Too many consecutive errors ({consecutive_errors}), terminating stream")
                    stream_terminated_early = True
                    break
                
                # Brief delay before continuing
                await asyncio.sleep(0.1)
                continue

    except Exception as outer_error:
        logger.error(f"Fatal streaming error: {outer_error}")
        stream_terminated_early = True

    # Always send final SSE events
    try:
        yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': text_block_index})}\n\n"
        
        for tool_data in current_tool_calls.values():
            yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': tool_data['index']})}\n\n"
        
        if stream_terminated_early and final_stop_reason == Constants.STOP_END_TURN:
            final_stop_reason = Constants.STOP_ERROR
        
        usage_data = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        yield f"event: {Constants.EVENT_MESSAGE_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_DELTA, 'delta': {'stop_reason': final_stop_reason, 'stop_sequence': None}, 'usage': usage_data})}\n\n"
        yield f"event: {Constants.EVENT_MESSAGE_STOP}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_STOP})}\n\n"
        
        # Log final statistics
        if malformed_chunks_count > 0:
            logger.info(f"Stream completed with {malformed_chunks_count} malformed chunks handled")
            
    except Exception as final_error:
        logger.error(f"Error sending final SSE events: {final_error}")

# Request Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    method = request.method
    path = request.url.path
    logger.debug(f"Request: {method} {path}")
    response = await call_next(request)
    return response

# Enhanced main endpoint focused on Gemini/Vertex AI
@app.post("/v1/messages")
async def create_message(request: MessagesRequest, raw_request: Request):
    try:
        logger.debug(f"📊 Processing request: Original={request.original_model}, Effective={request.model}, Stream={request.stream}")

        # Check streaming configuration
        if request.stream and config.emergency_disable_streaming:
            logger.warning("Streaming disabled via EMERGENCY_DISABLE_STREAMING")
            request.stream = False

        if request.stream and config.force_disable_streaming:
            logger.info("Streaming disabled via FORCE_DISABLE_STREAMING")
            request.stream = False

        # Convert request
        litellm_request = convert_anthropic_to_litellm(request)
        
        # Determine API key/credentials based on the final model's provider prefix
        if request.model.startswith("gemini/"):
            litellm_request["api_key"] = config.gemini_api_key
            logger.debug(f"Using Gemini (Google AI Studio) API key for model: {request.model}")
            
        elif request.model.startswith("vertex_ai/"):
            if config.vertex_project_id:
                litellm_request["vertex_project"] = config.vertex_project_id
                litellm_request["vertex_location"] = config.vertex_location
                logger.info(f"🔵 Using Vertex AI credentials (Project: {config.vertex_project_id}, Location: {config.vertex_location}) for model: {request.model}")

                # Check for Application Default Credentials or service account
                if config.google_application_credentials:
                    logger.info(f"🔵 Vertex AI using service account from: {config.google_application_credentials}")
                else:
                    logger.info("🔵 Vertex AI using Application Default Credentials (ADC)")
            else:
                raise HTTPException(status_code=400, detail="VERTEX_PROJECT_ID not set for Vertex AI model. Please configure VERTEX_PROJECT_ID environment variable.")
                
        else:
            # Default fallback - use Gemini
            litellm_request["api_key"] = config.gemini_api_key
            logger.debug(f"Using default Gemini API key for model: {request.model}")
        
        # Log request details
        num_tools = len(request.tools) if request.tools else 0
        log_request_beautifully(
            "POST", raw_request.url.path,
            request.original_model or request.model,
            litellm_request.get('model'),
            len(litellm_request['messages']),
            num_tools, 200
        )

        # Enhanced streaming with better retry logic
        if request.stream:
            streaming_retry_count = 0
            max_retries = config.max_streaming_retries
            
            while streaming_retry_count <= max_retries:
                try:
                    logger.debug(f"Attempting streaming (attempt {streaming_retry_count + 1}/{max_retries + 1})")
                    
                    # Add slight delay between retries
                    if streaming_retry_count > 0:
                        delay = min(0.5 * (2 ** streaming_retry_count), 2.0)  # Exponential backoff, max 2s
                        logger.debug(f"Waiting {delay}s before retry...")
                        await asyncio.sleep(delay)
                    
                    response_generator = await litellm.acompletion(**litellm_request)
                    
                    # Add streaming-specific headers to prevent buffering
                    headers = {
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",  # Disable nginx buffering
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "*"
                    }
                    
                    return StreamingResponse(
                        handle_streaming_with_recovery(response_generator, request),
                        media_type="text/event-stream",
                        headers=headers
                    )
                    
                except (litellm.exceptions.APIConnectionError, RuntimeError) as streaming_error:
                    streaming_retry_count += 1
                    error_msg = str(streaming_error)
                    
                    # Check for the specific malformed chunk error
                    if ("Error parsing chunk" in error_msg and 
                        "Expecting property name enclosed in double quotes" in error_msg):
                        
                        if streaming_retry_count <= max_retries:
                            logger.warning(f"Streaming chunk parsing error (attempt {streaming_retry_count}/{max_retries + 1}), retrying...")
                            continue
                        else:
                            logger.error(f"Streaming failed after {max_retries + 1} attempts due to malformed chunks, falling back to non-streaming")
                            break
                    else:
                        # Other streaming errors - could be connection issues
                        if streaming_retry_count <= max_retries:
                            logger.warning(f"Streaming error (attempt {streaming_retry_count}/{max_retries + 1}): {error_msg}")
                            continue
                        else:
                            logger.error(f"Streaming failed after {max_retries + 1} attempts, falling back to non-streaming")
                            break
                            
                except Exception as unexpected_error:
                    streaming_retry_count += 1
                    logger.error(f"Unexpected streaming error (attempt {streaming_retry_count}/{max_retries + 1}): {unexpected_error}")
                    
                    if streaming_retry_count <= max_retries:
                        continue
                    else:
                        logger.error(f"Streaming failed after {max_retries + 1} attempts due to unexpected errors, falling back to non-streaming")
                        break
            
            # If we get here, streaming failed - fall back to non-streaming
            logger.info("Falling back to non-streaming mode")
            litellm_request["stream"] = False
        
        # Non-streaming path (or fallback)
        if not request.stream or litellm_request.get("stream") == False:
            start_time = time.time()
            litellm_response = await litellm.acompletion(**litellm_request)
            elapsed_time = time.time() - start_time
            
            # Enhanced provider-specific response logging
            if request.model.startswith("vertex_ai/"):
                logger.info(f"🔵 VERTEX RESPONSE RECEIVED: Model={request.model}, Time={elapsed_time:.2f}s")
            elif request.model.startswith("gemini/"):
                logger.info(f"🟡 GEMINI RESPONSE RECEIVED: Model={request.model}, Time={elapsed_time:.2f}s")
            else:
                logger.info(f"✅ RESPONSE RECEIVED: Model={request.model}, Time={elapsed_time:.2f}s")
            
            anthropic_response = convert_litellm_to_anthropic(litellm_response, request)
            return anthropic_response

    except litellm.exceptions.APIError as e:
        logger.error(f"LiteLLM API Error: {e}")
        error_msg = classify_gemini_error(str(e))
        raise HTTPException(status_code=getattr(e, 'status_code', 500), detail=error_msg)
    except ConnectionError as e:
        logger.error(f"Connection Error: {e}")
        raise HTTPException(status_code=503, detail="Connection error. Please check your internet connection.")
    except TimeoutError as e:
        logger.error(f"Timeout Error: {e}")
        raise HTTPException(status_code=504, detail="Request timeout. Please try again.")
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        error_msg = classify_gemini_error(str(e))
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/v1/messages/count_tokens")
async def count_tokens(request: TokenCountRequest, raw_request: Request):
    try:
        # Create temporary request for conversion
        temp_request = MessagesRequest(
            model=request.model,
            max_tokens=1,
            messages=request.messages,
            system=request.system,
            tools=request.tools,
        )
        
        litellm_data = convert_anthropic_to_litellm(temp_request)
        
        # Log request
        num_tools = len(request.tools) if request.tools else 0
        log_request_beautifully(
            "POST", raw_request.url.path,
            request.original_model or request.model,
            litellm_data.get('model'),
            len(litellm_data['messages']), num_tools, 200
        )

        # Count tokens
        token_count = litellm.token_counter(
            model=litellm_data["model"],
            messages=litellm_data["messages"],
        )
        
        return TokenCountResponse(input_tokens=token_count)

    except Exception as e:
        logger.error(f"Error counting tokens: {str(e)}")
        error_msg = classify_gemini_error(str(e))
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {error_msg}")

@app.get("/health")
async def health_check():
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.7.0",
            "providers": {
                "gemini": {
                    "configured": bool(config.gemini_api_key),
                    "api_key_valid": config.validate_api_key()
                },
                "vertex_ai": {
                    "configured": bool(config.vertex_project_id),
                    "project_id": config.vertex_project_id,
                    "location": config.vertex_location,
                    "credentials_type": "service_account" if config.google_application_credentials else "adc"
                }
            },
            "preferred_provider": config.preferred_provider,
            "streaming_config": {
                "force_disabled": config.force_disable_streaming,
                "emergency_disabled": config.emergency_disable_streaming,
                "max_retries": config.max_streaming_retries
            }
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": "Health check failed"
            }
        )

@app.get("/test-connection")
async def test_connection():
    """Test API connectivity to configured providers"""
    try:
        test_results = {}
        
        # Test Gemini
        if config.gemini_api_key:
            try:
                test_response = await litellm.acompletion(
                    model="gemini/gemini-1.5-flash-latest",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5,
                    api_key=config.gemini_api_key
                )
                test_results["gemini"] = {
                    "status": "success",
                    "model_used": "gemini-1.5-flash-latest",
                    "response_id": getattr(test_response, 'id', 'unknown')
                }
            except Exception as e:
                test_results["gemini"] = {
                    "status": "failed",
                    "error": classify_gemini_error(str(e))
                }
        
        # Test Vertex AI
        if config.vertex_project_id:
            try:
                test_response = await litellm.acompletion(
                    model="vertex_ai/gemini-1.5-flash-preview-0514",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5,
                    vertex_project=config.vertex_project_id,
                    vertex_location=config.vertex_location
                )
                test_results["vertex_ai"] = {
                    "status": "success",
                    "model_used": "gemini-1.5-flash-preview-0514",
                    "project_id": config.vertex_project_id,
                    "response_id": getattr(test_response, 'id', 'unknown')
                }
            except Exception as e:
                test_results["vertex_ai"] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        return {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "test_results": test_results,
            "suggestions": {
                "gemini": "Check your GEMINI_API_KEY is valid",
                "vertex_ai": "Ensure VERTEX_PROJECT_ID is set and ADC/service account is configured"
            }
        }
        
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "failed",
                "error_type": "Test Error", 
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/")
async def root():
    return {
        "message": f"Gemini-Vertex AI Proxy v2.7.0",
        "status": "running",
        "config": {
            "preferred_provider": config.preferred_provider,
            "big_model": config.big_model,
            "small_model": config.small_model,
            "available_models": {
                "gemini": model_manager.gemini_models[:5],
                "vertex_ai": model_manager.vertex_ai_models[:5]
            },
            "max_tokens_limit": config.max_tokens_limit,
            "providers_configured": {
                "gemini": bool(config.gemini_api_key),
                "vertex_ai": bool(config.vertex_project_id)
            },
            "streaming": {
                "force_disabled": config.force_disable_streaming,
                "emergency_disabled": config.emergency_disable_streaming,
                "max_retries": config.max_streaming_retries
            }
        },
        "endpoints": {
            "messages": "/v1/messages",
            "count_tokens": "/v1/messages/count_tokens", 
            "health": "/health",
            "test_connection": "/test-connection"
        }
    }

# Enhanced logging utilities
class Colors:
    CYAN = "\033[96m"    # Original model
    BLUE = "\033[94m"    # Vertex AI
    YELLOW = "\033[93m"  # Gemini
    MAGENTA = "\033[95m" # Tools/Messages count
    GREEN = "\033[92m"   # Success
    RED = "\033[91m"     # Error
    RESET = "\033[0m"
    BOLD = "\033[1m"

def log_request_beautifully(method: str, path: str, requested_model: str, 
                           mapped_model: str, num_messages: int, 
                           num_tools: int, status_code: int):
    if not sys.stdout.isatty():
        print(f"{method} {path} - {requested_model} -> {mapped_model} ({num_messages} messages, {num_tools} tools)")
        return
    
    # Colorized logging for TTY
    req_display = f"{Colors.CYAN}{requested_model}{Colors.RESET}"
    
    # Determine target provider and apply appropriate color
    target_provider = "unknown"
    target_model_name = mapped_model
    target_color = Colors.YELLOW  # Default to Gemini color
    
    if "/" in mapped_model:
        try:
            target_provider, target_model_name = mapped_model.split("/", 1)
            if target_provider == "gemini":
                target_color = Colors.YELLOW
            elif target_provider == "vertex_ai":
                target_color = Colors.BLUE
        except ValueError:
            logger.warning(f"Could not parse provider from mapped model: {mapped_model}")
            target_provider = "unknown"
            target_model_name = mapped_model
    
    target_display = f"{target_color}{target_model_name}{Colors.RESET}"
    
    endpoint = path.split("?")[0] if "?" in path else path
    tools_str = f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET}"
    messages_str = f"{Colors.MAGENTA}{num_messages} messages{Colors.RESET}"
    
    if status_code == 200:
        status_str = f"{Colors.GREEN}✓ {status_code} OK{Colors.RESET}"
    else:
        status_str = f"{Colors.RED}✗ {status_code}{Colors.RESET}"

    log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
    model_line = f"{req_display} → {target_display} ({target_provider}) ({tools_str}, {messages_str})"

    print(log_line)
    print(model_line)
    sys.stdout.flush()

def validate_startup():
    """Validate configuration and connectivity on startup"""
    print("🔍 Validating startup configuration...")
    
    # Check API keys
    if not config.gemini_api_key:
        print("🔴 FATAL: GEMINI_API_KEY is not set")
        return False
    
    if not config.validate_api_key():
        print("⚠️ WARNING: Gemini API key format validation failed")
    
    # Check network connectivity (basic)
    try:
        import socket
        socket.create_connection(("8.8.8.8", 53), timeout=10)
        print("✅ Network connectivity: OK")
    except OSError:
        print("⚠️ WARNING: Network connectivity check failed")
        
    return True

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Gemini-Vertex AI Proxy v2.7.0")
        print("")
        print("Usage: uvicorn server:app --reload --host 0.0.0.0 --port 8082")
        print("")
        print("Required environment variables:")
        print("  GEMINI_API_KEY - Your Google Gemini API key")
        print("")
        print("Optional environment variables:")
        print("  PREFERRED_PROVIDER - Preferred provider (gemini or vertex)")
        print("  VERTEX_PROJECT_ID - Google Cloud Project ID for Vertex AI")
        print("  VERTEX_LOCATION - Vertex AI location (default: us-central1)")
        print("  GOOGLE_APPLICATION_CREDENTIALS - Path to service account JSON")
        print(f"  BIG_MODEL - Big model name (default: gemini-1.5-pro-latest)")
        print(f"  SMALL_MODEL - Small model name (default: gemini-1.5-flash-latest)")
        print(f"  HOST - Server host (default: 0.0.0.0)")
        print(f"  PORT - Server port (default: 8082)")
        print(f"  LOG_LEVEL - Logging level (default: WARNING)")
        print(f"  MAX_TOKENS_LIMIT - Token limit (default: 8192)")
        print(f"  REQUEST_TIMEOUT - Request timeout in seconds (default: 90)")
        print(f"  MAX_RETRIES - Maximum retries (default: 2)")
        print(f"  MAX_STREAMING_RETRIES - Maximum streaming retries (default: 12)")
        print(f"  FORCE_DISABLE_STREAMING - Force disable streaming (default: false)")
        print(f"  EMERGENCY_DISABLE_STREAMING - Emergency disable streaming (default: false)")
        print("")
        print("Available models by provider:")
        print("  Gemini:")
        for model in model_manager.gemini_models:
            print(f"    - {model}")
        print("  Vertex AI:")
        for model in model_manager.vertex_ai_models:
            print(f"    - {model}")
        sys.exit(0)

    # Validate startup configuration
    if not validate_startup():
        print("🔴 Startup validation failed. Please check your configuration.")
        sys.exit(1)

    # Configuration summary
    print("🚀 Gemini-Vertex AI Proxy v2.7.0")
    print(f"✅ Configuration loaded successfully")
    print(f"   Preferred Provider: {config.preferred_provider}")
    print(f"   Big Model: {config.big_model}")
    print(f"   Small Model: {config.small_model}")
    print(f"   Available Gemini Models: {len(model_manager.gemini_models)}")
    print(f"   Available Vertex AI Models: {len(model_manager.vertex_ai_models)}")
    print(f"   Max Tokens Limit: {config.max_tokens_limit}")
    print(f"   Request Timeout: {config.request_timeout}s")
    print(f"   Max Retries: {config.max_retries}")
    print(f"   Max Streaming Retries: {config.max_streaming_retries}")
    print(f"   Force Disable Streaming: {config.force_disable_streaming}")
    print(f"   Emergency Disable Streaming: {config.emergency_disable_streaming}")
    print(f"   Log Level: {config.log_level}")
    print(f"   Server: {config.host}:{config.port}")
    print("")

    # Configure uvicorn to run with minimal logs and streaming timeouts
    uvicorn.run(
        app, 
        host=config.host, 
        port=config.port, 
        log_level=config.log_level.lower(),
        timeout_keep_alive=300,  # 5 minutes keep-alive for long conversations
        timeout_graceful_shutdown=30,
        limit_max_requests=1000
    )

if __name__ == "__main__":
    main()
