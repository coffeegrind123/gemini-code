#!/usr/bin/env python3
"""
Comprehensive test suite for Gemini-Vertex AI Proxy.

This script provides tests for both streaming and non-streaming requests,
with various scenarios including tool use, multi-turn conversations,
model mapping, provider switching, and content blocks.

Usage:
  python tests.py                    # Run all tests
  python tests.py --no-streaming     # Skip streaming tests
  python tests.py --simple           # Run only simple tests
  python tests.py --tools            # Run tool-related tests only
  python tests.py --vertex-only      # Test only Vertex AI models
  python tests.py --gemini-only      # Test only Gemini models
  python tests.py --health-only      # Test only health/diagnostic endpoints
"""

import os
import json
import time
import httpx
import argparse
import asyncio
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
PROXY_BASE_URL = os.environ.get("PROXY_BASE_URL", "http://localhost:8082")
PROXY_API_URL = f"{PROXY_BASE_URL}/v1/messages"
PROXY_TOKEN_COUNT_URL = f"{PROXY_BASE_URL}/v1/messages/count_tokens"
PROXY_HEALTH_URL = f"{PROXY_BASE_URL}/health"
PROXY_TEST_CONNECTION_URL = f"{PROXY_BASE_URL}/test-connection"

# Test configuration
DEFAULT_MAX_TOKENS = 300
DEFAULT_TEMPERATURE = 0.7

# Headers for proxy requests (mimicking Claude Code)
proxy_headers = {
    "content-type": "application/json",
}

# Color codes for output
class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

def colorize(text, color):
    """Add color to text if stdout is a TTY."""
    if sys.stdout.isatty():
        return f"{color}{text}{Colors.RESET}"
    return text

# Tool definitions
calculator_tool = {
    "name": "calculator",
    "description": "Evaluate mathematical expressions",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate"
            }
        },
        "required": ["expression"]
    }
}

weather_tool = {
    "name": "weather",
    "description": "Get weather information for a location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city or location to get weather for"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature units"
            }
        },
        "required": ["location"]
    }
}

search_tool = {
    "name": "search",
    "description": "Search for information on the web",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string", 
                "description": "The search query"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "minimum": 1,
                "maximum": 10
            }
        },
        "required": ["query"]
    }
}

file_tool = {
    "name": "file_operations",
    "description": "Perform file operations like reading, writing, or listing files",
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["read", "write", "list", "delete"],
                "description": "The file operation to perform"
            },
            "path": {
                "type": "string",
                "description": "The file or directory path"
            },
            "content": {
                "type": "string",
                "description": "Content to write (for write operations)"
            }
        },
        "required": ["operation", "path"]
    }
}

# Test scenarios organized by category
TEST_SCENARIOS = {
    # ================= BASIC FUNCTIONALITY =================
    "simple_text": {
        "model": "claude-3-5-haiku-20241022",  # Will map to small model
        "max_tokens": DEFAULT_MAX_TOKENS,
        "messages": [
            {"role": "user", "content": "Hello! Can you tell me about the city of Paris in 2-3 sentences?"}
        ]
    },
    
    "simple_sonnet": {
        "model": "claude-3-5-sonnet-20241022",  # Will map to big model
        "max_tokens": DEFAULT_MAX_TOKENS,
        "messages": [
            {"role": "user", "content": "Explain quantum computing in simple terms."}
        ]
    },
    
    "direct_gemini": {
        "model": "gemini/gemini-1.5-flash-latest",  # Direct Gemini model
        "max_tokens": DEFAULT_MAX_TOKENS,
        "messages": [
            {"role": "user", "content": "What are the benefits of renewable energy?"}
        ]
    },
    
    "direct_vertex": {
        "model": "vertex_ai/gemini-2.0-flash",  # Direct Vertex AI model
        "max_tokens": DEFAULT_MAX_TOKENS,
        "messages": [
            {"role": "user", "content": "Describe the process of photosynthesis."}
        ]
    },
    
    # ================= SYSTEM PROMPTS =================
    "with_system_prompt": {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 400,
        "system": "You are a helpful assistant that always responds in a cheerful and encouraging tone.",
        "messages": [
            {"role": "user", "content": "I'm feeling nervous about a job interview tomorrow."}
        ]
    },
    
    "system_content_blocks": {
        "model": "claude-3-5-haiku-20241022",
        "max_tokens": 300,
        "system": [
            {"type": "text", "text": "You are a technical expert who explains complex topics clearly."}
        ],
        "messages": [
            {"role": "user", "content": "What is machine learning?"}
        ]
    },
    
    # ================= TOOL USE TESTS =================
    "basic_tool_use": {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 500,
        "messages": [
            {"role": "user", "content": "What is 142.5 + 87.3 divided by 3?"}
        ],
        "tools": [calculator_tool],
        "tool_choice": {"type": "auto"}
    },
    
    "multiple_tools": {
        "model": "gemini/gemini-1.5-pro-latest",
        "max_tokens": 600,
        "temperature": DEFAULT_TEMPERATURE,
        "system": "You are a helpful assistant that uses tools when appropriate. Be concise and precise.",
        "messages": [
            {"role": "user", "content": "I'm planning a trip to Tokyo next week. What's the weather like and can you search for some popular attractions?"}
        ],
        "tools": [weather_tool, search_tool],
        "tool_choice": {"type": "auto"}
    },
    
    "complex_tool_schema": {
        "model": "vertex_ai/gemini-1.5-flash-preview-0514",
        "max_tokens": 500,
        "messages": [
            {"role": "user", "content": "Can you help me organize my project files? I need to list the contents of my Documents folder."}
        ],
        "tools": [file_tool],
        "tool_choice": {"type": "auto"}
    },
    
    "tool_choice_specific": {
        "model": "claude-3-5-haiku-20241022",
        "max_tokens": 400,
        "messages": [
            {"role": "user", "content": "Tell me about machine learning and also calculate 50 * 12."}
        ],
        "tools": [calculator_tool, search_tool],
        "tool_choice": {"type": "tool", "name": "calculator"}
    },
    
    # ================= MULTI-TURN CONVERSATIONS =================
    "multi_turn_basic": {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 500,
        "messages": [
            {"role": "user", "content": "Let's do some math. What is 240 divided by 8?"},
            {"role": "assistant", "content": "To calculate 240 divided by 8, I'll perform the division:\n\n240 ÷ 8 = 30\n\nSo the result is 30."},
            {"role": "user", "content": "Now multiply that result by 7 and tell me the answer."}
        ],
        "tools": [calculator_tool],
        "tool_choice": {"type": "auto"}
    },
    
    "multi_turn_with_tools": {
        "model": "gemini/gemini-1.5-flash-latest",
        "max_tokens": 700,
        "messages": [
            {"role": "user", "content": "I need help with some calculations for my project."},
            {"role": "assistant", "content": "I'd be happy to help you with calculations for your project! What specific calculations do you need assistance with?"},
            {"role": "user", "content": "First, can you calculate the area of a rectangle that's 15.5 meters by 8.2 meters?"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "I'll calculate the area of that rectangle for you."},
                {"type": "tool_use", "id": "calc_1", "name": "calculator", "input": {"expression": "15.5 * 8.2"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "calc_1", "content": "127.1"},
                {"type": "text", "text": "Great! Now can you also calculate what 25% of that area would be?"}
            ]}
        ],
        "tools": [calculator_tool]
    },
    
    # ================= CONTENT BLOCKS =================
    "content_blocks_mixed": {
        "model": "claude-3-5-haiku-20241022",
        "max_tokens": 500,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "I need help with multiple tasks: "},
                {"type": "text", "text": "1. Calculate 125.75 / 4.5"},
                {"type": "text", "text": "2. Get weather for Seattle"},
                {"type": "text", "text": "Can you help with both?"}
            ]}
        ],
        "tools": [calculator_tool, weather_tool],
        "tool_choice": {"type": "auto"}
    },
    
    # ================= PARAMETER TESTING =================
    "temperature_test": {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 300,
        "temperature": 0.1,  # Low temperature for consistency
        "messages": [
            {"role": "user", "content": "Write a short creative story about a robot learning to paint."}
        ]
    },
    
    "top_p_top_k_test": {
        "model": "gemini/gemini-1.5-pro-latest",
        "max_tokens": 400,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 40,
        "messages": [
            {"role": "user", "content": "Generate 5 creative names for a new coffee shop."}
        ]
    },
    
    "stop_sequences_test": {
        "model": "claude-3-5-haiku-20241022",
        "max_tokens": 500,
        "stop_sequences": ["END", "STOP"],
        "messages": [
            {"role": "user", "content": "Count from 1 to 10, then write END when you're done."}
        ]
    },
    
    # ================= STREAMING TESTS =================
    "simple_streaming": {
        "model": "claude-3-5-haiku-20241022",
        "max_tokens": 200,
        "stream": True,
        "messages": [
            {"role": "user", "content": "Count from 1 to 10, with one number per line."}
        ]
    },
    
    "streaming_with_tools": {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 400,
        "stream": True,
        "messages": [
            {"role": "user", "content": "Calculate 156.7 + 89.3 and then tell me what that result divided by 5 would be."}
        ],
        "tools": [calculator_tool],
        "tool_choice": {"type": "auto"}
    },
    
    "long_streaming": {
        "model": "gemini/gemini-1.5-flash-latest",
        "max_tokens": 800,
        "stream": True,
        "temperature": 0.7,
        "messages": [
            {"role": "user", "content": "Write a detailed explanation of how neural networks work, including backpropagation."}
        ]
    },
    
    "streaming_vertex": {
        "model": "vertex_ai/gemini-2.0-flash",
        "max_tokens": 300,
        "stream": True,
        "messages": [
            {"role": "user", "content": "Explain the benefits of cloud computing in bullet points."}
        ]
    },
    
    # ================= THINKING TESTS =================
    "thinking_enabled": {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 600,
        "thinking": {"enabled": True},
        "messages": [
            {"role": "user", "content": "Solve this step by step: If a train travels 120 km in 2 hours, and then 180 km in 3 hours, what is its average speed for the entire journey?"}
        ]
    },
    
    "thinking_disabled": {
        "model": "gemini/gemini-1.5-pro-latest",
        "max_tokens": 400,
        "thinking": {"enabled": False},
        "messages": [
            {"role": "user", "content": "What is the capital of Australia?"}
        ]
    },
    
    # ================= ERROR HANDLING TESTS =================
    "high_token_limit": {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 10000,  # Should be capped by proxy
        "messages": [
            {"role": "user", "content": "Write a very short poem about nature."}
        ]
    },
    
    "empty_content": {
        "model": "claude-3-5-haiku-20241022",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": ""}
        ]
    }
}

# Required event types for Anthropic streaming responses
REQUIRED_EVENT_TYPES = {
    "message_start", 
    "content_block_start", 
    "content_block_delta", 
    "content_block_stop", 
    "message_delta", 
    "message_stop"
}

# ================= HEALTH CHECK TESTS =================

async def test_health_endpoint():
    """Test the health check endpoint."""
    print(colorize("\n" + "="*20 + " TESTING HEALTH ENDPOINT " + "="*20, Colors.BOLD))
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(PROXY_HEALTH_URL, timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                print(colorize("✅ Health check passed", Colors.GREEN))
                print(f"Status: {health_data.get('status')}")
                print(f"Version: {health_data.get('version')}")
                
                providers = health_data.get('providers', {})
                for provider, config in providers.items():
                    status_icon = "✅" if config.get('configured') else "❌"
                    print(f"{status_icon} {provider.title()}: {'configured' if config.get('configured') else 'not configured'}")
                
                return True
            else:
                print(colorize(f"❌ Health check failed: {response.status_code}", Colors.RED))
                print(response.text)
                return False
                
    except Exception as e:
        print(colorize(f"❌ Health check error: {e}", Colors.RED))
        return False

async def test_connection_endpoint():
    """Test the connection test endpoint."""
    print(colorize("\n" + "="*20 + " TESTING CONNECTION ENDPOINT " + "="*20, Colors.BOLD))
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(PROXY_TEST_CONNECTION_URL, timeout=30)
            
            if response.status_code == 200:
                conn_data = response.json()
                print(colorize("✅ Connection test passed", Colors.GREEN))
                
                test_results = conn_data.get('test_results', {})
                for provider, result in test_results.items():
                    status = result.get('status')
                    if status == 'success':
                        print(colorize(f"✅ {provider.title()}: Connected successfully", Colors.GREEN))
                        if 'model_used' in result:
                            print(f"   Model: {result['model_used']}")
                    else:
                        print(colorize(f"❌ {provider.title()}: {result.get('error', 'Unknown error')}", Colors.RED))
                
                return True
            else:
                print(colorize(f"❌ Connection test failed: {response.status_code}", Colors.RED))
                print(response.text)
                return False
                
    except Exception as e:
        print(colorize(f"❌ Connection test error: {e}", Colors.RED))
        return False

async def test_token_counting():
    """Test the token counting endpoint."""
    print(colorize("\n" + "="*20 + " TESTING TOKEN COUNTING " + "="*20, Colors.BOLD))
    
    test_data = {
        "model": "claude-3-5-haiku-20241022",
        "messages": [
            {"role": "user", "content": "Hello, how are you doing today?"}
        ]
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(PROXY_TOKEN_COUNT_URL, json=test_data, headers=proxy_headers, timeout=10)
            
            if response.status_code == 200:
                token_data = response.json()
                input_tokens = token_data.get('input_tokens', 0)
                print(colorize(f"✅ Token counting passed: {input_tokens} tokens", Colors.GREEN))
                return True
            else:
                print(colorize(f"❌ Token counting failed: {response.status_code}", Colors.RED))
                print(response.text)
                return False
                
    except Exception as e:
        print(colorize(f"❌ Token counting error: {e}", Colors.RED))
        return False

# ================= NON-STREAMING TESTS =================

async def test_request(test_name, request_data, check_tools=False):
    """Run a test with the given request data."""
    print(colorize(f"\n{'='*20} RUNNING TEST: {test_name} {'='*20, Colors.BOLD))
    
    # Log the request data (excluding messages for brevity)
    log_data = {k: v for k, v in request_data.items() if k != 'messages'}
    log_data['message_count'] = len(request_data.get('messages', []))
    print(f"Request config: {json.dumps(log_data, indent=2)}")
    
    start_time = time.time()
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(PROXY_API_URL, json=request_data, headers=proxy_headers, timeout=60)
            
        elapsed = time.time() - start_time
        print(f"Response time: {elapsed:.2f} seconds")
        
        if response.status_code != 200:
            print(colorize(f"❌ Request failed with status {response.status_code}", Colors.RED))
            try:
                error_data = response.json()
                print(f"Error: {error_data.get('detail', response.text)}")
            except:
                print(f"Error text: {response.text}")
            return False
        
        # Parse response
        response_data = response.json()
        
        # Basic structure validation
        required_fields = ['id', 'model', 'role', 'content', 'type', 'usage']
        for field in required_fields:
            if field not in response_data:
                print(colorize(f"❌ Missing required field: {field}", Colors.RED))
                return False
        
        # Validate response structure
        if response_data.get('role') != 'assistant':
            print(colorize(f"❌ Invalid role: {response_data.get('role')}", Colors.RED))
            return False
        
        if response_data.get('type') != 'message':
            print(colorize(f"❌ Invalid type: {response_data.get('type')}", Colors.RED))
            return False
        
        content = response_data.get('content', [])
        if not isinstance(content, list) or len(content) == 0:
            print(colorize("❌ Invalid or empty content", Colors.RED))
            return False
        
        # Check content blocks
        has_text = False
        has_tool_use = False
        
        for block in content:
            if block.get('type') == 'text':
                has_text = True
                text_content = block.get('text', '')
                if text_content:
                    print(f"Response preview: {text_content[:100]}{'...' if len(text_content) > 100 else ''}")
            elif block.get('type') == 'tool_use':
                has_tool_use = True
                print(colorize(f"Tool use: {block.get('name')} with input: {block.get('input')}", Colors.MAGENTA))
        
        # Validate tool usage if expected
        if check_tools:
            if has_tool_use:
                print(colorize("✅ Tool use detected as expected", Colors.GREEN))
            else:
                print(colorize("⚠️ No tool use detected (tools were available)", Colors.YELLOW))
        
        # Check usage information
        usage = response_data.get('usage', {})
        input_tokens = usage.get('input_tokens', 0)
        output_tokens = usage.get('output_tokens', 0)
        print(f"Token usage: {input_tokens} input, {output_tokens} output")
        
        print(colorize(f"✅ Test {test_name} passed!", Colors.GREEN))
        return True
        
    except asyncio.TimeoutError:
        print(colorize(f"❌ Test {test_name} timed out", Colors.RED))
        return False
    except Exception as e:
        print(colorize(f"❌ Error in test {test_name}: {str(e)}", Colors.RED))
        import traceback
        traceback.print_exc()
        return False

# ================= STREAMING TESTS =================

class StreamStats:
    """Track statistics about a streaming response."""
    
    def __init__(self):
        self.event_types = set()
        self.event_counts = {}
        self.first_event_time = None
        self.last_event_time = None
        self.total_chunks = 0
        self.events = []
        self.text_content = ""
        self.content_blocks = {}
        self.has_tool_use = False
        self.has_error = False
        self.error_message = ""
        self.text_content_by_block = {}
        self.ping_count = 0
        
    def add_event(self, event_data):
        """Track information about each received event."""
        now = datetime.now()
        if self.first_event_time is None:
            self.first_event_time = now
        self.last_event_time = now
        
        self.total_chunks += 1
        
        # Record event type and increment count
        if "type" in event_data:
            event_type = event_data["type"]
            self.event_types.add(event_type)
            self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1
            
            # Track ping events separately
            if event_type == "ping":
                self.ping_count += 1
            
            # Track specific event data
            elif event_type == "content_block_start":
                block_idx = event_data.get("index")
                content_block = event_data.get("content_block", {})
                if content_block.get("type") == "tool_use":
                    self.has_tool_use = True
                self.content_blocks[block_idx] = content_block
                self.text_content_by_block[block_idx] = ""
                
            elif event_type == "content_block_delta":
                block_idx = event_data.get("index")
                delta = event_data.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    self.text_content += text
                    if block_idx in self.text_content_by_block:
                        self.text_content_by_block[block_idx] += text
                        
        # Keep track of all events for debugging
        self.events.append(event_data)
                
    def get_duration(self):
        """Calculate the total duration of the stream in seconds."""
        if self.first_event_time is None or self.last_event_time is None:
            return 0
        return (self.last_event_time - self.first_event_time).total_seconds()
        
    def summarize(self):
        """Print a summary of the stream statistics."""
        print(f"Total chunks: {self.total_chunks}")
        print(f"Unique event types: {sorted(list(self.event_types))}")
        print(f"Event counts: {json.dumps(self.event_counts, indent=2)}")
        print(f"Duration: {self.get_duration():.2f} seconds")
        print(f"Ping events: {self.ping_count}")
        print(f"Has tool use: {self.has_tool_use}")
        
        # Print the first few lines of content
        if self.text_content:
            max_preview_lines = 5
            text_preview = "\n".join(self.text_content.strip().split("\n")[:max_preview_lines])
            print(f"Text preview:\n{text_preview}")
        else:
            print("No text content extracted")
            
        if self.has_error:
            print(f"Error: {self.error_message}")

async def test_streaming(test_name, request_data):
    """Run a streaming test with the given request data."""
    print(colorize(f"\n{'='*20} RUNNING STREAMING TEST: {test_name} {'='*20, Colors.BOLD))
    
    # Log the request data
    log_data = {k: v for k, v in request_data.items() if k != 'messages'}
    log_data['message_count'] = len(request_data.get('messages', []))
    print(f"Request config: {json.dumps(log_data, indent=2)}")
    
    # Ensure streaming is enabled
    stream_data = request_data.copy()
    stream_data["stream"] = True
    
    check_tools = "tools" in request_data
    stats = StreamStats()
    
    try:
        async with httpx.AsyncClient() as client:
            start_time = time.time()
            async with client.stream("POST", PROXY_API_URL, json=stream_data, headers=proxy_headers, timeout=120) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    print(colorize(f"❌ Streaming request failed: {response.status_code}", Colors.RED))
                    print(error_text.decode('utf-8'))
                    return False
                
                print(colorize("📡 Stream connected, receiving events...", Colors.BLUE))
                
                # Process each chunk
                buffer = ""
                async for chunk in response.aiter_text():
                    if not chunk.strip():
                        continue
                    
                    # Handle multiple events in one chunk
                    buffer += chunk
                    events = buffer.split("\n\n")
                    
                    # Process all complete events
                    for event_text in events[:-1]:
                        if not event_text.strip():
                            continue
                        
                        # Parse server-sent event format
                        if "data: " in event_text:
                            data_parts = []
                            for line in event_text.split("\n"):
                                if line.startswith("data: "):
                                    data_part = line[len("data: "):]
                                    if data_part == "[DONE]":
                                        break
                                    data_parts.append(data_part)
                            
                            if data_parts:
                                try:
                                    event_data = json.loads("".join(data_parts))
                                    stats.add_event(event_data)
                                except json.JSONDecodeError as e:
                                    print(f"Warning: Error parsing event: {e}")
                    
                    # Keep the last (potentially incomplete) event
                    buffer = events[-1] if events else ""
                
                # Process any remaining events
                if buffer.strip():
                    lines = buffer.strip().split("\n")
                    data_lines = [line[len("data: "):] for line in lines if line.startswith("data: ")]
                    if data_lines and data_lines[0] != "[DONE]":
                        try:
                            event_data = json.loads("".join(data_lines))
                            stats.add_event(event_data)
                        except:
                            pass
                
                elapsed = time.time() - start_time
                print(colorize(f"📡 Stream completed in {elapsed:.2f} seconds", Colors.BLUE))
        
        # Analyze results
        print("\n--- Stream Statistics ---")
        stats.summarize()
        
        # Validate streaming response
        required_missing = REQUIRED_EVENT_TYPES - stats.event_types
        if required_missing:
            print(colorize(f"⚠️ Missing required event types: {required_missing}", Colors.YELLOW))
        else:
            print(colorize("✅ All required event types present", Colors.GREEN))
        
        # Check for ping events (timeout protection)
        if stats.ping_count > 0:
            print(colorize(f"✅ Received {stats.ping_count} ping events (good for timeout protection)", Colors.GREEN))
        
        # Check content
        if stats.text_content or stats.has_tool_use:
            if check_tools and stats.has_tool_use:
                print(colorize("✅ Tool use detected in streaming response", Colors.GREEN))
            elif stats.text_content:
                print(colorize("✅ Text content received", Colors.GREEN))
            
            print(colorize(f"✅ Streaming test {test_name} passed!", Colors.GREEN))
            return True
        else:
            print(colorize(f"❌ No content received in streaming test {test_name}", Colors.RED))
            return False
        
    except asyncio.TimeoutError:
        print(colorize(f"❌ Streaming test {test_name} timed out", Colors.RED))
        return False
    except Exception as e:
        print(colorize(f"❌ Error in streaming test {test_name}: {str(e)}", Colors.RED))
        import traceback
        traceback.print_exc()
        return False

# ================= MAIN TEST RUNNER =================

async def run_tests(args):
    """Run all tests based on command-line arguments."""
    results = {}
    total_tests = 0
    
    # Health and diagnostic tests
    if args.health_only or not any([args.simple, args.tools_only, args.streaming_only]):
        print(colorize("\n\n=========== RUNNING HEALTH & DIAGNOSTIC TESTS ===========\n", Colors.BOLD))
        
        results["health_check"] = await test_health_endpoint()
        results["connection_test"] = await test_connection_endpoint()
        results["token_counting"] = await test_token_counting()
        total_tests += 3
        
        if args.health_only:
            # Print summary for health-only tests
            passed = sum(1 for v in results.values() if v)
            print(colorize(f"\n\n=========== HEALTH TEST SUMMARY ===========", Colors.BOLD))
            print(f"Health & Diagnostic: {passed}/{total_tests} tests passed")
            return passed == total_tests
    
    # Filter tests based on arguments
    filtered_scenarios = {}
    for test_name, test_data in TEST_SCENARIOS.items():
        # Skip based on provider filters
        model = test_data.get("model", "")
        if args.gemini_only and not (model.startswith("gemini/") or "haiku" in model or "sonnet" in model):
            continue
        if args.vertex_only and not model.startswith("vertex_ai/"):
            continue
        
        # Skip based on test type filters
        if args.simple and ("tools" in test_data or test_data.get("stream")):
            continue
        if args.tools_only and "tools" not in test_data:
            continue
        if args.streaming_only and not test_data.get("stream"):
            continue
        if args.no_streaming and test_data.get("stream"):
            continue
            
        filtered_scenarios[test_name] = test_data
    
    # Run non-streaming tests
    if not args.streaming_only:
        print(colorize("\n\n=========== RUNNING NON-STREAMING TESTS ===========\n", Colors.BOLD))
        
        for test_name, test_data in filtered_scenarios.items():
            if test_data.get("stream"):
                continue
                
            check_tools = "tools" in test_data
            result = await test_request(test_name, test_data, check_tools=check_tools)
            results[test_name] = result
            total_tests += 1
    
    # Run streaming tests
    if not args.no_streaming:
        print(colorize("\n\n=========== RUNNING STREAMING TESTS ===========\n", Colors.BOLD))
        
        for test_name, test_data in filtered_scenarios.items():
            if not test_data.get("stream"):
                continue
                
            result = await test_streaming(test_name, test_data)
            results[f"{test_name}_streaming"] = result
            total_tests += 1
    
    # Print comprehensive summary
    print(colorize(f"\n\n=========== COMPREHENSIVE TEST SUMMARY ===========", Colors.BOLD))
    
    # Group results by category
    health_tests = {k: v for k, v in results.items() if k in ["health_check", "connection_test", "token_counting"]}
    basic_tests = {k: v for k, v in results.items() if not k.endswith("_streaming") and k not in health_tests}
    streaming_tests = {k: v for k, v in results.items() if k.endswith("_streaming")}
    
    categories = [
        ("Health & Diagnostic", health_tests),
        ("Basic Functionality", basic_tests), 
        ("Streaming", streaming_tests)
    ]
    
    overall_passed = 0
    overall_total = 0
    
    for category_name, category_tests in categories:
        if not category_tests:
            continue
            
        passed = sum(1 for v in category_tests.values() if v)
        total = len(category_tests)
        overall_passed += passed
        overall_total += total
        
        print(f"\n{category_name}: {passed}/{total} tests passed")
        for test, result in category_tests.items():
            status = colorize("✅ PASS", Colors.GREEN) if result else colorize("❌ FAIL", Colors.RED)
            print(f"  {test}: {status}")
    
    success_rate = (overall_passed / overall_total * 100) if overall_total > 0 else 0
    print(f"\n{colorize('OVERALL', Colors.BOLD)}: {overall_passed}/{overall_total} tests passed ({success_rate:.1f}%)")
    
    if overall_passed == overall_total:
        print(colorize("\n🎉 All tests passed! Your proxy is working perfectly.", Colors.GREEN))
        return True
    else:
        failed_count = overall_total - overall_passed
        print(colorize(f"\n⚠️ {failed_count} test(s) failed. Check the detailed output above.", Colors.YELLOW))
        return False

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test suite for Gemini-Vertex AI Proxy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests.py                    # Run all tests
  python tests.py --simple           # Run only simple tests (no tools)
  python tests.py --tools-only       # Run only tool-related tests
  python tests.py --no-streaming     # Skip all streaming tests
  python tests.py --streaming-only   # Run only streaming tests
  python tests.py --gemini-only      # Test only Gemini models
  python tests.py --vertex-only      # Test only Vertex AI models
  python tests.py --health-only      # Test only health/diagnostic endpoints
        """
    )
    
    parser.add_argument("--no-streaming", action="store_true", help="Skip streaming tests")
    parser.add_argument("--streaming-only", action="store_true", help="Only run streaming tests")
    parser.add_argument("--simple", action="store_true", help="Only run simple tests (no tools)")
    parser.add_argument("--tools-only", action="store_true", help="Only run tool tests")
    parser.add_argument("--gemini-only", action="store_true", help="Test only Gemini models")
    parser.add_argument("--vertex-only", action="store_true", help="Test only Vertex AI models")
    parser.add_argument("--health-only", action="store_true", help="Test only health/diagnostic endpoints")
    
    args = parser.parse_args()
    
    # Validate conflicting arguments
    if args.streaming_only and args.no_streaming:
        print(colorize("Error: Cannot use --streaming-only and --no-streaming together", Colors.RED))
        sys.exit(1)
    
    if args.gemini_only and args.vertex_only:
        print(colorize("Error: Cannot use --gemini-only and --vertex-only together", Colors.RED))
        sys.exit(1)
    
    print(colorize("🧪 Starting Gemini-Vertex AI Proxy Test Suite", Colors.BOLD))
    print(f"Proxy URL: {PROXY_BASE_URL}")
    
    # Run tests
    success = await run_tests(args)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())
