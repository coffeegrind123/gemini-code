# Test Suite for Gemini-Vertex AI Proxy

This comprehensive test suite validates all functionality of your Gemini-Vertex AI proxy, including model mapping, provider switching, streaming, tool use, and error handling.

## Quick Start

1. **Install test dependencies:**
   ```bash
   pip install -r requirements-test.txt
   ```

2. **Start your proxy:**
   ```bash
   python server.py
   ```

3. **Run all tests:**
   ```bash
   python tests.py
   ```

## Test Categories

### 🏥 Health & Diagnostic Tests
- **Health Check**: Tests `/health` endpoint and provider configuration
- **Connection Test**: Tests `/test-connection` endpoint for all providers
- **Token Counting**: Tests `/v1/messages/count_tokens` endpoint

### 🧪 Basic Functionality Tests
- **Model Mapping**: Tests `haiku` → small model, `sonnet` → big model
- **Direct Models**: Tests direct `gemini/` and `vertex_ai/` model usage
- **System Prompts**: Tests system message handling
- **Parameters**: Tests temperature, top_p, top_k, stop_sequences
- **Content Blocks**: Tests complex message structures

### 🔧 Tool Use Tests
- **Basic Tools**: Single tool usage with simple schemas
- **Multiple Tools**: Multi-tool scenarios with complex schemas
- **Tool Choice**: Tests specific tool selection
- **Multi-turn**: Tool use across conversation turns

### 📡 Streaming Tests
- **Basic Streaming**: Simple text generation with streaming
- **Tool Streaming**: Tool use with streaming responses
- **Timeout Protection**: Tests keep-alive ping functionality
- **Provider Streaming**: Tests streaming on both Gemini and Vertex AI

### ⚡ Advanced Features
- **Thinking Mode**: Tests thinking configuration
- **Error Handling**: Tests edge cases and error recovery
- **Long Conversations**: Tests timeout handling

## Test Options

### Run Specific Test Categories
```bash
# Only health/diagnostic tests
python tests.py --health-only

# Only simple text tests (no tools)
python tests.py --simple

# Only tool-related tests
python tests.py --tools-only

# Only streaming tests
python tests.py --streaming-only

# Skip streaming tests
python tests.py --no-streaming
```

### Test Specific Providers
```bash
# Test only Gemini models
python tests.py --gemini-only

# Test only Vertex AI models
python tests.py --vertex-only
```

### Combined Options
```bash
# Test only Gemini streaming
python tests.py --gemini-only --streaming-only

# Test Vertex AI without streaming
python tests.py --vertex-only --no-streaming
```

## Understanding Test Output

### ✅ Success Indicators
- **Green checkmarks**: Tests passed
- **Model mapping confirmations**: `haiku` → `gemini-1.5-flash-latest`
- **Tool use detection**: Tools called correctly
- **Streaming events**: All required SSE events received
- **Ping events**: Keep-alive functionality working

### ⚠️ Warnings (Not Failures)
- **Missing tool use**: When tools available but not used
- **Missing ping events**: For short streams
- **Provider fallbacks**: When preferred provider unavailable

### ❌ Failure Indicators
- **Red X marks**: Tests failed
- **Missing required fields**: Response structure issues
- **Timeout errors**: Connection or processing issues
- **Invalid responses**: Malformed or empty responses

## Test Configuration

### Environment Variables
```bash
# Proxy configuration (optional)
PROXY_BASE_URL="http://localhost:8082"  # Default: http://localhost:8082

# Provider configuration (in your main .env)
GEMINI_API_KEY="your-key"
VERTEX_PROJECT_ID="your-project"
PREFERRED_PROVIDER="gemini"  # or "vertex"
```

### Customizing Tests

You can modify test scenarios in `tests.py`:

```python
# Add custom test scenario
TEST_SCENARIOS["custom_test"] = {
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 500,
    "messages": [
        {"role": "user", "content": "Your custom test prompt"}
    ],
    "tools": [your_custom_tool],  # Optional
    "stream": True  # Optional
}
```

## Troubleshooting

### Common Issues

**Tests failing with 503 errors:**
- Check that your proxy server is running
- Verify your API keys are configured correctly
- Test individual providers with `--gemini-only` or `--vertex-only`

**Streaming tests timing out:**
- Your proxy may have streaming disabled
- Check `FORCE_DISABLE_STREAMING` and `EMERGENCY_DISABLE_STREAMING` settings
- Try `--no-streaming` to test non-streaming functionality

**Vertex AI tests failing:**
- Ensure `VERTEX_PROJECT_ID` is set
- Verify Vertex AI API is enabled in Google Cloud
- Check authentication (ADC or service account)

**Tool use tests failing:**
- Check if your models support function calling
- Verify tool schemas are valid
- Some models may prefer text responses over tool calls

### Debug Mode

For detailed debugging, modify the test script:

```python
# Add this near the top for more verbose output
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Provider Status Check

Before running full tests, check provider status:
```bash
curl http://localhost:8082/health
curl http://localhost:8082/test-connection
```

## Test Results Interpretation

### Excellent (90-100% pass rate)
Your proxy is working perfectly across all providers and features.

### Good (70-89% pass rate)
Most functionality works, some edge cases or specific providers may have issues.

### Needs Attention (50-69% pass rate)
Core functionality works but significant issues exist with specific features.

### Critical Issues (<50% pass rate)
Major configuration or connectivity problems need immediate attention.

## Contributing Test Cases

To add new test scenarios:

1. **Add to `TEST_SCENARIOS`** with descriptive name
2. **Include appropriate tags** (tools, streaming, etc.)
3. **Test edge cases** and error conditions
4. **Document expected behavior** in comments

Example:
```python
"your_test_name": {
    "model": "claude-3-5-haiku-20241022",
    "max_tokens": 300,
    "messages": [...],
    "tools": [...],  # If testing tools
    "stream": True,  # If testing streaming
    # Add comments about what this tests
}
```
