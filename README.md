# Gemini & Vertex AI for Claude Code: An Anthropic-Compatible Proxy

This server acts as a bridge, enabling you to use **Claude Code** with Google's powerful **Gemini** and **Vertex AI** models. It translates API requests and responses between the Anthropic format (used by Claude Code) and Google's formats (via LiteLLM), allowing seamless integration with both Google AI Studio and Google Cloud Vertex AI.

![Claude Code with Gemini Proxy](image.png)

## Features

- **Claude Code Compatibility**: Directly use the Claude Code CLI with Google Gemini and Vertex AI models.
- **Dual Provider Support**: Choose between Google AI Studio (Gemini) or Google Cloud Vertex AI based on your needs.
- **Seamless Model Mapping**: Intelligently maps Claude Code model requests (e.g., `haiku`, `sonnet`, `opus` aliases) to your chosen Google models.
- **Enterprise Ready**: Full Vertex AI support with service account authentication and project isolation.
- **LiteLLM Integration**: Leverages LiteLLM for robust and flexible interaction with Google's APIs.
- **Enhanced Streaming Support**: Handles streaming responses with robust error recovery, timeout handling, and keep-alive pings.
- **Complete Tool Use for Claude Code**: Translates Claude Code's tool usage (function calling) to and from Google's format, with robust handling of tool results.
- **Advanced Error Handling**: Provides specific and actionable error messages for common API issues with automatic fallback strategies.
- **Resilient Architecture**: Gracefully handles API instability with smart retry logic and fallback to non-streaming modes.
- **Diagnostic Endpoints**: Includes `/health` and `/test-connection` for easier troubleshooting of your setup.
- **Token Counting**: Offers a `/v1/messages/count_tokens` endpoint compatible with Claude Code.

## Recent Improvements (v2.7.0)

### 🔵 Vertex AI Enterprise Support
- **Full Vertex AI Integration**: Complete support for Google Cloud Vertex AI with project isolation
- **Service Account Authentication**: Support for both ADC and service account credentials
- **Multi-Region Support**: Configure your preferred Vertex AI location (us-central1, europe-west1, etc.)
- **Provider Selection**: Choose between Google AI Studio (gemini) or Vertex AI (vertex) as your preferred provider

### ⏰ Enhanced Timeout Handling
- **Keep-Alive Pings**: Automatic ping messages every 30 seconds during streaming to prevent timeouts
- **Extended Timeouts**: 10-minute request timeouts for long conversations and complex tasks
- **Connection Stability**: Improved handling of long-running requests and network interruptions
- **Streaming Headers**: Enhanced headers to prevent proxy buffering issues

### 🛡️ Enhanced Error Resilience
- **Malformed Chunk Recovery**: Automatically detects and handles malformed JSON chunks from streaming
- **Smart Retry Logic**: Exponential backoff with configurable retry limits for streaming errors
- **Graceful Fallback**: Seamlessly switches to non-streaming mode when streaming fails
- **Buffer Management**: Intelligent chunk buffering and reconstruction for incomplete JSON
- **Connection Stability**: Handles API 500 Internal Server Errors with automatic retry

### 📊 Improved Monitoring
- **Multi-Provider Status**: Health checks for both Gemini and Vertex AI configurations
- **Detailed Error Classification**: Specific guidance for different types of API errors
- **Enhanced Logging**: Comprehensive error tracking with provider-specific indicators
- **Real-time Diagnostics**: Better connection testing for both providers

## Prerequisites

- A Google AI Studio API key (for Gemini) **OR** Google Cloud Project with Vertex AI enabled
- Python 3.8+
- Claude Code CLI installed (e.g., `npm install -g @anthropic-ai/claude-code`)

## Provider Options

### Option 1: Google AI Studio (Gemini) - Recommended for Individual Use
- **Setup**: Just need a Gemini API key from [Google AI Studio](https://aistudio.google.com)
- **Cost**: Pay-per-use pricing
- **Models**: Latest Gemini models with immediate availability
- **Ideal for**: Individual developers, personal projects, quick setup

### Option 2: Google Cloud Vertex AI - Recommended for Enterprise
- **Setup**: Requires Google Cloud Project with Vertex AI API enabled
- **Cost**: Enterprise pricing with committed use discounts available
- **Models**: Enterprise-grade Gemini models with enhanced security
- **Features**: Project isolation, VPC networking, audit logging, compliance certifications
- **Ideal for**: Businesses, teams, production workloads

## Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/coffeegrind123/gemini-code.git # Or your fork
    cd gemini-code
    ```

2.  **Create and activate a virtual environment** (recommended):
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**:
    Copy the example environment file:
    ```bash
    cp .env.example .env
    ```

### For Google AI Studio (Gemini) Setup:
Edit `.env` and add your Gemini API key:
```dotenv
# Required: Your Google AI Studio API key
GEMINI_API_KEY="your-google-ai-studio-key"

# Use Gemini as default provider
PREFERRED_PROVIDER="gemini"
```

### For Vertex AI Setup:
Edit `.env` and configure Vertex AI:
```dotenv
# Required: Your Google AI Studio API key (still needed as fallback)
GEMINI_API_KEY="your-google-ai-studio-key"

# Vertex AI Configuration
VERTEX_PROJECT_ID="your-gcp-project-id"
VERTEX_LOCATION="us-central1"
GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"  # Optional if using ADC

# Use Vertex AI as preferred provider
PREFERRED_PROVIDER="vertex"
```

**Vertex AI Authentication Options:**
- **Service Account**: Set `GOOGLE_APPLICATION_CREDENTIALS` to path of service account JSON
- **Application Default Credentials (ADC)**: Run `gcloud auth application-default login` and leave `GOOGLE_APPLICATION_CREDENTIALS` empty

### Additional Configuration Options:
```dotenv
# Optional: Model mappings for Claude Code aliases
BIG_MODEL="gemini-1.5-pro-latest"    # For 'sonnet' or 'opus' requests
SMALL_MODEL="gemini-1.5-flash-latest" # For 'haiku' requests

# Optional: Server settings
HOST="0.0.0.0"
PORT="8082"
LOG_LEVEL="WARNING"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Optional: Performance and reliability settings
MAX_TOKENS_LIMIT="8192"           # Max tokens for responses
REQUEST_TIMEOUT="90"              # Request timeout in seconds
MAX_RETRIES="2"                   # LiteLLM retries to API
MAX_STREAMING_RETRIES="12"        # Streaming-specific retry attempts

# Optional: Streaming control (use if experiencing issues)
FORCE_DISABLE_STREAMING="false"     # Disable streaming globally
EMERGENCY_DISABLE_STREAMING="false" # Emergency streaming disable
```

5.  **Run the server**:
    ```bash
    python server.py
    ```
    For development with auto-reload:
    ```bash
    uvicorn server:app --host 0.0.0.0 --port 8082 --reload
    ```
    View all startup options:
    ```bash
    python server.py --help
    ```

## Usage with Claude Code

1.  **Start the Proxy Server**: Ensure the proxy server is running (see step 5 above).

2.  **Configure Claude Code to Use the Proxy**:
    Set the `ANTHROPIC_BASE_URL` environment variable when running Claude Code:
    ```bash
    ANTHROPIC_BASE_URL=http://localhost:8082 claude
    ```
    Replace `localhost:8082` if your proxy is running on a different host or port.

3.  **Utilize `CLAUDE.md` for Optimal Performance**:
    - Copy the included `CLAUDE.md` file to your project directory
    - Claude Code automatically reads this file and incorporates its guidance
    - Essential for optimal Google model performance with Claude Code workflows

## How It Works: Powering Claude Code with Google Models

1.  **Claude Code Request**: You issue a command or prompt in the Claude Code CLI.
2.  **Anthropic Format**: Claude Code sends an API request (in Anthropic's Messages API format) to the proxy server.
3.  **Proxy Translation (Anthropic to Google)**: The proxy server:
    *   Receives the Anthropic-formatted request
    *   Maps Claude model aliases to your configured Google models
    *   Translates message structure, content blocks, and tool definitions
    *   Routes to either Gemini (Google AI Studio) or Vertex AI based on your preference
4.  **Google API Interaction**: LiteLLM sends the request to your chosen Google service
5.  **Google Response**: The Google model processes the request and returns the response
6.  **Proxy Translation (Google to Anthropic)**: The proxy server:
    *   Handles streaming responses with timeout protection and keep-alive pings
    *   Recovers from malformed chunks and API errors automatically
    *   Converts Google's output back to Anthropic Messages API format
7.  **Response to Claude Code**: Claude Code receives the properly formatted response

## Model Mapping for Claude Code

The proxy intelligently maps Claude Code requests to Google models:

- **"haiku"** requests → `SMALL_MODEL` (default: `gemini-1.5-flash-latest`)
- **"sonnet"** or **"opus"** requests → `BIG_MODEL` (default: `gemini-1.5-pro-latest`)
- **Direct model names** (e.g., `gemini/gemini-2.0-flash`) → Used directly
- **Provider routing** based on `PREFERRED_PROVIDER` setting

### Available Models by Provider:

**Gemini (Google AI Studio):**
- `gemini-1.5-pro-latest`
- `gemini-1.5-flash-latest`
- `gemini-2.0-flash-exp`
- `gemini-exp-1206`
- And more...

**Vertex AI:**
- `gemini-2.5-pro-preview-03-25`
- `gemini-2.0-flash`
- `gemini-1.5-pro-preview-0514`
- `gemini-1.5-flash-preview-0514`
- And more...

## Endpoints

- `POST /v1/messages`: Primary endpoint for Claude Code messages, fully compatible with Anthropic Messages API
- `POST /v1/messages/count_tokens`: Token counting using Google's tokenization
- `GET /health`: Comprehensive health status for both Gemini and Vertex AI configurations
- `GET /test-connection`: Tests connectivity to all configured providers
- `GET /`: Configuration summary and available endpoints

## Error Handling & Troubleshooting

### Provider-Specific Issues

**Gemini (Google AI Studio) Issues:**
- **API Key**: Verify your key at [Google AI Studio](https://aistudio.google.com)
- **Rate Limits**: Check quota limits in Google AI Studio console
- **Model Access**: Ensure you have access to the requested models

**Vertex AI Issues:**
- **Project Setup**: Ensure Vertex AI API is enabled in your Google Cloud Project
- **Authentication**: Verify service account permissions or ADC setup
- **Location**: Confirm your `VERTEX_LOCATION` supports the requested models
- **Billing**: Ensure your Google Cloud Project has billing enabled

### Common Issues and Solutions

**Streaming Errors (malformed chunks):**
- The proxy automatically handles malformed JSON chunks
- Increase `MAX_STREAMING_RETRIES` for more resilient streaming
- Set `FORCE_DISABLE_STREAMING=true` as a temporary fix

**Connection Timeouts:**
- The proxy now includes automatic keep-alive pings every 30 seconds
- Increase `REQUEST_TIMEOUT` for very long conversations
- Check firewall settings for long-lived connections

**Authentication Errors:**
- **Gemini**: Verify `GEMINI_API_KEY` format (should start with `AIza`)
- **Vertex AI**: Check service account permissions or run `gcloud auth application-default login`

### Emergency Mode

If you experience persistent issues:
```bash
# Disable streaming temporarily
export EMERGENCY_DISABLE_STREAMING=true

# Switch to fallback provider
export PREFERRED_PROVIDER=gemini  # or vertex
```

### Diagnostic Commands

```bash
# Test all configured providers
curl http://localhost:8082/test-connection

# Check health status
curl http://localhost:8082/health

# View configuration
curl http://localhost:8082/
```

## Logging

The server provides detailed logs with provider-specific indicators:

- 🟡 **Gemini operations** (Google AI Studio)
- 🔵 **Vertex AI operations** (Google Cloud)
- ⚠️ **Error recovery** and fallback notifications
- ⏰ **Timeout handling** and keep-alive management

Adjust verbosity with `LOG_LEVEL`:
- `DEBUG`: Detailed request/response logging and error recovery steps
- `INFO`: General operation logging with provider routing
- `WARNING`: Error recovery and fallback notifications (recommended)
- `ERROR`: Only errors and failures

## The `CLAUDE.md` File: Optimizing Google Models for Claude Code

The included `CLAUDE.md` file is essential for optimal performance:

**Purpose:**
- Tailors Google models to Claude Code's specific requirements
- Improves tool reliability and code generation quality
- Provides context about development environment and standards
- Bridges differences between Anthropic and Google model behaviors

**Usage:**
Claude Code automatically reads `CLAUDE.md` from your project directory and includes its content in the system prompt. Always copy this file to your project root for best results.

## Performance Tips

### Model Selection
- **Fast responses**: Use `gemini-1.5-flash-latest` 
- **Complex tasks**: Use `gemini-1.5-pro-latest`
- **Latest features**: Try `gemini-2.0-flash-exp` or `gemini-exp-1206`

### Provider Selection
- **Individual use**: Google AI Studio (Gemini) for simplicity
- **Enterprise use**: Vertex AI for enhanced security and compliance
- **Hybrid approach**: Use both providers with intelligent fallback

### Performance Optimization
- **Streaming**: Keep enabled for better interactivity (automatic error recovery included)
- **Timeouts**: Increase `REQUEST_TIMEOUT` for complex requests
- **Retries**: Adjust `MAX_STREAMING_RETRIES` based on network stability
- **Keep-alive**: Automatic 30-second pings prevent timeout issues

## Contributing

Contributions, issues, and feature requests are welcome! 

Areas where contributions are especially valuable:
- Additional Google model support
- Enhanced Vertex AI features
- Performance optimizations
- Error recovery improvements
- Documentation enhancements

## Thanks

This project builds upon the foundational work of [claude-code-proxy by @1rgs](https://github.com/1rgs/claude-code-proxy). Special thanks to the community for testing and feedback on multi-provider support and timeout handling improvements.
