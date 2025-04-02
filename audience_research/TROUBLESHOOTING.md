# Troubleshooting Guide for Audience Research Agent

## Tavily Search Issues

### API Key Not Found Error

**Error Message:** `TavilyError: Tavily API key not found. Please ensure the TAVILY_API_KEY environment variable is set.`

**Solution:**
1. Create an account at [Tavily](https://tavily.com/) to get your API key
2. Set the environment variable:
   ```bash
   export TAVILY_API_KEY=your_tavily_api_key_here
   ```
3. Alternatively, create a `.env` file in the root directory of the project with:
   ```
   TAVILY_API_KEY=your_tavily_api_key_here
   ```
4. You can also specify the API key directly in the config.yml file:
   ```yaml
   internet_search:
     _type: tavily_internet_search
     description: "Search the web for market information and competitor data"
     max_results: 5
     api_key: "your_tavily_api_key_here"
   ```

### SSL Certificate Errors

**Error Message:** `SSLCertVerificationError: certificate verify failed`

**Solution:**
If you're getting SSL certificate verification errors when connecting to external APIs, try:

1. Update your Python SSL certificates:
   ```bash
   pip install --upgrade certifi
   ```
2. Set the proper SSL certificate path:
   ```bash
   export SSL_CERT_FILE=/path/to/cacert.pem
   ```

## Input Format Errors

### JSON Parsing Errors

**Error Message:** `Unable to parse structured tool input from Action Input. Using Action Input as is. Parsing error: Expecting value: line 1 column 1 (char 0)`

**Solution:**
Ensure your tool inputs are properly formatted as JSON objects:

1. For internet searches:
   ```
   Action: internet_search
   Action Input: {"query": "your search query here"}
   ```

2. For other tools, use the proper JSON structure:
   ```
   Action: tool_name
   Action Input: {"input_data": {"key": "value"}}
   ```

## LLM Connection Issues

### NVIDIA API Connection Errors

**Error Message:** `Cannot connect to host api.nvidia.com:443 ssl:True [SSLCertVerificationError: certificate verify failed: Hostname mismatch]`

**Solution:**
1. Ensure your NVIDIA API key is correctly set:
   ```bash
   export NVIDIA_API_KEY=your_nvidia_api_key_here
   ```
2. Check your internet connection and firewall settings
3. Try using an OpenAI LLM as a fallback by setting in the config.yml:
   ```yaml
   workflow:
     llm_name: openai_llm  # Use OpenAI instead of NIM
   ```

## Recursion Limit Errors

**Error Message:** `Recursion limit of 32 reached without hitting a stop condition`

**Solution:**
Increase the recursion limit in the config.yml file:
```yaml
workflow:
  recursion_limit: 50  # Increase this value
```

## Installation Issues

If you encounter issues with package installations:

1. Ensure you have the correct extras installed:
   ```bash
   pip install -e 'agentiq[langchain]'
   pip install -e 'audience_research[langchain]'
   ```

2. If you see "extras not found" warnings, update your pip:
   ```bash
   pip install --upgrade pip
   ```

3. Check for compatibility issues between packages:
   ```bash
   pip check
   ```

## Agent Not Using Tools Correctly

If the agent isn't using tools correctly or is getting stuck in loops:

1. Increase the max_retries and retry_parsing_errors in config.yml:
   ```yaml
   workflow:
     max_retries: 5
     retry_parsing_errors: true
   ```

2. Add more detailed additional_instructions to guide the agent:
   ```yaml
   additional_instructions: |
     Always format your inputs as valid JSON...
   ``` 