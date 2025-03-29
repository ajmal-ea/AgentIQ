<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http:/www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

![NVIDIA AgentIQ](./docs/source/_static/agentiq_banner.png "AgentIQ banner image")

# NVIDIA AgentIQ

AgentIQ is a flexible library designed to seamlessly integrate your enterprise agents—regardless of framework—with various data sources and tools. By treating agents, tools, and agentic workflows as simple function calls, AgentIQ enables true composability: build once and reuse anywhere.

## Key Features

- [**Framework Agnostic:**](https://docs.nvidia.com/agentiq/latest/concepts/plugins.html) Works with any agentic framework, so you can use your current technology stack without replatforming.
- [**Reusability:**](https://docs.nvidia.com/agentiq/latest/guides/sharing-workflows-and-tools.html) Every agent, tool, or workflow can be combined and repurposed, allowing developers to leverage existing work in new scenarios.
- [**Rapid Development:**](https://docs.nvidia.com/agentiq/latest/guides/create-customize-workflows.html) Start with a pre-built agent, tool, or workflow, and customize it to your needs.
- [**Profiling:**](https://docs.nvidia.com/agentiq/latest/guides/profiler.html) Profile entire workflows down to the tool and agent level, track input/output tokens and timings, and identify bottlenecks.
- [**Observability:**](https://docs.nvidia.com/agentiq/latest/guides/observe-workflow-with-phoenix.html) Monitor and debug your workflows with any OpenTelemetry-compatible observability tool.
- [**Evaluation System:**](https://docs.nvidia.com/agentiq/latest/guides/evaluate.html) Validate and maintain accuracy of agentic workflows with built-in evaluation tools.
- [**User Interface:**](https://docs.nvidia.com/agentiq/latest/guides/using-agentiq-ui-and-server.html) Use the AgentIQ UI chat interface to interact with your agents, visualize output, and debug workflows.
- [**MCP Compatibility**](https://docs.nvidia.com/agentiq/latest/components/mcp.html) Compatible with Model Context Protocol (MCP), allowing tools served by MCP Servers to be used as AgentIQ functions.

With AgentIQ, you can move quickly, experiment freely, and ensure reliability across all your agent-driven projects.

## Links

 * [Documentation](https://docs.nvidia.com/agentiq/latest/index.html): Explore the full documentation for AgentIQ.
 * [About AgentIQ](https://docs.nvidia.com/agentiq/latest/intro/why-agentiq.html): Learn more about the benefits of using AgentIQ.
 * [Get Started Guide](https://docs.nvidia.com/agentiq/latest/intro/get-started.html): Set up your environment and start building with AgentIQ.
 * [Examples](https://github.com/NVIDIA/AgentIQ/tree/main/examples#readme): Explore examples of AgentIQ workflows.
 * [Create and Customize AgentIQ Workflows](https://docs.nvidia.com/agentiq/latest/guides/create-customize-workflows.html): Learn how to create and customize AgentIQ workflows.
 * [Evaluate with AgentIQ](https://docs.nvidia.com/agentiq/latest/guides/evaluate.html): Learn how to evaluate your AgentIQ workflows.
 * [Troubleshooting](https://docs.nvidia.com/agentiq/latest/troubleshooting.html): Get help with common issues.


## Get Started

### Prerequisites

Before you begin using AgentIQ, ensure that you meet the following software prerequisites.

- Install [Git](https://git-scm.com/)
- Install [Git Large File Storage](https://git-lfs.github.com/) (LFS)
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Install From Source

1. Clone the AgentIQ repository to your local machine.
    ```bash
    git clone git@github.com:NVIDIA/AgentIQ.git agentiq
    cd agentiq
    ```

2. Initialize, fetch, and update submodules in the Git repository.
    ```bash
    git submodule update --init --recursive
    ```

3. Fetch the data sets by downloading the LFS files.
    ```bash
    git lfs install
    git lfs fetch
    git lfs pull
    ```

4. Create a Python environment.
    ```bash
    uv venv --seed .venv
    source .venv/bin/activate
    ```

5. Install the AgentIQ library.
    To install the AgentIQ library along with all of the optional dependencies. Including developer tools (`--all-groups`) and all of the dependencies needed for profiling and plugins (`--all-extras`) in the source repository, run the following:
    ```bash
    uv sync --all-groups --all-extras
    ```

    Alternatively to install just the core AgentIQ without any plugins, run the following:
    ```bash
    uv sync
    ```

    At this point individual plugins, which are located under the `packages` directory, can be installed with the following command `uv pip install -e '.[<plugin_name>]'`.
    For example, to install the `langchain` plugin, run the following:
    ```bash
    uv pip install -e '.[langchain]'
    ```

    > [!NOTE]
    > Many of the example workflows require plugins, and following the documented steps in one of these examples will in turn install the necessary plugins. For example following the steps in the `examples/simple/README.md` guide will install the `agentiq-langchain` plugin if you haven't already done so.


    In addition to plugins, there are optional dependencies needed for profiling. To install these dependencies, run the following:
    ```bash
    uv pip install -e '.[profiling]'
    ```

6. Verify the installation using the AgentIQ CLI

   ```bash
   aiq --version
   ```

   This should output the AgentIQ version which is currently installed.

## Hello World Example

1. Ensure you have set the `NVIDIA_API_KEY` environment variable to allow the example to use NVIDIA NIMs. An API key can be obtained by visiting [`build.nvidia.com`](https://build.nvidia.com/) and creating an account.

   ```bash
   export NVIDIA_API_KEY=<your_api_key>
   ```

2. Create the AgentIQ workflow configuration file. This file will define the agents, tools, and workflows that will be used in the example. Save the following as `workflow.yaml`:

   ```yaml
   functions:
      # Add a tool to search wikipedia
      wikipedia_search:
         _type: wiki_search
         max_results: 2

   llms:
      # Tell AgentIQ which LLM to use for the agent
      nim_llm:
         _type: nim
         model_name: meta/llama-3.1-70b-instruct
         temperature: 0.0

   workflow:
      # Use an agent that 'reasons' and 'acts'
      _type: react_agent
      # Give it access to our wikipedia search tool
      tool_names: [wikipedia_search]
      # Tell it which LLM to use
      llm_name: nim_llm
      # Make it verbose
      verbose: true
      # Retry parsing errors because LLMs are non-deterministic
      retry_parsing_errors: true
      # Retry up to 3 times
      max_retries: 3
   ```

3. Run the Hello World example using the `aiq` CLI and the `workflow.yaml` file.

   ```bash
   aiq run --config_file workflow.yaml --input "List five subspecies of Aardvarks"
   ```

   This will run the workflow and output the results to the console.

   ```console
   Workflow Result:
   ['Here are five subspecies of Aardvarks:\n\n1. Orycteropus afer afer (Southern aardvark)\n2. O. a. adametzi  Grote, 1921 (Western aardvark)\n3. O. a. aethiopicus  Sundevall, 1843\n4. O. a. angolensis  Zukowsky & Haltenorth, 1957\n5. O. a. erikssoni  Lönnberg, 1906']
   ```

## Feedback

We would love to hear from you! Please file an issue on [GitHub](https://github.com/NVIDIA/AgentIQ/issues) if you have any feedback or feature requests.

## Acknowledgements

We would like to thank the following open source projects that made AgentIQ possible:

- [CrewAI](https://github.com/crewAIInc/crewAI)
- [FastAPI](https://github.com/tiangolo/fastapi)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Llama-Index](https://github.com/run-llama/llama_index)
- [Mem0ai](https://github.com/mem0ai/mem0)
- [Ragas](https://github.com/explodinggradients/ragas)
- [Semantic Kernel](https://github.com/microsoft/semantic-kernel)
- [uv](https://github.com/astral-sh/uv)

# LLM-Powered Audience Research Agent

This audience research agent provides comprehensive, data-driven insights about target audiences using a combination of web search and advanced LLM processing. It transforms raw web data into structured, actionable audience intelligence.

## Features

- **Demographic Analysis**: Detailed demographic breakdown of target audiences based on industry and region, including age distribution, gender, income levels, education, geographic concentrations, and psychographics.

- **Buyer Persona Creation**: Generation of realistic, data-driven buyer personas with detailed attributes including goals, challenges, communication preferences, and buying processes.

- **Competitor Analysis**: Comprehensive analysis of competitors in a specific industry, including their strengths, weaknesses, market positioning, and product offerings.

- **Web-Search Based Intelligence**: All insights are gathered in real-time from diverse web sources for up-to-date, factually-grounded results.

- **LLM-Powered Analysis**: Advanced NeMo and Groq LLMs process and structure raw data into meaningful insights.

- **Data Validation**: Secondary LLM validation ensures accuracy and consistency of all results.

- **Multi-Layered Fallback Systems**: Graceful degradation through tiered fallback systems when external services are unavailable.

## Architecture

The agent follows a multi-stage process for audience intelligence gathering:

1. **Data Collection**: Using Tavily API to perform targeted web searches across diverse sources
2. **Data Analysis**: Using NVIDIA NeMo models to analyze and structure the raw data
3. **Insight Generation**: Transforming structured data into comprehensive audience intelligence
4. **Validation**: Using Groq LLM to validate outputs for consistency and accuracy
5. **Fallback Handling**: Multi-tiered fallback approaches when services are unavailable

## Setup

### Prerequisites

- Python 3.9+
- Required Python packages (see requirements.txt)
- API keys for the services used

### Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Copy the `.env.example` file to `.env` and add your API keys:
   ```
   cp .env.example .env
   ```
4. Edit the `.env` file to add your API keys:
   - `TAVILY_API_KEY`: Required for web search functionality
   - `NVIDIA_API_KEY`: For primary LLM analysis
   - `GROQ_API_KEY`: For secondary validation
   
### Configuration

The agent can be configured through the following environment variables:

- `ENABLE_CACHE`: Set to `true` to enable caching of search results (default: true)
- `CACHE_EXPIRY_SECONDS`: How long to cache results (default: 3600 seconds)
- `LOGGING_LEVEL`: Set to DEBUG, INFO, WARNING, or ERROR (default: INFO)

## Usage

Import and use the functions in your Python application:

```python
from aiq_audience_research import demographic_analysis, persona_builder, competitor_analysis

# Get demographic analysis
demographics = await demographic_analysis(
    industry="healthcare",
    region="United States"
)

# Create a buyer persona
persona = await persona_builder(
    industry="software", 
    region="Europe",
    job_roles=["marketing manager", "CMO"],
    product_needs=["marketing automation", "analytics"]
)

# Analyze competitors
competitors = await competitor_analysis(
    industry="e-commerce",
    region="United States",
    competitors=["Amazon", "Shopify", "BigCommerce"]
)
```

## External Services

This agent relies on the following external services:

- **Tavily API**: Web search capabilities
- **NVIDIA API**: Primary LLM for analysis
- **Groq API**: Secondary LLM for validation

See `THIRD_PARTY_SERVICES.md` for details on services used.

## Error Handling

The agent implements a multi-layered approach to error handling:

1. **Primary Web Search + LLM**: Main path using web search and NeMo LLM
2. **LLM-Only Fallback**: If web search fails, uses the LLM's knowledge
3. **Minimal Fallback**: If both above fail, returns minimal structured data

## Limitations

- Results are limited by the quality and recency of available web data
- Free tier API limits may restrict heavy usage (100 searches/day on Tavily free tier)
- LLM analysis has inherent limitations in factual accuracy and consistency

## License

This project is licensed under the MIT License - see the LICENSE file for details.
