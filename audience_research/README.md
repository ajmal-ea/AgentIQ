# Audience Research Agent

A comprehensive Audience Research Agent system built with NVIDIA AgentIQ. This system leverages a multi-agent architecture with specialized sub-agents to perform detailed audience research for any target market.

## Architecture

The system consists of multiple specialized agents that work together:

1. **Keyword Research Agent**: Identifies relevant keywords for a target market
2. **Competitor Analysis Agent**: Analyzes competitor strategies and positioning
3. **Demographic Analysis Agent**: Gathers demographic and psychographic insights
4. **Persona Builder Agent**: Creates detailed audience personas
5. **Audience Research Coordinator (Master Agent)**: Orchestrates the entire workflow

Additional tools:
- **Web Search Tool**: Simulates web search for gathering real-time data

## Installation

If you have not already done so, follow the instructions in the NVIDIA AgentIQ Install Guide to create the development environment and install AgentIQ.

### Install this Workflow:

From the root directory of the AgentIQ library, run:

```bash
pip install -e audience_research
```

### Set Up API Keys

You need to set your NVIDIA API key as an environment variable:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

## Usage

Run the workflow with the audience research query:

```bash
aiq run --config_file audience_research/src/audience_research/configs/config.yml --input "Digital marketing software for small businesses"
```

## Key Outputs

The system generates:
- Detailed audience persona documents
- Keyword lists for targeting
- Competitor analysis reports
- Comprehensive audience research report

## Configuration

The workflow is configured through YAML files:
- `config.yml`: Main workflow configuration

You can customize the LLM models used by modifying the configuration file.

## Advanced Features

- **Modular Design**: Each agent is a specialized module that can be used independently
- **Dynamic Prompts**: LLM-driven prompts adapt to different data inputs
- **Minimal Hardcoding**: Configuration files allow flexibility without rewriting code
- **Orchestration**: Master agent coordinates the workflow and compiles the final output 