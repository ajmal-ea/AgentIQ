# Audience Research Agent for Google AdWords

This agent uses NVIDIA AgentIQ to perform in-depth audience research for prospect marketing in Google AdWords. It combines multiple tool calls to gather information about keywords, competitors, demographics, and compile them into detailed audience personas that can be used for targeted marketing campaigns.

## Overview

The Audience Research Agent:
- Performs keyword research for target markets and industries
- Conducts competitor analysis to identify strategies and target audiences
- Analyzes demographic and psychographic data
- Creates detailed audience personas as output
- Integrates with Google AdWords (mock implementation)

## Setup

### Prerequisites

1. Install [NVIDIA AgentIQ](https://github.com/NVIDIA/AgentIQ)
2. Set up your environment with the required API keys:
   ```bash
   export NVIDIA_API_KEY=<your_api_key>  # Required for NIM models
   ```
   
3. Optional: For Google AdWords integration (currently simulated)
   ```bash
   export GOOGLE_ADS_CLIENT_ID=<your_client_id>
   export GOOGLE_ADS_CLIENT_SECRET=<your_client_secret>
   ```

### Installation

1. Clone this repository or copy the files to your project directory
2. Install required dependencies:
   ```bash
   pip install -e .
   ```

## Using the Agent

The Audience Research Agent can be used through any of the standard AgentIQ commands:

### Using `aiq run`

```bash
aiq run --config_file development/audience_research/workflow.yml --input "Research audience for enterprise software companies targeting healthcare organizations"
```

For more control, you can use the provided wrapper script:

```bash
python development/audience_research/run_audience_agent.py --industry "software" --query "enterprise software" --region "United States" --output-dir "./output" --command run
```

### Using `aiq serve`

Start a web server for the agent:

```bash
aiq serve --config_file development/audience_research/workflow.yml
```

Or use the wrapper script:

```bash
python development/audience_research/run_audience_agent.py --industry "software" --query "enterprise software" --command serve
```

Once the server is running, you can send requests to it:

```bash
curl --request POST \
  --url http://localhost:8000/generate \
  --header 'Content-Type: application/json' \
  --data '{"input_message": "Research audience for enterprise software companies targeting healthcare organizations"}'
```

### Using `aiq eval`

Evaluate the agent's performance:

```bash
aiq eval --config_file development/audience_research/workflow.yml --input "Research audience for enterprise software companies targeting healthcare organizations"
```

Or use the wrapper script:

```bash
python development/audience_research/run_audience_agent.py --industry "software" --query "enterprise software" --command eval
```

## Observability with Phoenix

The agent is configured with tracing support using Phoenix. To enable observability:

1. Install Phoenix:
   ```bash
   pip install arize-phoenix
   ```

2. Start the Phoenix server:
   ```bash
   phoenix serve
   ```

3. Run your workflow with any of the aiq commands
4. Open Phoenix UI at http://localhost:6006 to view traces

## Available Tools

The Audience Research Agent provides the following custom tools:

1. **keyword_research**: Performs keyword research for a specified query
   - Parameters: `query`, `max_results`, `include_search_volume`, `include_competition`
   
2. **competitor_analysis**: Analyzes competitors in a specified industry
   - Parameters: `industry`, `competitor` (optional), `max_results`, `include_strategies`, `include_target_audience`
   
3. **demographic_analysis**: Analyzes demographic and psychographic data
   - Parameters: `industry`, `region`, `include_psychographics`
   
4. **persona_builder**: Creates audience personas based on research data
   - Parameters: `industry`, `persona_name`, `number_of_personas`, `include_demographics`, `include_psychographics`, `include_online_behavior`

5. **audience_research**: Coordinates all the above tools for end-to-end research
   - Parameters: `mode`, `output_directory`

## Command-line Arguments

When using the `run_audience_agent.py` script, the following arguments are available:

```
--industry      Industry to research (e.g., software, marketing, finance)
--query         Keyword query for research
--region        Region to focus on
--competitor    Specific competitor to analyze (optional)
--personas      Number of personas to generate
--persona-name  Name for the persona
--output-dir    Directory to save output files
--config        Path to workflow configuration file
--command       AgentIQ command to run (run, serve, eval)
```

## Integration with AdWords

After generating audience personas, you can create an AdWords campaign with:

```bash
python development/audience_research/adwords_integration.py --result-file ./output/software_audience_research_result.json --campaign-name "Healthcare Software Campaign" --output-file ./output/adwords_campaign.json
```

## Example Workflow

1. The agent receives a query about a target audience
2. It performs keyword research to identify relevant search terms
3. It analyzes competitors in the industry to understand their targeting
4. It gathers demographic and psychographic data about the audience
5. It combines all this information to create detailed personas
6. The personas can be exported to various formats or used to create AdWords campaigns

## Advanced Configuration

The agent behavior can be customized by modifying the `workflow.yml` file:
- Change the LLM model
- Adjust the number of results
- Enable/disable features
- Update the system prompt

You can also override specific configuration values using the `--override` flag:

```bash
aiq run --config_file development/audience_research/workflow.yml --input "Research audience for enterprise software" --override llms.nim_llm.temperature 0.5
```

## Troubleshooting

If you encounter issues with the agent:

1. Check that `NVIDIA_API_KEY` is properly set
2. Ensure you have the latest version of AgentIQ installed
3. Verify your workflow.yml configuration
4. Check the logs for specific error messages
5. Verify Phoenix is running if you're using observability features

## Future Improvements

See the `todolist.md` file for planned enhancements to the Audience Research Agent.