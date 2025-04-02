# SPDX-FileCopyrightText: Copyright (c) 2025
# SPDX-License-Identifier: Apache-2.0

import logging
import json
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv

load_dotenv(override=True)

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import EmbedderRef, LLMRef
from aiq.data_models.function import FunctionBaseConfig
from aiq.data_models.workflow import WorkflowBaseConfig, WorkflowInfo
from pydantic import BaseModel
from aiq.graph.graph import Graph
from aiq.agent.react_agent.prompt import react_agent_prompt

logger = logging.getLogger(__name__)

# Keyword Research Agent
class KeywordResearchConfig(FunctionBaseConfig, name="keyword_research"):
    description: str = "Research and identify relevant keywords for a target market"
    llm_name: LLMRef
    query: str

# Define the input schema for keyword research
class KeywordResearchInput(BaseModel):
    input_data: Dict[str, Any]

@register_function(config_type=KeywordResearchConfig)
async def keyword_research(config: KeywordResearchConfig, builder: Builder):
    """Agent for keyword research"""
    from langchain_core.tools import Tool, StructuredTool
    from langchain_core.pydantic_v1 import BaseModel, Field
    
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    
    async def _search_keywords(query: str) -> str:
        """Search for relevant keywords for a target market"""
        prompt = f"""
        You are a keyword research specialist. Your task is to identify relevant keywords for the following market:
        
        {query}
        
        Please provide a list of at least 30 keywords organized by:
        1. Primary keywords (high volume, high intent)
        2. Secondary keywords (related terms)
        3. Long-tail keywords (specific phrases)
        
        For each keyword, include:
        - Estimated search volume category (High/Medium/Low)
        - Competition level (High/Medium/Low)
        - Relevance to the target market (High/Medium/Low)
        
        Format your response as a JSON object with these categories.
        """
        
        response = await llm.ainvoke(prompt)
        return response
    
    # Process structured input
    async def _process_input(input_data: Any) -> str:
        """Process the input data from React agent"""
        logger.info(f"Performing keyword research with input: {input_data}")
        
        # Handle different input formats
        query = ""
        
        # Try to handle both dict and string inputs
        if isinstance(input_data, dict):
            # Extract query from input data dictionary
            if "input_data" in input_data and isinstance(input_data["input_data"], dict):
                # Extract from nested input_data structure
                query = input_data["input_data"].get("query", "")
            else:
                # Try direct key access
                query = input_data.get("query", "")
        elif isinstance(input_data, str):
            # If input is a string, use it directly
            query = input_data.strip('"\'')
            logger.info(f"Using string input as query: {query}")
        
        # Fall back to config query if not provided in input
        if not query:
            query = config.query
            logger.info(f"Using fallback query from config: {query}")
            
        # Execute the keyword search
        result = await _search_keywords(query)
        return result
    
    # Define the structured tool using Pydantic schema
    class KeywordInputSchema(BaseModel):
        input_data: Dict[str, Any] = Field(
            description="Input data dictionary with 'query' field for keyword research"
        )
    
    # Create a structured tool that accepts the schema
    tool = StructuredTool.from_function(
        func=_process_input,
        name="keyword_research",
        description=config.description,
        args_schema=KeywordInputSchema,
        handle_tool_error=True,
    )
    
    yield FunctionInfo.from_tool(tool, description=config.description)

# Competitor Analysis Agent
class CompetitorAnalysisConfig(FunctionBaseConfig, name="competitor_analysis"):
    description: str = "Analyze strategies of competitors in a target market"
    llm_name: LLMRef
    market: str
    industry: str
    competitors: Optional[List[str]] = None

# Define the input schema for competitor analysis
class CompetitorAnalysisInput(BaseModel):
    input_data: Dict[str, Any]

@register_function(config_type=CompetitorAnalysisConfig)
async def competitor_analysis(config: CompetitorAnalysisConfig, builder: Builder):
    """Agent for competitor analysis"""
    from langchain_core.tools import Tool, StructuredTool
    from langchain_core.pydantic_v1 import BaseModel, Field
    
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    
    async def _analyze_competitors(market: str, industry: str, competitors: Optional[List[str]] = None) -> str:
        """Analyze competitor strategies in a given market"""
        competitors_text = ""
        if competitors:
            competitors_text = "Specifically analyze these competitors: " + ", ".join(competitors)
            
        prompt = f"""
        You are a competitor analysis specialist. Your task is to analyze the competitor landscape for:
        
        Market: {market}
        Industry: {industry}
        {competitors_text}
        
        Please provide a detailed analysis including:
        1. Market positioning of key competitors
        2. Target audience similarities and differences
        3. Marketing strategies and channels used
        4. Content and messaging approaches
        5. Strengths and weaknesses of each competitor
        
        Format your response as a JSON object with these categories.
        """
        
        response = await llm.ainvoke(prompt)
        return response
    
    # Process structured input
    async def _process_input(input_data: Any) -> str:
        """Process the input data from React agent"""
        logger.info(f"Analyzing competitors with input: {input_data}")
        
        # Initialize with default values
        market = config.market
        industry = config.industry
        competitors = config.competitors
        
        # Try to handle both dict and string inputs
        if isinstance(input_data, dict):
            # Extract from input data dictionary
            if "input_data" in input_data and isinstance(input_data["input_data"], dict):
                # Extract from nested input_data structure
                inner_data = input_data["input_data"]
                market = inner_data.get("market", market)
                industry = inner_data.get("industry", industry)
                if "competitors" in inner_data:
                    competitors = inner_data["competitors"]
            else:
                # Try direct key access
                market = input_data.get("market", market)
                industry = input_data.get("industry", industry)
                if "competitors" in input_data:
                    competitors = input_data["competitors"]
        elif isinstance(input_data, str):
            # If input is a string, try to use it as market
            input_str = input_data.strip('"\'')
            if input_str:
                market = input_str
                logger.info(f"Using string input as market: {market}")
            
        # Execute the competitor analysis    
        result = await _analyze_competitors(market, industry, competitors)
        return result
    
    # Define the structured tool using Pydantic schema
    class CompetitorInputSchema(BaseModel):
        input_data: Dict[str, Any] = Field(
            description="Input data dictionary with 'market', 'industry', and 'competitors' fields"
        )
    
    # Create a structured tool that accepts the schema
    tool = StructuredTool.from_function(
        func=_process_input,
        name="competitor_analysis",
        description=config.description,
        args_schema=CompetitorInputSchema,
        handle_tool_error=True,
    )
    
    yield FunctionInfo.from_tool(tool, description=config.description)

# Demographic Analysis Agent
class DemographicAnalysisConfig(FunctionBaseConfig, name="demographic_analysis"):
    description: str = "Analyze demographics and psychographics of a target market"
    llm_name: LLMRef
    market: str
    industry: str

# Define the input schema for demographic analysis
class DemographicAnalysisInput(BaseModel):
    input_data: Dict[str, Any]

@register_function(config_type=DemographicAnalysisConfig)
async def demographic_analysis(config: DemographicAnalysisConfig, builder: Builder):
    """Agent for demographic analysis"""
    from langchain_core.tools import Tool, StructuredTool
    from langchain_core.pydantic_v1 import BaseModel, Field
    
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    
    async def _analyze_demographics(market: str, industry: str) -> str:
        """Analyze demographics and psychographics of a target market"""
        prompt = f"""
        You are a demographic analysis specialist. Your task is to analyze the demographic and psychographic characteristics of:
        
        Market: {market}
        Industry: {industry}
        
        Please provide a detailed analysis including:
        1. Key demographic data (age ranges, gender distribution, income levels, education, location)
        2. Psychographic insights (interests, values, attitudes, lifestyle choices)
        3. Behavioral patterns (purchasing habits, content consumption, decision-making factors)
        4. Pain points and motivations
        5. Technology usage and preferences
        
        Format your response as a JSON object with these categories.
        """
        
        response = await llm.ainvoke(prompt)
        return response
    
    # Process structured input
    async def _process_input(input_data: Any) -> str:
        """Process the input data from React agent"""
        logger.info(f"Analyzing demographics with input: {input_data}")
        
        # Initialize with default values
        market = config.market
        industry = config.industry
        
        # Try to handle both dict and string inputs
        if isinstance(input_data, dict):
            # Extract from input data dictionary
            if "input_data" in input_data and isinstance(input_data["input_data"], dict):
                # Extract from nested input_data structure
                inner_data = input_data["input_data"]
                market = inner_data.get("market", market)
                industry = inner_data.get("industry", industry)
            else:
                # Try direct key access
                market = input_data.get("market", market)
                industry = input_data.get("industry", industry)
        elif isinstance(input_data, str):
            # If input is a string, try to use it as market
            input_str = input_data.strip('"\'')
            if input_str:
                market = input_str
                logger.info(f"Using string input as market: {market}")
            
        # Execute the demographic analysis
        result = await _analyze_demographics(market, industry)
        return result
    
    # Define the structured tool using Pydantic schema
    class DemographicInputSchema(BaseModel):
        input_data: Dict[str, Any] = Field(
            description="Input data dictionary with 'market' and 'industry' fields"
        )
    
    # Create a structured tool that accepts the schema
    tool = StructuredTool.from_function(
        func=_process_input,
        name="demographic_analysis",
        description=config.description,
        args_schema=DemographicInputSchema,
        handle_tool_error=True,
    )
    
    yield FunctionInfo.from_tool(tool, description=config.description)

# Persona Builder Agent
class PersonaBuilderConfig(FunctionBaseConfig, name="persona_builder"):
    description: str = "Build detailed audience personas based on research data"
    llm_name: LLMRef
    market: str
    industry: str
    keyword_data: Optional[str] = None
    competitor_data: Optional[str] = None
    demographic_data: Optional[str] = None

# Define the input schema for persona builder to match React agent's input format
class PersonaBuilderInput(BaseModel):
    input_data: Dict[str, Any]

@register_function(config_type=PersonaBuilderConfig)
async def persona_builder(config: PersonaBuilderConfig, builder: Builder):
    """Agent for building audience personas"""
    from langchain_core.tools import Tool
    from langchain_core.pydantic_v1 import BaseModel, Field
    
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    
    async def _build_personas(market: str, industry: str, keyword_data: Optional[str] = None, 
                            competitor_data: Optional[str] = None, 
                            demographic_data: Optional[str] = None) -> str:
        """Build detailed audience personas based on research data"""
        # Prepare the research data context
        research_context = f"Market: {market}\nIndustry: {industry}\n\n"
        
        if keyword_data:
            research_context += f"Keyword Research Data:\n{keyword_data}\n\n"
        
        if competitor_data:
            research_context += f"Competitor Analysis Data:\n{competitor_data}\n\n"
            
        if demographic_data:
            research_context += f"Demographic Analysis Data:\n{demographic_data}\n\n"
        
        prompt = f"""
        You are a persona development specialist. Your task is to create 3-5 detailed audience personas based on the following research data:
        
        {research_context}
        
        For each persona, include:
        1. Name, age, occupation, and other demographic details
        2. Goals and motivations
        3. Pain points and challenges
        4. Purchasing behavior and decision factors
        5. Media consumption habits
        6. Technology usage patterns
        7. Preferred communication channels
        8. Relevant quotes or statements that capture their mindset
        
        Include a photo description for each persona. Format your response as a JSON object with each persona as a separate entity.
        """
        
        response = await llm.ainvoke(prompt)
        return response
    
    # New implementation to support structured input from React agent
    async def _process_input_data(input_data: Any) -> str:
        """Process the input data from React agent"""
        logger.info(f"Building persona with input data: {input_data}")
        
        # Initialize with default values
        persona_name = ""
        persona_description = ""
        market = config.market
        industry = config.industry
        
        # Try to handle both dict and string inputs
        if isinstance(input_data, dict):
            # Extract from input data dictionary
            if "input_data" in input_data and isinstance(input_data["input_data"], dict):
                # Extract from nested input_data structure
                inner_data = input_data["input_data"]
                persona_name = inner_data.get("name", "")
                persona_description = inner_data.get("description", "")
                market = inner_data.get("market", market)
                industry = inner_data.get("industry", industry)
            else:
                # Try direct key access
                persona_name = input_data.get("name", "")
                persona_description = input_data.get("description", "")
                market = input_data.get("market", market)
                industry = input_data.get("industry", industry)
        elif isinstance(input_data, str):
            # If input is a string, try to use it as description
            input_str = input_data.strip('"\'')
            if input_str:
                persona_description = input_str
                logger.info(f"Using string input as persona description: {persona_description}")
        
        # Use the persona description as additional context
        additional_context = f"\nTarget Persona: {persona_name}\nPersona Description: {persona_description}\n" if persona_name else ""
        
        # Generate a specialized market/industry based on description if available
        if persona_description and not market:
            market = persona_description
        
        # Build the personas using the existing function
        result = await _build_personas(
            market, 
            industry,
            config.keyword_data, 
            config.competitor_data, 
            config.demographic_data
        )
        
        return result
    
    # Define the structured tool using Pydantic schema
    class PersonaInputSchema(BaseModel):
        input_data: Dict[str, Any] = Field(
            description="Input data dictionary with 'name' and 'description' fields for the persona"
        )
    
    # Create a structured tool that accepts the schema
    from langchain_core.tools import StructuredTool
    
    tool = StructuredTool.from_function(
        func=_process_input_data,
        name="persona_builder",
        description=config.description,
        args_schema=PersonaInputSchema,
        handle_tool_error=True,
    )
    
    yield FunctionInfo.from_tool(tool, description=config.description)

# Audience Research Coordinator (Master Agent)
class AudienceResearchCoordinatorConfig(FunctionBaseConfig, name="audience_research_coordinator"):
    description: str = "Coordinate the entire audience research workflow"
    llm_name: LLMRef
    market: str
    competitors: Optional[List[str]] = None
    
@register_function(config_type=AudienceResearchCoordinatorConfig)
async def audience_research_coordinator(config: AudienceResearchCoordinatorConfig, builder: Builder):
    """Master agent for coordinating the audience research workflow"""
    from langchain_core.tools import Tool
    
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    
    async def _coordinate_research(market: str, competitors: Optional[List[str]] = None) -> str:
        """Coordinate the entire audience research workflow"""
        # 1. Generate market context
        context_prompt = f"""
        You are a market research specialist. Please provide a brief overview of the following market:
        
        Market: {market}
        
        Focus on key characteristics, trends, and challenges in this market.
        """
        
        market_context = await llm.ainvoke(context_prompt)
        
        # 2. Generate the final report
        report_prompt = f"""
        You are an audience research coordinator. Your task is to create a comprehensive audience research report for:
        
        Market: {market}
        Market Context: {market_context}
        
        Your report should include the following sections:
        1. Executive Summary
        2. Market Overview
        3. Keyword Analysis (what terms and topics are important in this market)
        4. Competitor Landscape (key players and their positioning)
        5. Audience Demographics and Psychographics
        6. Audience Personas (3-5 detailed personas)
        7. Content Strategy Recommendations
        8. Channel Strategy Recommendations
        9. Next Steps and Action Plan
        
        Make all sections detailed and actionable. This report will be used to guide marketing strategy.
        """
        
        report = await llm.ainvoke(report_prompt)
        return report
    
    async def _inner(market: str) -> str:
        result = await _coordinate_research(market, config.competitors)
        return result
    
    yield FunctionInfo.from_fn(_inner, description=config.description)

# Web Search Tool - Tavily Internet Search Wrapper
class TavilySearchConfig(FunctionBaseConfig, name="tavily_internet_search"):
    """Configuration for Tavily internet search tool"""
    description: str = "Search the web for market information and competitor data. USAGE: Input must be a JSON with 'query' field, e.g. {\"query\": \"your search query\"}"
    max_results: int = 5
    api_key: str = ""  # Optional API key, will use environment variable if not provided

# Define the input schema for Tavily search
class TavilySearchInput(BaseModel):
    query: str = ""

@register_function(config_type=TavilySearchConfig)
async def tavily_internet_search(config: TavilySearchConfig, builder: Builder):
    """Wrapper for the Tavily internet search tool that handles both JSON and string inputs"""
    import os
    from langchain_core.tools import StructuredTool
    from langchain_core.pydantic_v1 import BaseModel, Field
    from langchain_community.tools import TavilySearchResults
    
    # Set Tavily API key from config or environment
    if config.api_key:
        os.environ["TAVILY_API_KEY"] = config.api_key
    
    # Helper function to parse input and extract the query
    async def _parse_input(input_data: Union[str, Dict]) -> str:
        """Parse the input data to extract the query"""
        query = ""
        
        # Try to handle different input formats
        if isinstance(input_data, str):
            # If input is a string, use it directly as the query
            # Remove any surrounding quotes and trim whitespace
            query = input_data.strip().strip('"\'')
            
            # Handle edge cases where the string might be a quoted JSON or contains a query with quotes
            if query.startswith('{') and query.endswith('}'):
                # Might be a JSON string, try to parse it
                try:
                    json_data = json.loads(query)
                    if isinstance(json_data, dict) and "query" in json_data:
                        query = json_data["query"]
                except:
                    # If parsing fails, keep the original string
                    pass
            
            logger.info(f"Using string input as query: {query}")
        elif isinstance(input_data, dict):
            # If input is a dict, try multiple ways to extract the query
            if "query" in input_data:
                # Direct query field
                query = input_data["query"]
            elif "input_data" in input_data and isinstance(input_data["input_data"], dict):
                # Nested input_data structure
                query = input_data["input_data"].get("query", "")
            
            logger.info(f"Extracted query from dict: {query}")
        else:
            # Default to empty string if input format is unrecognized
            logger.warning(f"Unrecognized input format: {type(input_data)}")
        
        return query
    
    # Function to perform the actual search using Tavily
    async def _perform_search(query: str) -> str:
        """Perform the search using Tavily Search API"""
        logger.info(f"Performing Tavily search with query: {query}")
        
        try:
            # Use the LangChain Tavily tool directly instead of trying to get it from builder
            # This is more reliable as it ensures we have a proper Tavily tool instance
            tavily_search = TavilySearchResults(max_results=config.max_results)
            search_results = await tavily_search.ainvoke({'query': query})
            
            # Format the results for better readability
            formatted_results = "\n\n---\n\n".join(
                [f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>' 
                for doc in search_results]
            )
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error performing Tavily search: {str(e)}")
            # Return a graceful error message
            return f"Sorry, I encountered an error while searching: {str(e)}"
    
    # The main function that handles any input format
    async def _flexible_search(input_data: Any) -> str:
        """Flexible search function that handles various input formats"""
        # Parse the input to extract the query
        query = await _parse_input(input_data)
        
        if not query:
            return "Please provide a search query."
        
        # Perform the search
        result = await _perform_search(query)
        return result
    
    # Define the schema for structured input
    class SearchInputSchema(BaseModel):
        query: str = Field(
            description="The search query to use for finding information"
        )
    
    # Create a structured tool
    tool = StructuredTool.from_function(
        func=_flexible_search,
        name="internet_search",
        description=config.description,
        args_schema=SearchInputSchema,
        handle_tool_error=True,
    )
    
    yield FunctionInfo.from_tool(tool, description=config.description)

class OutputParserFixConfig(FunctionBaseConfig, name="output_parser_fix"):
    """Configuration for the output parser fix tool that uses LLM to reformat output."""
    description: str = "Fix output parsing errors using LLM"
    llm_name: Optional[LLMRef] = None

@register_function(config_type=OutputParserFixConfig)
async def output_parser_fix(config: OutputParserFixConfig, builder: Builder):
    """Register a tool to fix output parsing errors using LLM."""
    from langchain_core.tools import Tool
    from .output_parser_fix import reparse_with_llm, clean_output
    
    # Get the LLM or use the default one
    llm = None
    if config.llm_name:
        llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    
    async def _fix_output(text: str) -> str:
        """Fix the output format of a ReAct agent response or clean up final output."""
        logger.info(f"Processing output: {text[:100]}...")
        
        # If llm was not provided in config, get a default one
        nonlocal llm
        if llm is None:
            # Get any available LLM
            available_llm_names = await builder.list_llms()
            if available_llm_names:
                llm = await builder.get_llm(available_llm_names[0], wrapper_type=LLMFrameworkEnum.LANGCHAIN)
            else:
                return "Error: No LLM available to fix output format."
        
        # Check if this looks like a final answer that needs cleanup
        if "AUDIENCE RESEARCH REPORT" in text or "FINAL ANSWER" in text.upper():
            logger.info("Detected final output - cleaning up")
            try:
                cleaned_output = await clean_output(text, llm)
                return cleaned_output
            except Exception as e:
                logger.error(f"Error cleaning final output: {e}")
                return text
        
        # Otherwise, try to fix the ReAct format
        try:
            # Use our LLM-based reparser
            agent_output = await reparse_with_llm(
                original_output=text,
                llm=llm,
                error_message="The output format needs to be fixed for the ReAct agent."
            )
            
            # Return the reformatted output
            if hasattr(agent_output, "return_values") and "output" in agent_output.return_values:
                return agent_output.return_values["output"]
            elif hasattr(agent_output, "log"):
                return agent_output.log
            elif hasattr(agent_output, "tool_input"):
                # If we got a valid AgentAction, format it properly
                tool = agent_output.tool
                tool_input = agent_output.tool_input
                return f"Thought: I need to use the {tool} tool.\nAction: {tool}\nAction Input: {tool_input}"
            else:
                return str(agent_output)
        except Exception as e:
            logger.error(f"Error fixing output format: {e}")
            return f"I encountered an error when trying to fix the format. Let me try again with a simpler approach:\n\n{text}"
    
    tool = Tool.from_function(
        func=_fix_output,
        name="output_parser_fix",
        description="Fix output parsing errors using LLM or clean up final audience research reports.",
        handle_tool_error=True,
    )
    
    yield FunctionInfo.from_tool(tool, description=config.description) 