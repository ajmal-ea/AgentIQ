# SPDX-FileCopyrightText: Copyright (c) 2025
# SPDX-License-Identifier: Apache-2.0

import logging
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv(override=True)

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import EmbedderRef, LLMRef
from aiq.data_models.function import FunctionBaseConfig
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Keyword Research Agent
class KeywordResearchConfig(FunctionBaseConfig, name="keyword_research"):
    description: str = "Research and identify relevant keywords for a target market"
    llm_name: LLMRef
    query: str

@register_function(config_type=KeywordResearchConfig)
async def keyword_research(config: KeywordResearchConfig, builder: Builder):
    """Agent for keyword research"""
    from langchain_core.tools import Tool
    
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
    
    async def _inner(query: str) -> str:
        result = await _search_keywords(query)
        return result
    
    yield FunctionInfo.from_fn(_inner, description=config.description)

# Competitor Analysis Agent
class CompetitorAnalysisConfig(FunctionBaseConfig, name="competitor_analysis"):
    description: str = "Analyze strategies of competitors in a target market"
    llm_name: LLMRef
    market: str
    industry: str
    competitors: Optional[List[str]] = None

@register_function(config_type=CompetitorAnalysisConfig)
async def competitor_analysis(config: CompetitorAnalysisConfig, builder: Builder):
    """Agent for competitor analysis"""
    from langchain_core.tools import Tool
    
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
    
    async def _inner(market: str) -> str:
        result = await _analyze_competitors(market, config.industry, config.competitors)
        return result
    
    yield FunctionInfo.from_fn(_inner, description=config.description)

# Demographic Analysis Agent
class DemographicAnalysisConfig(FunctionBaseConfig, name="demographic_analysis"):
    description: str = "Analyze demographics and psychographics of a target market"
    llm_name: LLMRef
    market: str
    industry: str

@register_function(config_type=DemographicAnalysisConfig)
async def demographic_analysis(config: DemographicAnalysisConfig, builder: Builder):
    """Agent for demographic analysis"""
    from langchain_core.tools import Tool
    
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
    
    async def _inner(market: str) -> str:
        result = await _analyze_demographics(market, config.industry)
        return result
    
    yield FunctionInfo.from_fn(_inner, description=config.description)

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
    async def _process_input_data(input_data: Dict[str, Any]) -> str:
        """Process the input data from React agent"""
        logger.info(f"Building persona with input data: {input_data}")
        
        # Extract name and description from input data if available
        persona_name = input_data.get("name", "")
        persona_description = input_data.get("description", "")
        
        # Use the persona description as additional context
        additional_context = f"\nTarget Persona: {persona_name}\nPersona Description: {persona_description}\n" if persona_name else ""
        
        # Generate a specialized market/industry based on description if available
        market = persona_description if persona_description else config.market
        industry = config.industry
        
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

# Web Search Tool
class WebSearchConfig(FunctionBaseConfig, name="web_search"):
    """Configuration for web search tool"""
    description: str = "Search the web for market information and competitor data"
    llm_name: LLMRef
    query: str

@register_function(config_type=WebSearchConfig)
async def web_search(config: WebSearchConfig, builder: Builder):
    """Web search tool for research"""
    from langchain_core.tools import Tool
    import requests
    from bs4 import BeautifulSoup
    
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    
    async def _search_web(query: str) -> str:
        """
        Simulates a web search by generating synthetic search results.
        
        In a production environment, this would connect to a real search API
        or use a web scraper with proper rate limiting and permissions.
        """
        # For demo purposes, we'll have the LLM generate simulated search results
        prompt = f"""
        You are a web search simulator. Generate realistic search results for the following query:
        
        Query: {query}
        
        Format your response as a list of 5-7 search results, each with:
        1. Title (realistic page title)
        2. URL (realistic but fictional URL)
        3. Snippet (short excerpt from the page)
        
        The results should be diverse, realistic, and provide genuinely useful information about the query topic.
        """
        
        search_results = await llm.ainvoke(prompt)
        return search_results
    
    async def _analyze_webpage(url: str) -> str:
        """
        Simulates webpage content analysis by generating synthetic content.
        
        In a production environment, this would fetch and parse actual webpage content
        with proper permissions and rate limiting.
        """
        # For demo purposes, we'll have the LLM generate simulated webpage content
        prompt = f"""
        You are a webpage content simulator. Generate realistic content for a webpage with this URL:
        
        URL: {url}
        
        Format your response as:
        1. Page Title
        2. Main Content (several paragraphs of realistic content that would appear on this page)
        3. Key Points/Findings (bullet points of important information)
        
        Make the content detailed, realistic, and focused on the topic implied by the URL.
        """
        
        webpage_content = await llm.ainvoke(prompt)
        return webpage_content
    
    async def _inner(query: str) -> str:
        """Main function for the web search tool"""
        search_results = await _search_web(query)
        
        # Add analysis context
        analysis_prompt = f"""
        You are a market research analyst. Analyze the following search results to extract key insights:
        
        SEARCH RESULTS FOR: {query}
        {search_results}
        
        Provide a summary of the most important findings, trends, and data points from these results.
        Focus on information that would be relevant for audience research and market analysis.
        """
        
        analysis = await llm.ainvoke(analysis_prompt)
        
        return f"Search Results:\n{search_results}\n\nAnalysis:\n{analysis}"
    
    yield FunctionInfo.from_fn(_inner, description=config.description) 