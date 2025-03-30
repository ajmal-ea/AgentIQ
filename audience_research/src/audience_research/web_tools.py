# SPDX-FileCopyrightText: Copyright (c) 2025
# SPDX-License-Identifier: Apache-2.0

import logging
import json
from typing import Optional, Dict, Union, Any, Type, List

from dotenv import load_dotenv

load_dotenv(override=True)

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig
from pydantic import Field, BaseModel

logger = logging.getLogger(__name__)

class WebSearchConfig(FunctionBaseConfig, name="web_search"):
    """Configuration for web search tool"""
    description: str = "Search the web for market information and competitor data"
    llm_name: LLMRef

@register_function(config_type=WebSearchConfig)
async def web_search(config: WebSearchConfig, builder: Builder):
    """Web search tool for research"""
    from langchain.schema import StrOutputParser
    from langchain_core.tools import Tool
    
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    
    async def _search_web(query: str) -> str:
        """
        Simulates a web search by generating synthetic search results.
        
        In a production environment, this would connect to a real search API
        or use a web scraper with proper rate limiting and permissions.
        """
        logger.info(f"Performing web search for: {query}")
        
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
    
    class WebSearchInput(BaseModel):
        query: str = Field(description="The search query")
    
    # Create a proper LangChain structured tool that handles input correctly
    from langchain_core.tools import StructuredTool
    
    async def _run_search(query: str) -> str:
        """Run the search with proper preprocessing of input"""
        # Clean up input if it contains quotes
        if isinstance(query, str):
            query = query.strip()
            if query.startswith('"') and query.endswith('"'):
                query = query[1:-1]
        return await _search_web(query)
    
    web_search_tool = StructuredTool.from_function(
        func=_run_search,
        name="web_search",
        description="Search the web for information about the given query.",
        args_schema=WebSearchInput,
        return_direct=False,
        coroutine=True,
    )
    
    # Yield the FunctionInfo with the tool
    yield FunctionInfo.from_tool(web_search_tool, description=config.description)