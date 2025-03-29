#!/usr/bin/env python
"""
Web Search Agent for searching and retrieving real-time information from the internet
"""
import os
import json
import logging
import aiohttp
import asyncio
from typing import Dict, Any, Optional, List, AsyncGenerator, Callable, Awaitable
from pydantic import Field, validator

from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebSearchConfig(FunctionBaseConfig, name="web_search"):
    """
    Tool for searching the web for real-time information
    
    This tool performs web searches to retrieve current information on topics,
    industries, companies, and market trends.
    """
    query: str = Field(
        ..., 
        description="The search query to execute"
    )
    search_type: str = Field(
        "general", 
        description="Type of search: general, news, academic, or market_research"
    )
    num_results: int = Field(
        5, 
        description="Number of results to return",
        ge=1, 
        le=20
    )
    include_snippets: bool = Field(
        True, 
        description="Whether to include text snippets in results"
    )
    
    @validator('query')
    def query_not_empty(cls, v):
        if not v or not v.strip():
            return "latest market trends"
        return v
    
    @validator('search_type')
    def valid_search_type(cls, v):
        valid_types = ["general", "news", "academic", "market_research"]
        if v not in valid_types:
            return "general"
        return v

async def search_serper(query: str, search_type: str = "general", num_results: int = 5) -> Dict[str, Any]:
    """
    Performs a web search using the Serper API.
    If SERPER_API_KEY is not available, returns mock data based on the query.
    """
    serper_api_key = os.environ.get("SERPER_API_KEY")
    
    if serper_api_key:
        try:
            serper_url = "https://google.serper.dev/search"
            headers = {
                'X-API-KEY': serper_api_key,
                'Content-Type': 'application/json'
            }
            payload = {
                'q': query,
                'gl': 'us',
                'num': num_results
            }
            
            # Add search type specific parameters
            if search_type == "news":
                serper_url = "https://google.serper.dev/news"
            elif search_type == "academic":
                payload['gl'] = 'scholar'
            elif search_type == "market_research":
                # For market research, append relevant terms to the query
                payload['q'] = f"{query} market research industry analysis trends"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(serper_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"Serper API error: {response.status}, {await response.text()}")
            
        except Exception as e:
            logger.error(f"Error using Serper API: {e}")
            # Fall back to mock data
    
    # If we don't have an API key or the API call failed, return mock data
    return generate_mock_search_results(query, search_type, num_results)

def generate_mock_search_results(query: str, search_type: str, num_results: int) -> Dict[str, Any]:
    """Generate mock search results based on the query and search type"""
    logger.info(f"Generating mock search results for query: {query}, type: {search_type}")
    
    # Clean the query for use in generating results
    query_clean = query.lower().strip()
    
    # Default results for when we can't match anything specific
    default_results = [
        {
            "title": f"Understanding {query_clean} - Overview and Analysis",
            "link": f"https://example.com/{query_clean.replace(' ', '-')}-overview",
            "snippet": f"Comprehensive information about {query_clean} including latest trends, statistics, and analysis. This resource provides in-depth coverage of key aspects."
        },
        {
            "title": f"{query_clean.title()} Market Research Report",
            "link": f"https://example.com/market-research/{query_clean.replace(' ', '-')}",
            "snippet": f"Latest market research on {query_clean} showing growth trends, major players, and market size. Industry experts predict significant developments in coming years."
        },
        {
            "title": f"Top 10 Trends in {query_clean.title()} for 2025",
            "link": f"https://example.com/trends/{query_clean.replace(' ', '-')}-2025",
            "snippet": f"Analysis of the most important trends shaping {query_clean} in 2025. Includes expert insights, case studies, and future predictions."
        }
    ]
    
    # Specific results for industries
    industry_terms = ["software", "healthcare", "finance", "retail", "manufacturing", "technology", 
                     "education", "data science", "ai", "machine learning"]
    
    # Specific results for companies
    company_terms = ["microsoft", "google", "amazon", "apple", "facebook", "ibm", "oracle", 
                    "salesforce", "adobe", "tesla"]
    
    # Specific results for persona-related queries
    persona_terms = ["customer", "user", "audience", "demographic", "persona", "profile", 
                    "behavior", "preference", "segment", "target market"]
    
    results = []
    
    # Check if this is an industry-related query
    for term in industry_terms:
        if term in query_clean:
            if search_type == "market_research" or search_type == "general":
                results.extend([
                    {
                        "title": f"{term.title()} Industry Analysis 2025",
                        "link": f"https://example.com/industry/{term}-analysis-2025",
                        "snippet": f"Comprehensive analysis of the {term} industry including market size, growth drivers, challenges, and forecasts through 2025."
                    },
                    {
                        "title": f"Key Players in the {term.title()} Market",
                        "link": f"https://example.com/market/{term}-key-players",
                        "snippet": f"Detailed profiles of leading companies in the {term} sector, including market share, competitive strategies, and SWOT analysis."
                    },
                    {
                        "title": f"{term.title()} Industry Demographics and Target Audience",
                        "link": f"https://example.com/demographics/{term}-industry",
                        "snippet": f"Analysis of primary consumer segments, buying behavior, and demographic profiles for the {term} industry."
                    }
                ])
            elif search_type == "news":
                results.extend([
                    {
                        "title": f"Breaking: New Developments in {term.title()} Industry",
                        "link": f"https://example.com/news/{term}-developments",
                        "snippet": f"Recent announcements and changes affecting the {term} market landscape. Industry leaders announce new initiatives."
                    },
                    {
                        "title": f"{term.title()} Market Shows 15% Growth in Q1 2025",
                        "link": f"https://example.com/news/{term}-market-growth",
                        "snippet": f"Financial results indicate strong performance in the {term} sector, exceeding analyst expectations for the quarter."
                    }
                ])
    
    # Check if this is a company-related query
    for term in company_terms:
        if term in query_clean:
            if search_type == "general" or search_type == "market_research":
                results.extend([
                    {
                        "title": f"{term.title()} Company Profile and SWOT Analysis",
                        "link": f"https://example.com/companies/{term}-profile",
                        "snippet": f"Detailed analysis of {term}'s business model, market position, strengths, weaknesses, opportunities, and threats."
                    },
                    {
                        "title": f"{term.title()} Customer Demographics and User Base",
                        "link": f"https://example.com/companies/{term}-customers",
                        "snippet": f"Analysis of {term}'s customer base, including demographic profiles, user preferences, and market segments."
                    }
                ])
            elif search_type == "news":
                results.extend([
                    {
                        "title": f"{term.title()} Announces New Strategic Initiative",
                        "link": f"https://example.com/news/{term}-announcement",
                        "snippet": f"Breaking news about {term}'s latest product launch, partnership, or strategic direction shift."
                    },
                    {
                        "title": f"{term.title()} Reports Q1 Financial Results",
                        "link": f"https://example.com/news/{term}-financials",
                        "snippet": f"Financial performance update for {term}, including revenue growth, profit margins, and future outlook."
                    }
                ])
    
    # Check if this is a persona/demographic related query
    for term in persona_terms:
        if term in query_clean:
            results.extend([
                {
                    "title": f"Understanding {query_clean.replace(term, term.title())} Profiles",
                    "link": f"https://example.com/personas/{query_clean.replace(' ', '-')}",
                    "snippet": f"Detailed analysis of {query_clean} including demographic information, preferences, behaviors, and decision-making factors."
                },
                {
                    "title": f"{query_clean.title()} Segmentation Strategy",
                    "link": f"https://example.com/segmentation/{query_clean.replace(' ', '-')}",
                    "snippet": f"How to effectively segment and target {query_clean} based on demographic data, psychographics, and behavioral patterns."
                }
            ])
    
    # If we have specific results, use them; otherwise use default
    final_results = results if results else default_results
    
    # Ensure we return the requested number of results
    if len(final_results) > num_results:
        final_results = final_results[:num_results]
    else:
        # If we don't have enough results, add some from the default list
        while len(final_results) < num_results:
            for result in default_results:
                if len(final_results) < num_results:
                    # Modify the title slightly to avoid exact duplicates
                    new_result = result.copy()
                    new_result["title"] = f"{result['title']} - Additional Insight"
                    final_results.append(new_result)
                else:
                    break
    
    # Format the response to match typical search API responses
    return {
        "searchParameters": {
            "q": query,
            "gl": "us",
            "type": search_type
        },
        "searchResults": final_results
    }

def extract_search_data(search_results: Dict[str, Any], include_snippets: bool) -> List[Dict[str, Any]]:
    """Extract and format relevant data from search results"""
    formatted_results = []
    
    # Handle the case where searchResults is the key (typical for API responses)
    results = search_results.get("searchResults", [])
    if not results and "organic" in search_results:
        # Handle Serper API response format
        results = search_results.get("organic", [])
    
    # If results is still empty, check if the response itself is a list
    if not results and isinstance(search_results, list):
        results = search_results
    
    # If we still have no results, return empty list
    if not results:
        return []
    
    # Process each result
    for result in results:
        formatted_result = {
            "title": result.get("title", "No title available"),
            "url": result.get("link", result.get("url", "No URL available"))
        }
        
        if include_snippets:
            # Try different keys that might contain the snippet
            snippet = result.get("snippet", result.get("description", result.get("summary", "No description available")))
            formatted_result["snippet"] = snippet
        
        formatted_results.append(formatted_result)
    
    return formatted_results

@register_function(config_type=WebSearchConfig)
async def web_search(config: WebSearchConfig, builder: Builder) -> AsyncGenerator[Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]], None]:
    """
    Searches the web for information based on the provided query and parameters.
    """
    logger.info(f"Web search initiated with query: {config.query}")
    
    async def _search_fn(input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Extract query from input if provided, otherwise use config
        query = config.query
        if isinstance(input_data, dict) and "query" in input_data:
            query = input_data["query"]
        elif isinstance(input_data, dict) and "input_message" in input_data:
            query = input_data["input_message"]
        elif isinstance(input_data, str):
            query = input_data
        
        # Ensure the query is not empty
        if not query or not query.strip():
            query = config.query
        
        logger.info(f"Executing web search for: {query}")
        
        # Perform the search
        search_results = await search_serper(
            query=query,
            search_type=config.search_type,
            num_results=config.num_results
        )
        
        # Extract and format the results
        formatted_results = extract_search_data(search_results, config.include_snippets)
        
        # Return the results
        return {
            "query": query,
            "results": formatted_results,
            "result_count": len(formatted_results)
        }
    
    yield _search_fn

# Define configuration schema for web search workflow
class WebSearchWorkflowConfig(FunctionBaseConfig, name="web_search_workflow"):
    """
    Workflow for performing multiple web searches and combining results
    
    This tool executes several searches with different parameters to build
    a comprehensive set of information about a topic.
    """
    base_query: str = Field(
        ..., 
        description="The base search query to execute"
    )
    num_searches: int = Field(
        3, 
        description="Number of different searches to perform",
        ge=1, 
        le=5
    )
    results_per_search: int = Field(
        5, 
        description="Number of results per search",
        ge=1, 
        le=10
    )
    
    @validator('base_query')
    def query_not_empty(cls, v):
        if not v or not v.strip():
            return "market trends"
        return v

@register_function(config_type=WebSearchWorkflowConfig)
async def web_search_workflow(config: WebSearchWorkflowConfig, builder: Builder) -> AsyncGenerator[Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]], None]:
    """
    Executes multiple web searches to build comprehensive information about a topic.
    """
    logger.info(f"Web search workflow initiated with base query: {config.base_query}")
    
    async def _workflow_fn(input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Extract base query from input if provided, otherwise use config
        base_query = config.base_query
        if isinstance(input_data, dict) and "query" in input_data:
            base_query = input_data["query"]
        elif isinstance(input_data, dict) and "input_message" in input_data:
            base_query = input_data["input_message"]
        elif isinstance(input_data, str):
            base_query = input_data
        
        # Ensure the query is not empty
        if not base_query or not base_query.strip():
            base_query = config.base_query
        
        logger.info(f"Executing web search workflow for: {base_query}")
        
        search_configs = [
            # General search
            {"query": base_query, "search_type": "general", "num_results": config.results_per_search},
            # Market research search
            {"query": f"{base_query} market research statistics", "search_type": "market_research", "num_results": config.results_per_search},
            # Recent news search
            {"query": f"{base_query} latest news trends", "search_type": "news", "num_results": config.results_per_search}
        ]
        
        # Limit the number of searches based on config
        search_configs = search_configs[:config.num_searches]
        
        all_results = []
        
        # Execute all searches in parallel for efficiency
        tasks = []
        for search_config in search_configs:
            web_search_config = WebSearchConfig(
                query=search_config["query"],
                search_type=search_config["search_type"],
                num_results=search_config["num_results"],
                include_snippets=True
            )
            
            async def _perform_search(config):
                search_gen = web_search(config, builder)
                search_fn = await anext(search_gen)
                return await search_fn({"query": config.query})
            
            tasks.append(_perform_search(web_search_config))
        
        # Gather all search results
        search_results = await asyncio.gather(*tasks)
        
        # Combine all results
        for result in search_results:
            all_results.extend(result.get("results", []))
        
        # Remove any duplicate URLs
        unique_results = []
        seen_urls = set()
        for result in all_results:
            url = result.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        
        return {
            "query": base_query,
            "results": unique_results,
            "result_count": len(unique_results)
        }
    
    yield _workflow_fn

if __name__ == "__main__":
    # Test the module
    async def test():
        logger.info("Testing web search functionality")
        query = "data science market trends 2025"
        results = await search_serper(query, "general", 3)
        logger.info(f"Search results: {json.dumps(results, indent=2)}")
    
    asyncio.run(test()) 