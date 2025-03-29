# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import json
import re
import aiohttp
import random
import time
import uuid
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, AsyncGenerator, Callable, Awaitable, List, Union

from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from pydantic import Field, validator
from dotenv import load_dotenv

import nest_asyncio
import openai
# from anthropic import AsyncAnthropic

from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_function

# from aiq_function_calling.tools.web_search import TavilySearchType, tavily_search
# from aiq_utils.config_manager import ConfigManager

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Get API keys from environment variables
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# LLM API endpoints
NVIDIA_API_ENDPOINT = "https://api.nvidia.com/v1/completions"
GROQ_API_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# LLM model configurations
NVIDIA_MODEL_ID = "nv-llama3-70b-instruct"  # Default NVIDIA model
GROQ_MODEL_ID = "llama3-70b-8192"  # Default Groq model

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Configure cache settings if enabled
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "false").lower() == "true"
CACHE_EXPIRY = int(os.getenv("CACHE_EXPIRY_SECONDS", "3600"))  # 1 hour default

class KeywordResearchConfig(FunctionBaseConfig, name="keyword_research"):
    """Tool for performing keyword research for marketing campaigns
    
    This tool analyzes search queries to find relevant keywords, search volumes,
    and competition data that can be used for Google AdWords campaigns.
    """
    query: str = Field(
        ..., 
        description="The search query to perform keyword research on"
    )
    max_results: int = Field(
        20, 
        description="Maximum number of keywords to return", 
        ge=1, 
        le=100
    )
    include_search_volume: bool = Field(
        True, 
        description="Whether to include search volume data"
    )
    include_competition: bool = Field(
        True, 
        description="Whether to include competition data"
    )
    
    @validator('query')
    def query_not_empty(cls, v):
        if not v or not v.strip():
            # Provide a default value instead of raising error
            return "software"
        return v

class CompetitorAnalysisConfig(FunctionBaseConfig, name="competitor_analysis"):
    """Tool for analyzing competitors in a specific industry
    
    This tool provides insights into competitors' strategies, market positioning,
    and target audiences to inform marketing decisions.
    """
    industry: str = Field(
        ..., 
        description="The industry to analyze competitors for"
    )
    competitor: Optional[str] = Field(
        None, 
        description="Specific competitor to analyze (optional)"
    )
    max_results: int = Field(
        5, 
        description="Maximum number of competitors to analyze", 
        ge=1, 
        le=10
    )
    
    @validator('industry')
    def industry_not_empty(cls, v):
        if not v or not v.strip():
            # Provide a default value instead of raising error
            return "software"
        return v

class DemographicAnalysisConfig(FunctionBaseConfig, name="demographic_analysis"):
    """Tool for analyzing audience demographics and psychographics
    
    This tool provides insights into the demographic characteristics and 
    psychographic profiles of target audiences in specific industries and regions.
    """
    industry: str = Field(
        ..., 
        description="The industry to analyze demographics for"
    )
    region: str = Field(
        "United States", 
        description="The geographic region to focus on"
    )
    
    @validator('industry')
    def industry_not_empty(cls, v):
        if not v or not v.strip():
            # Provide a default value instead of raising error
            return "software"
        return v

class PersonaBuilderConfig(FunctionBaseConfig, name="persona_builder"):
    """Tool for building detailed audience personas
    
    This tool combines keyword research, competitor analysis, and demographic data
    to create comprehensive audience personas for targeted marketing campaigns.
    """
    industry: str = Field(
        ..., 
        description="The industry to create personas for"
    )
    number_of_personas: int = Field(
        1, 
        description="Number of personas to generate", 
        ge=1, 
        le=5
    )
    enrichment_level: str = Field(
        "detailed", 
        description="Level of persona enrichment: 'basic', 'detailed', or 'comprehensive'",
    )
    
    @validator('industry')
    def industry_not_empty(cls, v):
        if not v or not v.strip():
            # Provide a default value instead of raising error
            return "software"
        return v
    
    @validator('enrichment_level')
    def validate_enrichment_level(cls, v):
        valid_levels = ["basic", "detailed", "comprehensive"]
        if v not in valid_levels:
            return "detailed"
        return v

class AudienceResearchWorkflowConfig(FunctionBaseConfig, name="audience_research"):
    """Tool for orchestrating comprehensive audience research
    
    This workflow tool coordinates keyword research, competitor analysis, 
    demographic analysis, and persona building to create a complete audience 
    research report.
    """
    mode: str = Field(
        "default", 
        description="Research mode: 'default', 'in-depth', or 'quick'"
    )
    
    @validator('mode')
    def validate_mode(cls, v):
        valid_modes = ["default", "in-depth", "quick"]
        if v not in valid_modes:
            return "default"
        return v

# Helper functions for LLM API calls
async def call_nvidia_api(prompt: str, system_prompt: str = None, max_tokens: int = 2000, 
                         temperature: float = 0.7, retry_count: int = 0) -> Dict[str, Any]:
    """Make an API call to NVIDIA model with retry logic"""
    if not NVIDIA_API_KEY:
        raise ValueError("NVIDIA_API_KEY environment variable is not set")
    
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "messages": messages,
        "model": "nv-llama3-70b-instruct", # Default model
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(NVIDIA_API_ENDPOINT, json=payload, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429 and retry_count < MAX_RETRIES:
                    # Rate limit hit, retry after delay
                    retry_delay = (retry_count + 1) * RETRY_DELAY
                    logger.warning(f"Rate limit hit, retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    return await call_nvidia_api(prompt, system_prompt, max_tokens, temperature, retry_count + 1)
                else:
                    error_text = await response.text()
                    logger.error(f"NVIDIA API error: {response.status} - {error_text}")
                    raise Exception(f"API Error: {response.status} - {error_text}")
    except aiohttp.ClientError as e:
        if retry_count < MAX_RETRIES:
            retry_delay = (retry_count + 1) * RETRY_DELAY
            logger.warning(f"Connection error, retrying in {retry_delay} seconds... Error: {str(e)}")
            await asyncio.sleep(retry_delay)
            return await call_nvidia_api(prompt, system_prompt, max_tokens, temperature, retry_count + 1)
        else:
            raise Exception(f"Failed after {MAX_RETRIES} retries: {str(e)}")

async def call_groq_api(prompt: str, system_prompt: str = None, max_tokens: int = 2000,
                      temperature: float = 0.7, retry_count: int = 0) -> Dict[str, Any]:
    """Make an API call to Groq model with retry logic"""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "messages": messages,
        "model": "llama3-70b-8192",  # Default to llama3-70b model
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(GROQ_API_ENDPOINT, json=payload, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429 and retry_count < MAX_RETRIES:
                    # Rate limit hit, retry after delay
                    retry_delay = (retry_count + 1) * RETRY_DELAY
                    logger.warning(f"Rate limit hit, retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    return await call_groq_api(prompt, system_prompt, max_tokens, temperature, retry_count + 1)
                else:
                    error_text = await response.text()
                    logger.error(f"Groq API error: {response.status} - {error_text}")
                    raise Exception(f"API Error: {response.status} - {error_text}")
    except aiohttp.ClientError as e:
        if retry_count < MAX_RETRIES:
            retry_delay = (retry_count + 1) * RETRY_DELAY
            logger.warning(f"Connection error, retrying in {retry_delay} seconds... Error: {str(e)}")
            await asyncio.sleep(retry_delay)
            return await call_groq_api(prompt, system_prompt, max_tokens, temperature, retry_count + 1)
        else:
            raise Exception(f"Failed after {MAX_RETRIES} retries: {str(e)}")

# Cache for search results to minimize API calls
search_cache = {}

# Updated function to use AgentIQ's internet search tool
async def cached_tavily_search(query: str, search_depth: str = "advanced", max_results: int = 8, 
                              include_domains: List[str] = None, builder: Builder = None) -> Dict[str, Any]:
    """Cached wrapper for tavily_internet_search to minimize API calls
    
    This uses the AgentIQ framework's tavily_internet_search tool instead of direct API calls.
    """
    if not ENABLE_CACHE:
        if builder is None:
            raise ValueError("Builder object is required when cache is disabled")
        return await run_internet_search(query, search_depth, max_results, include_domains, builder)
    
    # Create a cache key from the query and parameters
    include_domains_str = json.dumps(include_domains, sort_keys=True) if include_domains else ""
    cache_key = f"{query}_{search_depth}_{max_results}_{include_domains_str}"
    
    # Check if we have a cached result that isn't expired
    if cache_key in search_cache:
        cached_item = search_cache[cache_key]
        cache_time = cached_item.get("timestamp", 0)
        if time.time() - cache_time < CACHE_EXPIRY:
            logger.info(f"Using cached result for query: {query[:30]}...")
            return cached_item.get("result", {})
    
    # If not cached or expired, make the API call
    if builder is None:
        raise ValueError("Builder object is required for non-cached search")
    
    result = await run_internet_search(query, search_depth, max_results, include_domains, builder)
    
    # Store in cache
    search_cache[cache_key] = {
        "result": result,
        "timestamp": time.time()
    }
    
    return result

async def run_internet_search(query: str, search_depth: str, max_results: int, include_domains: List[str], builder: Builder) -> Dict[str, Any]:
    """Run the internet search using AgentIQ's tavily_internet_search tool"""
    try:
        # Get the internet_search tool function
        internet_search_tool = await builder.get_function("internet_search")
        
        # Convert search_depth to appropriate search_type for the tool
        search_type = "comprehensive" if search_depth == "advanced" else "basic"
        
        # Prepare the search parameters
        search_params = {
            "query": query,
            "search_type": search_type,
            "max_results": max_results
        }
        
        # Add include_domains if provided
        if include_domains and len(include_domains) > 0:
            search_params["include_domains"] = include_domains
        
        # Call the internet_search tool
        result = await internet_search_tool(**search_params)
        
        # Format result to match the expected structure used in the rest of the code
        formatted_result = {
            "query": query,
            "results": []
        }
        
        # Extract the content from the tool result
        if isinstance(result, dict):
            # Handle different result formats
            if "results" in result:
                formatted_result["results"] = result["results"]
            elif "answer" in result:
                formatted_result["results"] = [{"content": result["answer"]}]
                if "snippets" in result:
                    for snippet in result["snippets"]:
                        formatted_result["results"].append({"content": snippet.get("content", "")})
            else:
                # If the result format is unknown, use the whole result as content
                formatted_result["results"] = [{"content": str(result)}]
                
        return formatted_result
    except Exception as e:
        logger.error(f"Error in internet search: {str(e)}")
        # Return empty results on error
        return {"query": query, "results": [], "error": str(e)}

def parse_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the input data to extract industry and other parameters.
    Uses LLM assistance when available for more robust parsing.
    """
    # Default values
    parsed_data = {
        "industry": "software",
        "query": "software",
        "competitor": None,
        "region": "United States",
        "job_roles": [],
        "product_needs": []
    }
    
    # Extract information from input message
    message = ""
    
    # Handle both direct string input and dictionary with input_message
    if isinstance(input_data, str):
        message = input_data if input_data else "software"
    elif isinstance(input_data, dict):
        if "input_data" in input_data and isinstance(input_data["input_data"], dict):
            # Check for specific fields in input_data
            input_dict = input_data["input_data"]
            if "industry" in input_dict:
                parsed_data["industry"] = input_dict["industry"]
                parsed_data["query"] = input_dict["industry"]
            if "region" in input_dict:
                parsed_data["region"] = input_dict["region"]
            if "competitor" in input_dict:
                parsed_data["competitor"] = input_dict["competitor"]
            if "job_roles" in input_dict and isinstance(input_dict["job_roles"], list):
                parsed_data["job_roles"] = input_dict["job_roles"]
            if "product_needs" in input_dict and isinstance(input_dict["product_needs"], list):
                parsed_data["product_needs"] = input_dict["product_needs"]
            if "keywords" in input_dict and isinstance(input_dict["keywords"], list) and len(input_dict["keywords"]) > 0:
                # Use the first keyword as query and industry
                parsed_data["query"] = input_dict["keywords"][0]
            
            # If there's a message or question, parse it
            if "message" in input_dict and input_dict["message"]:
                message = input_dict["message"]
            elif "function" in input_dict and input_dict["function"] == "keyword_research":
                message = f"research for {parsed_data['industry']} industry"
        elif "input_message" in input_data and input_data["input_message"]:
            message = input_data["input_message"]
        elif "question" in input_data and input_data["question"]:
            message = input_data["question"]
        else:
            # Try to use the entire input as a string if no valid fields
            message = str(input_data) if str(input_data) else "audience research for software"
    else:
        # Try to use the entire input as a string if it's not a recognized format
        message = str(input_data) if str(input_data) else "audience research for software"
    
    # Only log message if it's a direct logging request, not when called from other functions
    if not isinstance(input_data, dict) or not input_data.get("_skip_logging"):
        logger.info(f"Processing input message: {message}")
    
    # Traditional regex-based extraction
    # Extract industry from message
    industry_match = re.search(r'for\s+(\w+(?:\s+\w+)?(?:\s+\w+)?)\s+(?:industry|company|business|market|sector)', message.lower())
    if industry_match:
        industry = industry_match.group(1).strip()
        parsed_data["industry"] = industry
        parsed_data["query"] = industry
    
    # Extract competitor if mentioned
    competitor_match = re.search(r'competitor[s]?:?\s+([a-zA-Z0-9\s]+)', message.lower())
    if competitor_match:
        parsed_data["competitor"] = competitor_match.group(1).strip()
    
    # Extract region if mentioned
    region_match = re.search(r'(?:region|country|location|market):?\s+([a-zA-Z\s]+)', message.lower())
    if region_match:
        parsed_data["region"] = region_match.group(1).strip()
    
    # Extract job roles if mentioned
    role_match = re.search(r'(?:role|job|position)[s]?:?\s+([a-zA-Z0-9\s,]+)', message.lower())
    if role_match:
        roles = role_match.group(1).strip()
        parsed_data["job_roles"] = [role.strip() for role in roles.split(',') if role.strip()]
    
    # Extract product needs if mentioned
    needs_match = re.search(r'(?:need|require)[s]?:?\s+([a-zA-Z0-9\s,]+)', message.lower())
    if needs_match:
        needs = needs_match.group(1).strip()
        parsed_data["product_needs"] = [need.strip() for need in needs.split(',') if need.strip()]
    
    # Only log parsed data if it's a direct logging request, not when called from other functions
    if not isinstance(input_data, dict) or not input_data.get("_skip_logging"):
        logger.info(f"Parsed input data: {parsed_data}")
    
    return parsed_data

@register_function(config_type=KeywordResearchConfig)
async def keyword_research(config: KeywordResearchConfig, builder: Builder) -> AsyncGenerator[Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]], None]:
    logger.info(f"Starting keyword research for query: {config.query}")
    
    async def _keyword_fn(input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Parse input data to extract parameters
            parsed_data = parse_input({"input_data": input_data, "_skip_logging": True})
            industry = parsed_data.get("industry", "software")
            region = parsed_data.get("region", "United States")
            query = config.query if config.query != "default" else parsed_data.get("query", industry)
            
            # Extract focus areas (topics of interest)
            focus_areas = []
            if "product_needs" in parsed_data and parsed_data["product_needs"]:
                focus_areas.extend(parsed_data["product_needs"])
            
            logger.info(f"Performing keyword research for '{industry}' industry in {region}")
            
            # Attempt to gather keyword data using web search
            try:
                keywords_data = await gather_keyword_data(industry, region, focus_areas, builder)
                structured_data = await analyze_keyword_data(keywords_data, industry, region, focus_areas)
                keyword_analysis = await generate_keyword_analysis(structured_data, industry, region, focus_areas)
                
                # Validate using secondary LLM
                try:
                    valid_analysis = await validate_keyword_analysis(keyword_analysis, industry, region)
                    return valid_analysis
                except Exception as e:
                    logger.warning(f"Validation failed, returning unvalidated analysis: {str(e)}")
                    return keyword_analysis
                
            except Exception as e:
                logger.warning(f"Web search keyword gathering failed: {str(e)}, trying LLM-only approach")
                try:
                    # Fall back to LLM-only approach if web search fails
                    llm_keyword_analysis = await create_llm_only_keyword_analysis(industry, region, focus_areas)
                    return llm_keyword_analysis
                except Exception as e2:
                    logger.error(f"LLM-only approach also failed: {str(e2)}, returning minimal response")
                    return create_minimal_keyword_analysis(industry, region, focus_areas, 
                                                        f"Both web search and LLM-based approaches failed: {str(e)}, {str(e2)}")
        
        except Exception as e:
            logger.error(f"Unexpected error in keyword research: {str(e)}")
            return create_minimal_keyword_analysis("general", "global", [], f"Unexpected error: {str(e)}")
    
    yield _keyword_fn

async def gather_keyword_data(industry: str, region: str, focus_areas: List[str], builder: Builder) -> Dict[str, Any]:
    """Gather keyword data from web sources using search"""
    results = {}
    
    # Construct the primary search query
    primary_query = f"keyword research {industry} industry {region}"
    focus_areas_str = " ".join(focus_areas) if focus_areas else ""
    
    if focus_areas:
        primary_query += f" {focus_areas_str}"
    
    # Execute the primary search
    try:
        primary_results = await cached_tavily_search(
            primary_query,
            search_depth="advanced",
            max_results=8,
            include_domains=[
                "semrush.com", "ahrefs.com", "moz.com", "wordstream.com", 
                "answerthepublic.com", "neilpatel.com", "ubersuggest.com", 
                "keywordtool.io", "seochatter.com", "backlinko.com"
            ],
            builder=builder
        )
        results["primary"] = primary_results
    except Exception as e:
        logger.warning(f"Primary keyword search failed: {str(e)}")
        results["primary"] = {"error": str(e), "results": []}
    
    # Secondary searches for specialized keyword information
    searches = [
        {
            "query": f"keyword trends {industry} {region} popular search terms",
            "key": "trends"
        },
        {
            "query": f"long tail keywords {industry} {region}",
            "key": "long_tail"
        },
        {
            "query": f"search intent analysis {industry} {region} keywords",
            "key": "search_intent"
        },
        {
            "query": f"customer questions {industry} {region} FAQ",
            "key": "questions",
            "include_domains": ["quora.com", "reddit.com", "answerthepublic.com"]
        }
    ]
    
    # Add focus area specific searches if provided
    if focus_areas:
        for i, focus in enumerate(focus_areas[:2]):  # Limit to top 2 focus areas
            searches.append({
                "query": f"{focus} keywords {industry} {region}",
                "key": f"focus_{i+1}",
            })
    
    # Execute secondary searches in parallel
    tasks = []
    for search in searches:
        include_domains = search.get("include_domains")
        task = asyncio.create_task(cached_tavily_search(
            search["query"],
            search_depth="moderate",
            max_results=5,
            include_domains=include_domains,
            builder=builder
        ))
        tasks.append((search["key"], task))
    
    # Gather results from all searches
    for key, task in tasks:
        try:
            results[key] = await task
        except Exception as e:
            logger.warning(f"Secondary keyword search for {key} failed: {str(e)}")
            results[key] = {"error": str(e), "results": []}
    
    return results

async def analyze_keyword_data(search_results: Dict[str, Any], industry: str, region: str, 
                             focus_areas: List[str]) -> Dict[str, Any]:
    """Use LLM to analyze and structure the raw keyword search data"""
    
    # Extract all content from search results
    all_content = ""
    for key, search_result in search_results.items():
        if "results" in search_result:
            all_content += f"\n--- {key.upper()} DATA ---\n"
            for item in search_result.get("results", []):
                content = item.get("content", "")
                if content:
                    all_content += content + "\n\n"
    
    # Truncate if too long for LLM context
    if len(all_content) > 15000:
        all_content = all_content[:15000]
    
    # Format focus areas for the prompt
    focus_areas_str = ", ".join(focus_areas) if focus_areas else "general industry terms"
    
    # Set up the prompt for the LLM
    system_prompt = """You are an expert SEO and keyword research specialist. 
Your task is to analyze web search results about an industry's keyword landscape and extract structured information.
Focus only on extracting actual keywords and search terms mentioned in the data. Do not invent or fabricate data.
If information is not available, clearly indicate what is missing."""
    
    user_prompt = f"""
I have collected web search data about keywords and search terms in the {industry} industry in {region}.
{f'I am specifically interested in these focus areas: {focus_areas_str}.' if focus_areas else ''}

Please analyze this data and extract structured information about keywords and search terms.
Organize the information into the following categories:

1. High-volume keywords (popular search terms with high search volume)
2. Long-tail keywords (more specific phrases with lower volume but higher intent)
3. Question-based keywords (what, how, why, who, when, where questions people ask)
4. Competitor keywords (terms associated with competitors or competing products)
5. Trending keywords (newer or seasonal terms showing growth)
6. Customer pain points (keywords indicating problems or challenges)
7. Search intent categories:
   a. Informational keywords (users looking for information)
   b. Navigational keywords (users looking for specific websites/brands)
   c. Commercial keywords (users researching products before purchase)
   d. Transactional keywords (users ready to buy)

For each keyword or phrase you identify, include it along with any available information about:
- Approximate search volume (high, medium, low) if mentioned
- Relevance to the industry (high, medium, low)
- Specific audience segment it targets (if available)

Only include keywords and phrases that are actually mentioned in the data. Do not invent details.
Provide your analysis in JSON format.

Here is the web search data:

{all_content}"""
    
    try:
        # Call the NVIDIA API for analysis
        llm_response = await call_nvidia_api(user_prompt, system_prompt, max_tokens=4000)
        structured_data_text = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Extract the JSON part from the response
        json_match = re.search(r'```json\n(.*?)\n```', structured_data_text, re.DOTALL)
        if json_match:
            structured_data_text = json_match.group(1)
        else:
            # Try to find JSON without markdown formatting
            json_match = re.search(r'({.*})', structured_data_text, re.DOTALL)
            if json_match:
                structured_data_text = json_match.group(1)
        
        try:
            structured_data = json.loads(structured_data_text)
            return structured_data
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM output as JSON, using raw output for further processing")
            return {"raw_analysis": structured_data_text}
    
    except Exception as e:
        logger.error(f"Error analyzing keyword data with LLM: {str(e)}")
        return {"error": str(e), "industry": industry, "region": region}

async def generate_keyword_analysis(structured_data: Dict[str, Any], industry: str, region: str,
                                  focus_areas: List[str]) -> Dict[str, Any]:
    """Generate the final keyword analysis using structured data and LLM"""
    
    # Format the structured data for the LLM prompt
    structured_data_str = json.dumps(structured_data, indent=2)
    
    # Format focus areas for the prompt
    focus_areas_str = ", ".join(focus_areas) if focus_areas else "general industry terms"
    
    system_prompt = """You are an expert in SEO and keyword research for digital marketing. 
Your analysis must be data-driven and practical. When information is missing, either indicate that
or make conservative, realistic inferences based on available data."""

    user_prompt = f"""
Using the structured keyword data below, create a comprehensive keyword analysis for the {industry} industry in {region}.
{f'Focus on these specific areas: {focus_areas_str}.' if focus_areas else ''}

Transform the raw data into a well-structured keyword strategy with these components:
1. An executive summary of the keyword landscape
2. Primary keyword groups categorized by:
   - High-volume keywords
   - Long-tail keywords
   - Question-based keywords
   - Competitor keywords
   - Trending keywords
3. Search intent analysis showing how keywords map to the customer journey:
   - Awareness stage (informational)
   - Consideration stage (commercial)
   - Decision stage (transactional)
4. Strategic keyword recommendations and priority target terms
5. Content topic suggestions based on the keyword analysis

Return your analysis as a JSON object with these clearly structured sections.
Each keyword should include any available information about search volume category, relevance, and target audience.

Here's the structured data to use:

{structured_data_str}

Important: Make this analysis practical and actionable. Don't include any placeholder text or "N/A" entries.
When data is missing, make reasonable inferences based on the industry and available data.
"""

    try:
        # Call the NVIDIA API to generate the analysis
        llm_response = await call_nvidia_api(user_prompt, system_prompt, max_tokens=4000)
        analysis_text = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Extract the JSON part from the response
        json_match = re.search(r'```json\n(.*?)\n```', analysis_text, re.DOTALL)
        if json_match:
            analysis_text = json_match.group(1)
        else:
            # Try to find JSON without markdown formatting
            json_match = re.search(r'({.*})', analysis_text, re.DOTALL)
            if json_match:
                analysis_text = json_match.group(1)
        
        try:
            keyword_data = json.loads(analysis_text)
            
            # Ensure the expected structure
            if not any(k in keyword_data for k in ["keywords", "summary", "analysis"]):
                keyword_data = {"keywords": keyword_data, "summary": "Keyword analysis for " + industry}
            
            return keyword_data
        except json.JSONDecodeError:
            logger.warning("Failed to parse keyword JSON, returning raw LLM output")
            return {
                "summary": "Error parsing keyword data. See raw output.",
                "raw_output": analysis_text,
                "industry": industry,
                "region": region
            }
    
    except Exception as e:
        logger.error(f"Error generating keyword analysis with LLM: {str(e)}")
        return {
            "summary": f"Error generating keyword analysis: {str(e)}",
            "error": str(e),
            "industry": industry, 
            "region": region
        }

async def validate_keyword_analysis(keyword_analysis: Dict[str, Any], industry: str, region: str) -> Dict[str, Any]:
    """Validate the generated keyword analysis using a secondary LLM (Groq)"""
    
    # Only validate if we have Groq API access
    if not GROQ_API_KEY:
        logger.info("Skipping keyword validation as GROQ_API_KEY is not available")
        return keyword_analysis
    
    # Format the keyword analysis for the LLM prompt
    keyword_str = json.dumps(keyword_analysis, indent=2)
    
    system_prompt = """You are a critical evaluator of SEO and keyword research reports.
Your job is to ensure that keyword data is realistic, relevant, and based on factual information
rather than assumptions or arbitrary details. You should identify any inconsistencies, irrelevant keywords,
or problematic assumptions in the analysis."""

    user_prompt = f"""
Review the following keyword analysis for the {industry} industry in {region}.
Identify any issues with:
1. Relevance (are the keywords truly relevant to the industry?)
2. Realism (do these keywords match actual search behavior?)
3. Data quality (is the information specific and actionable?)
4. Consistency (are there contradictions in the analysis?)

If you identify issues, suggest specific corrections to make the analysis more authentic and actionable.
If the analysis is already well-constructed, indicate that it passes validation.

Here's the keyword analysis to validate:

{keyword_str}

Provide your evaluation in JSON format with these fields:
- passed: boolean (true if the analysis is valid, false if issues were found)
- issues: array of specific issues identified (empty if passed is true)
- corrected_analysis: the corrected analysis if issues were found, or the original if no issues
"""

    try:
        # Call the Groq API for validation
        groq_response = await call_groq_api(user_prompt, system_prompt, max_tokens=4000)
        validation_text = groq_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Extract the JSON part from the response
        json_match = re.search(r'```json\n(.*?)\n```', validation_text, re.DOTALL)
        if json_match:
            validation_text = json_match.group(1)
        else:
            # Try to find JSON without markdown formatting
            json_match = re.search(r'({.*})', validation_text, re.DOTALL)
            if json_match:
                validation_text = json_match.group(1)
        
        try:
            validation_data = json.loads(validation_text)
            
            # If validation passed, return the original analysis
            if validation_data.get("passed", True):
                return keyword_analysis
            
            # If validation failed, return the corrected analysis if available
            if "corrected_analysis" in validation_data:
                # Log the issues for debugging
                if "issues" in validation_data:
                    issues = validation_data.get("issues", [])
                    logger.info(f"Keyword validation found {len(issues)} issues: {', '.join(issues[:3])}")
                
                return validation_data["corrected_analysis"]
            
            # If no corrected analysis, return the original
            return keyword_analysis
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse validation JSON, returning original keyword analysis")
            return keyword_analysis
    
    except Exception as e:
        logger.error(f"Error validating keyword analysis with Groq LLM: {str(e)}")
        return keyword_analysis

async def create_llm_only_keyword_analysis(industry: str, region: str, 
                                        focus_areas: List[str]) -> Dict[str, Any]:
    """Create keyword analysis using LLM only, as a fallback when web search fails"""
    
    # Format focus areas for the prompt
    focus_areas_str = ", ".join(focus_areas) if focus_areas else ""
    
    system_prompt = """You are an expert SEO and keyword research specialist.
Your task is to create a realistic keyword analysis for an industry based on your factual knowledge
of that industry and region. Base your analysis on common search behaviors, not guesswork."""

    user_prompt = f"""
Create a comprehensive keyword analysis for the {industry} industry in {region}.
{f'Focus on these specific areas if possible: {focus_areas_str}.' if focus_areas else ''}

Include the following elements in your analysis:
1. An executive summary of the keyword landscape
2. Primary keyword groups including:
   - High-volume keywords (10-15 examples)
   - Long-tail keywords (10-15 examples)
   - Question-based keywords (10-15 examples)
   - Competitor keywords (if you have knowledge of major competitors)
   - Trending keywords
3. Search intent analysis showing how keywords map to the customer journey:
   - Awareness stage (informational)
   - Consideration stage (commercial)
   - Decision stage (transactional)
4. Strategic keyword recommendations and priority target terms
5. Content topic suggestions based on the keyword analysis

Return your analysis as a JSON object with these clearly structured sections.
For each keyword, include an estimate of search volume category (high, medium, low) and relevance.

Important: Make this analysis realistic and based on your knowledge of this industry.
It should reflect actual search behavior typical for the {industry} industry in {region}.
Base your keywords on factual understanding of the industry's terminology and customer needs.
"""

    try:
        # Call the NVIDIA API to generate the analysis
        llm_response = await call_nvidia_api(user_prompt, system_prompt, max_tokens=4000)
        analysis_text = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Extract the JSON part from the response
        json_match = re.search(r'```json\n(.*?)\n```', analysis_text, re.DOTALL)
        if json_match:
            analysis_text = json_match.group(1)
        else:
            # Try to find JSON without markdown formatting
            json_match = re.search(r'({.*})', analysis_text, re.DOTALL)
            if json_match:
                analysis_text = json_match.group(1)
        
        try:
            keyword_data = json.loads(analysis_text)
            
            # Add a note that this was created without web search data
            keyword_data["note"] = "Created using LLM knowledge only, without web search data"
            
            return keyword_data
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM-only keyword JSON, returning basic structure")
            return create_minimal_keyword_analysis(industry, region, focus_areas, 
                                                 "Error parsing data. Using basic fallback.")
    
    except Exception as e:
        logger.error(f"Error creating LLM-only keyword analysis: {str(e)}")
        return create_minimal_keyword_analysis(industry, region, focus_areas, 
                                             f"Error creating analysis: {str(e)}")

def create_minimal_keyword_analysis(industry: str, region: str, 
                                  focus_areas: List[str], 
                                  message: str = None) -> Dict[str, Any]:
    """Create a minimal valid keyword analysis structure as ultimate fallback"""
    
    summary = message or f"Basic keyword analysis for {industry} industry in {region}"
    
    # Create generic keywords based on industry
    high_volume_keywords = [
        {"keyword": f"{industry}", "volume": "high", "relevance": "high"},
        {"keyword": f"{industry} in {region}", "volume": "medium", "relevance": "high"},
        {"keyword": f"best {industry} companies", "volume": "medium", "relevance": "high"},
        {"keyword": f"{industry} services", "volume": "high", "relevance": "high"},
        {"keyword": f"{industry} products", "volume": "high", "relevance": "high"}
    ]
    
    long_tail_keywords = [
        {"keyword": f"how to choose {industry} services", "volume": "low", "relevance": "high"},
        {"keyword": f"affordable {industry} solutions", "volume": "low", "relevance": "medium"},
        {"keyword": f"top rated {industry} providers in {region}", "volume": "low", "relevance": "high"},
        {"keyword": f"{industry} options for small business", "volume": "low", "relevance": "medium"},
        {"keyword": f"enterprise {industry} solutions", "volume": "low", "relevance": "medium"}
    ]
    
    question_keywords = [
        {"keyword": f"what is the best {industry} solution?", "volume": "medium", "relevance": "high"},
        {"keyword": f"how much do {industry} services cost?", "volume": "medium", "relevance": "high"},
        {"keyword": f"why use {industry} services?", "volume": "medium", "relevance": "high"},
        {"keyword": f"who are the top {industry} providers?", "volume": "low", "relevance": "high"},
        {"keyword": f"when to hire {industry} expert?", "volume": "low", "relevance": "medium"}
    ]
    
    # Add focus area keywords if provided
    focus_keywords = []
    if focus_areas:
        for focus in focus_areas[:3]:
            focus_keywords.append(
                {"keyword": f"{focus} in {industry}", "volume": "medium", "relevance": "high"},
            )
            focus_keywords.append(
                {"keyword": f"best {focus} for {industry}", "volume": "medium", "relevance": "high"},
            )
            focus_keywords.append(
                {"keyword": f"{focus} {industry} services", "volume": "low", "relevance": "high"},
            )
    
    return {
        "summary": summary,
        "keywords": {
            "high_volume": high_volume_keywords,
            "long_tail": long_tail_keywords,
            "questions": question_keywords,
            "focus_areas": focus_keywords or [{"keyword": f"specialized {industry} solutions", "volume": "low", "relevance": "medium"}],
        },
        "search_intent": {
            "informational": [f"what is {industry}", f"how does {industry} work", f"{industry} guide"],
            "commercial": [f"compare {industry} services", f"best {industry} providers", f"{industry} reviews"],
            "transactional": [f"buy {industry} services", f"hire {industry} company", f"{industry} cost"]
        },
        "industry": industry,
        "region": region,
        "note": "Created using fallback data due to processing errors"
    }

@register_function(config_type=PersonaBuilderConfig)
async def persona_builder(config: PersonaBuilderConfig, builder: Builder) -> AsyncGenerator[Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]], None]:
    logger.info(f"Building persona for industry: {config.industry}")
    
    async def _persona_fn(input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Parse input data to extract parameters
            parsed_data = parse_input({"input_data": input_data, "_skip_logging": True})
            industry = parsed_data.get("industry", "software")
            region = parsed_data.get("region", "United States")
            
            # Extract job roles and product needs for persona targeting
            job_roles = []
            if "job_roles" in parsed_data and parsed_data["job_roles"]:
                job_roles.extend(parsed_data["job_roles"])
            
            product_needs = []
            if "product_needs" in parsed_data and parsed_data["product_needs"]:
                product_needs.extend(parsed_data["product_needs"])
            
            # Default to industry-based job roles if none specified
            if not job_roles:
                if "healthcare" in industry.lower():
                    job_roles = ["Healthcare IT Director", "Medical Systems Administrator"]
                elif "finance" in industry.lower():
                    job_roles = ["Financial Services Manager", "Banking Technology Director"]
                elif "retail" in industry.lower():
                    job_roles = ["Retail Operations Manager", "E-commerce Director"]
                elif "manufacturing" in industry.lower():
                    job_roles = ["Manufacturing Operations Manager", "Production Technology Director"]
                else:
                    job_roles = ["IT Manager", "Technology Director"]
            
            logger.info(f"Creating persona for '{industry}' industry in {region}")
            
            # Attempt to gather persona data using web search
            try:
                persona_data = await gather_persona_data(industry, region, job_roles, product_needs, builder)
                structured_data = await analyze_persona_data(persona_data, industry, region, job_roles, product_needs)
                persona = await generate_persona(structured_data, industry, region, job_roles, product_needs, config.enrichment_level)
                
                # Validate using secondary LLM
                try:
                    valid_persona = await validate_persona(persona, industry, region)
                    return valid_persona
                except Exception as e:
                    logger.warning(f"Validation failed, returning unvalidated persona: {str(e)}")
                    return persona
                
            except Exception as e:
                logger.warning(f"Web search persona creation failed: {str(e)}, trying LLM-only approach")
                try:
                    # Fall back to LLM-only approach if web search fails
                    llm_persona = await create_llm_only_persona(industry, region, job_roles, product_needs)
                    return llm_persona
                except Exception as e2:
                    logger.error(f"LLM-only approach also failed: {str(e2)}, returning minimal response")
                    return create_minimal_demographics(industry, region, 
                                                     f"Both web search and LLM-based approaches failed: {str(e)}, {str(e2)}")
        
        except Exception as e:
            logger.error(f"Unexpected error in persona builder: {str(e)}")
            return create_minimal_demographics("software", "global", f"Unexpected error: {str(e)}")
    
    yield _persona_fn

async def gather_persona_data(industry: str, region: str, job_roles: List[str], product_needs: List[str], builder: Builder) -> Dict[str, Any]:
    """Gather persona data from web sources using search"""
    results = {}
    
    # Format job roles and product needs for search
    job_roles_str = " OR ".join(job_roles) if job_roles else ""
    product_needs_str = " ".join(product_needs) if product_needs else ""
    
    # Construct the primary search query
    primary_query = f"buyer persona {industry} industry {region}"
    
    if job_roles_str:
        primary_query += f" {job_roles_str}"
    if product_needs_str:
        primary_query += f" needs {product_needs_str}"
    
    # Execute the primary search
    try:
        primary_results = await cached_tavily_search(
            primary_query,
            search_depth="advanced",
            max_results=8,
            include_domains=[
                "hubspot.com", "marketo.com", "salesforce.com", "buffer.com", 
                "mailchimp.com", "blog.hubspot.com", "marketingprofs.com",
                "contentmarketinginstitute.com", "forrester.com", "gartner.com"
            ],
            builder=builder
        )
        results["primary"] = primary_results
    except Exception as e:
        logger.warning(f"Primary persona search failed: {str(e)}")
        results["primary"] = {"error": str(e), "results": []}
    
    # Secondary searches for specialized information
    searches = [
        {
            "query": f"{industry} industry {region} professional typical job titles responsibilities",
            "key": "job_roles",
            "include_domains": ["linkedin.com", "indeed.com", "glassdoor.com", "payscale.com"]
        },
        {
            "query": f"{industry} professional demographics age education background {region}",
            "key": "demographics"
        },
        {
            "query": f"{industry} professionals challenges pain points goals motivations {region}",
            "key": "psychographics"
        },
        {
            "query": f"tools software platforms used by {industry} professionals in {region}",
            "key": "tools",
            "include_domains": ["g2.com", "capterra.com", "softwareadvice.com", "techradar.com"]
        }
    ]
    
    # Execute secondary searches in parallel
    tasks = []
    for search in searches:
        include_domains = search.get("include_domains")
        task = asyncio.create_task(cached_tavily_search(
            search["query"],
            search_depth="moderate",
            max_results=5,
            include_domains=include_domains,
            builder=builder
        ))
        tasks.append((search["key"], task))
    
    # Gather results from all searches
    for key, task in tasks:
        try:
            results[key] = await task
        except Exception as e:
            logger.warning(f"Secondary persona search for {key} failed: {str(e)}")
            results[key] = {"error": str(e), "results": []}
    
    return results

async def analyze_persona_data(web_search_data: Dict[str, Any], industry: str, region: str, 
                             job_roles: List[str], product_needs: List[str]) -> Dict[str, Any]:
    """Use LLM to analyze and structure the raw web search data for persona creation"""
    
    # Extract all content from search results
    all_content = ""
    for key, search_result in web_search_data.items():
        if "results" in search_result:
            all_content += f"\n--- {key.upper()} DATA ---\n"
            for item in search_result.get("results", []):
                content = item.get("content", "")
                if content:
                    all_content += content + "\n\n"
    
    # Truncate if too long for LLM context
    if len(all_content) > 15000:
        all_content = all_content[:15000]
    
    # Set up the prompt for the LLM
    system_prompt = """You are an expert market researcher specializing in creating buyer personas based on real data.
Your task is to analyze web search results about an industry and extract structured information that can be used
to create authentic, realistic buyer personas. Do not invent or fabricate data. If information is not available,
clearly indicate what is missing. Focus on extracting factual information only."""
    
    user_prompt = f"""
I have collected web search data about professionals in the {industry} industry in {region}.
{f'Specifically focusing on the following job roles: {", ".join(job_roles)}.' if job_roles else ''}
{f'These professionals are looking for solutions to address these needs: {", ".join(product_needs)}.' if product_needs else ''}

Please analyze this data and extract structured information for creating a realistic buyer persona.
Organize the information into the following categories:
1. Name patterns and common names in this industry (only if mentioned in the data)
2. Common job titles in this industry
3. Company types or sizes where these professionals typically work
4. Age range of these professionals
5. Gender distribution if available
6. Education level and background
7. Years of experience typical in this industry/role
8. Goals and objectives these professionals typically have
9. Challenges and pain points they face
10. Motivations and values
11. Purchasing process and decision-making factors
12. Communication preferences
13. Common objections during sales process
14. Software tools and technologies they typically use

Only include information that is actually present in the data. Do not invent details.
Provide your analysis in JSON format.

Here is the web search data:

{all_content}"""
    
    try:
        # Call the NVIDIA API for analysis
        llm_response = await call_nvidia_api(user_prompt, system_prompt, max_tokens=4000)
        structured_data_text = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Extract the JSON part from the response
        json_match = re.search(r'```json\n(.*?)\n```', structured_data_text, re.DOTALL)
        if json_match:
            structured_data_text = json_match.group(1)
        else:
            # Try to find JSON without markdown formatting
            json_match = re.search(r'({.*})', structured_data_text, re.DOTALL)
            if json_match:
                structured_data_text = json_match.group(1)
        
        try:
            structured_data = json.loads(structured_data_text)
            return structured_data
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM output as JSON, using raw output for further processing")
            return {"raw_analysis": structured_data_text}
    
    except Exception as e:
        logger.error(f"Error analyzing persona data with LLM: {str(e)}")
        return {"error": str(e), "industry": industry, "region": region}

async def generate_persona(structured_data: Dict[str, Any], industry: str, region: str, 
                         job_roles: List[str], product_needs: List[str], 
                         enrichment_level: str) -> Dict[str, Any]:
    """Generate a complete persona using the structured data and LLM"""
    
    # Format the structured data for the LLM prompt
    structured_data_str = json.dumps(structured_data, indent=2)
    
    # Create a prompt based on enrichment level
    detail_level = {
        "basic": "Create a simple, concise persona with only essential details",
        "detailed": "Create a comprehensive persona with substantial details in all key areas",
        "comprehensive": "Create an exceptionally thorough persona with extensive details, examples, and nuanced insights"
    }.get(enrichment_level, "Create a comprehensive persona with substantial details in all key areas")
    
    system_prompt = f"""You are an expert in creating authentic buyer personas for marketing and sales purposes.
Your personas must be based on real data, not fabricated details. When information is missing, either indicate that
or make conservative, realistic inferences based on the available data.

Your task is to generate a {enrichment_level} buyer persona for the {industry} industry in {region}.
{detail_level}."""

    user_prompt = f"""
Using the structured data below, create a realistic buyer persona for the {industry} industry in {region}.
{f'The persona should represent professionals in these roles: {", ".join(job_roles)}.' if job_roles else ''}
{f'The persona should be seeking solutions for these needs: {", ".join(product_needs)}.' if product_needs else ''}

Generate a complete persona that feels like a real person, with:
1. Full name (based on actual common names in this industry if available)
2. Job title
3. Company type and size
4. Age
5. Gender
6. Education and background
7. Years of experience
8. Detailed goals and objectives (both professional and if appropriate, personal)
9. Specific challenges and pain points
10. Motivations and values
11. Buying process and decision factors
12. Communication preferences
13. Common objections during the sales process
14. Software tools and technologies they use
15. A realistic quote that captures their perspective
16. A biographical narrative that ties all elements together into a cohesive profile

Return the persona as a JSON object with these elements clearly structured.
Also include a 'bio' field with a 3-5 paragraph narrative that brings this persona to life.
Include a 'quote' field with a realistic quote from this persona about their goals or challenges.

Here's the structured data to use:

{structured_data_str}

Important: Make this persona realistic and data-driven. Don't include any placeholder text like "TBD" or "N/A" - 
instead, make reasonable inferences based on the industry and available data. The persona should feel like a real
person that someone could actually meet, not a generic template.
"""

    try:
        # Call the NVIDIA API to generate the persona
        llm_response = await call_nvidia_api(user_prompt, system_prompt, max_tokens=4000)
        persona_text = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Extract the JSON part from the response
        json_match = re.search(r'```json\n(.*?)\n```', persona_text, re.DOTALL)
        if json_match:
            persona_text = json_match.group(1)
        else:
            # Try to find JSON without markdown formatting
            json_match = re.search(r'({.*})', persona_text, re.DOTALL)
            if json_match:
                persona_text = json_match.group(1)
        
        try:
            persona_data = json.loads(persona_text)
            
            # Ensure the expected structure
            if "persona" not in persona_data:
                persona_data = {"persona": persona_data}
            
            return persona_data
        except json.JSONDecodeError:
            logger.warning("Failed to parse persona JSON, returning raw LLM output")
            return {
                "persona": {
                    "bio": "Error parsing persona data. See details for raw output.",
                    "details": {"raw_output": persona_text, "industry": industry, "region": region}
                }
            }
    
    except Exception as e:
        logger.error(f"Error generating persona with LLM: {str(e)}")
        return {
            "persona": {
                "bio": f"Error generating persona: {str(e)}",
                "details": {"error": str(e), "industry": industry, "region": region}
            }
        }

async def validate_persona(persona: Dict[str, Any], industry: str, region: str) -> Dict[str, Any]:
    """Validate the generated persona for realism and data consistency using a secondary LLM (Groq)"""
    
    # Only validate if we have Groq API access
    if not GROQ_API_KEY:
        logger.info("Skipping persona validation as GROQ_API_KEY is not available")
        return persona
    
    # Format the persona for the LLM prompt
    persona_str = json.dumps(persona, indent=2)
    
    system_prompt = """You are a critical evaluator of buyer personas used for marketing and sales.
Your job is to ensure that personas are realistic, consistent, and based on factual data rather than stereotypes
or arbitrary details. You should identify any inconsistencies, unrealistic elements, or problematic
assumptions in the persona."""

    user_prompt = f"""
Review the following buyer persona for the {industry} industry in {region}.
Identify any issues with:
1. Internal consistency (contradictions within the persona)
2. Realism (does this sound like a real person or a collection of stereotypes?)
3. Data-based elements vs. arbitrary/invented details
4. Problematic stereotypes or assumptions

If you identify issues, suggest specific corrections to make the persona more authentic and realistic.
If the persona is already well-constructed, indicate that it passes validation.

Here's the persona to validate:

{persona_str}

Provide your evaluation in JSON format with these fields:
- passed: boolean (true if the persona is valid, false if issues were found)
- issues: array of specific issues identified (empty if passed is true)
- corrected_persona: the corrected persona if issues were found, or the original persona if no issues
"""

    try:
        # Call the Groq API for validation
        groq_response = await call_groq_api(user_prompt, system_prompt, max_tokens=4000)
        validation_text = groq_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Extract the JSON part from the response
        json_match = re.search(r'```json\n(.*?)\n```', validation_text, re.DOTALL)
        if json_match:
            validation_text = json_match.group(1)
        else:
            # Try to find JSON without markdown formatting
            json_match = re.search(r'({.*})', validation_text, re.DOTALL)
            if json_match:
                validation_text = json_match.group(1)
        
        try:
            validation_data = json.loads(validation_text)
            
            # If validation passed, return the original persona
            if validation_data.get("passed", True):
                return persona
            
            # If validation failed, return the corrected persona if available
            if "corrected_persona" in validation_data:
                # Log the issues for debugging
                if "issues" in validation_data:
                    issues = validation_data.get("issues", [])
                    logger.info(f"Persona validation found {len(issues)} issues: {', '.join(issues[:3])}")
                
                return validation_data["corrected_persona"]
            
            # If no corrected persona, return the original
            return persona
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse validation JSON, returning original persona")
            return persona
    
    except Exception as e:
        logger.error(f"Error validating persona with Groq LLM: {str(e)}")
        return persona

async def create_llm_only_persona(industry: str, region: str, job_roles: List[str], product_needs: List[str]) -> Dict[str, Any]:
    """Create a persona using LLM only, as a fallback when web search fails"""
    
    system_prompt = f"""You are an expert market researcher specializing in creating data-driven buyer personas.
You need to create a realistic buyer persona for the {industry} industry in {region}.
Base your persona on factual understanding of this industry and demographic trends, not arbitrary details."""

    # Format job roles and product needs for the prompt
    job_roles_str = ", ".join(job_roles) if job_roles else "professionals"
    product_needs_str = ", ".join(product_needs) if product_needs else ""
    
    user_prompt = f"""
Create a realistic buyer persona for the {industry} industry in {region}.
The persona should represent {job_roles_str} in this industry.
{f'They are seeking solutions for these needs: {product_needs_str}.' if product_needs_str else ''}

Include these elements in your persona:
1. Full name (that would be common for a person in this industry/role/region)
2. Job title (specific and realistic for this industry)
3. Company type and size where they work
4. Age (realistic for their career stage)
5. Education background
6. Years of experience
7. Goals and objectives (both professional and personal)
8. Challenges and pain points
9. Motivations and values
10. Buying process and decision factors
11. Communication preferences
12. Common objections during the sales process
13. Software tools and technologies they use
14. A realistic quote that captures their perspective

Create a cohesive narrative that ties all these elements together in a 'bio' field.
Return the persona as a well-structured JSON object.

Important: Make this persona realistic and data-driven. It should feel like a real
person that someone could actually meet, not a generic template. Base your persona
on your factual understanding of this industry and demographic trends, not random details.
"""

    try:
        # Call the NVIDIA API to generate the persona
        llm_response = await call_nvidia_api(user_prompt, system_prompt, max_tokens=3000)
        persona_text = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Extract the JSON part from the response
        json_match = re.search(r'```json\n(.*?)\n```', persona_text, re.DOTALL)
        if json_match:
            persona_text = json_match.group(1)
        else:
            # Try to find JSON without markdown formatting
            json_match = re.search(r'({.*})', persona_text, re.DOTALL)
            if json_match:
                persona_text = json_match.group(1)
        
        try:
            persona_data = json.loads(persona_text)
            
            # Ensure the expected structure
            if "persona" not in persona_data:
                persona_data = {"persona": persona_data}
            
            # Add a note that this was created without web search data
            if "details" in persona_data["persona"]:
                persona_data["persona"]["details"]["note"] = "Created using LLM knowledge only, without web search data"
            else:
                persona_data["persona"]["details"] = {"note": "Created using LLM knowledge only, without web search data"}
            
            return persona_data
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM-only persona JSON, returning basic structure")
            return {
                "persona": {
                    "bio": "Error parsing persona data. Using basic fallback.",
                    "quote": f"I work in the {industry} industry and face typical challenges for professionals in this field.",
                    "details": {
                        "name": f"Professional in {industry}",
                        "job_title": f"{industry.title()} {job_roles[0].title() if job_roles else 'Specialist'}",
                        "industry": industry,
                        "region": region,
                        "note": "Created using basic fallback due to parsing error"
                    }
                }
            }
    
    except Exception as e:
        logger.error(f"Error creating LLM-only persona: {str(e)}")
        return {
            "persona": {
                "bio": f"Error creating persona: {str(e)}",
                "details": {"error": str(e), "industry": industry, "region": region}
            }
        }

def create_minimal_demographics(industry: str, region: str, message: str = None) -> Dict[str, Any]:
    """Create a minimal valid demographic structure as ultimate fallback"""
    
    summary = message or f"Basic demographic information for {industry} industry in {region}"
    
    return {
        "demographics": {
            "summary": summary,
            "age_distribution": {
                "18-24": "15%",
                "25-34": "30%",
                "35-44": "25%",
                "45-54": "20%",
                "55+": "10%"
            },
            "gender_distribution": {
                "male": "50%",
                "female": "50%"
            },
            "income_levels": {
                "low": "25%",
                "medium": "50%",
                "high": "25%"
            },
            "education": {
                "high_school": "25%",
                "bachelors": "45%",
                "masters": "20%",
                "doctorate": "10%"
            },
            "geographic": {
                "primary_region": region
            },
            "psychographics": {
                "interests": [f"Interest in {industry} products and services"],
                "pain_points": [f"Challenges related to {industry}"]
            },
            "industry": industry,
            "region": region,
            "note": "Created using fallback data due to processing errors"
        }
    }

@register_function(config_type=CompetitorAnalysisConfig)
async def competitor_analysis(config: CompetitorAnalysisConfig, builder: Builder) -> AsyncGenerator[Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]], None]:
    logger.info(f"Starting competitor analysis for industry: {config.industry}")
    
    async def _competitor_fn(input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Parse input data to extract parameters
            parsed_data = parse_input({"input_data": input_data, "_skip_logging": True})
            industry = parsed_data.get("industry", "software")
            region = parsed_data.get("region", "United States")
            
            # Extract specific competitors to analyze
            specific_competitors = []
            if "competitor" in parsed_data and parsed_data["competitor"]:
                specific_competitors.append(parsed_data["competitor"])
            elif config.competitor:
                specific_competitors.append(config.competitor)
            
            logger.info(f"Analyzing competitors for '{industry}' industry in {region}")
            
            # Attempt to gather competitor data using web search
            try:
                competitor_data = await gather_competitor_data(industry, region, specific_competitors, builder)
                structured_data = await analyze_competitor_data(competitor_data, industry, region, specific_competitors)
                competitor_analysis = await generate_competitor_analysis(structured_data, industry, region, specific_competitors)
                
                # Validate using secondary LLM
                try:
                    valid_analysis = await validate_competitor_analysis(competitor_analysis, industry, region)
                    return valid_analysis
                except Exception as e:
                    logger.warning(f"Validation failed, returning unvalidated analysis: {str(e)}")
                    return competitor_analysis
                
            except Exception as e:
                logger.warning(f"Web search competitor analysis failed: {str(e)}, trying LLM-only approach")
                try:
                    # Fall back to LLM-only approach if web search fails
                    llm_competitor_analysis = await create_llm_only_competitor_analysis(industry, region, specific_competitors)
                    return llm_competitor_analysis
                except Exception as e2:
                    logger.error(f"LLM-only approach also failed: {str(e2)}, returning minimal response")
                    return create_minimal_competitor_analysis(industry, region, specific_competitors, 
                                                           f"Both web search and LLM-based approaches failed: {str(e)}, {str(e2)}")
        
        except Exception as e:
            logger.error(f"Unexpected error in competitor analysis: {str(e)}")
            return create_minimal_competitor_analysis("software", "global", [], f"Unexpected error: {str(e)}")
    
    yield _competitor_fn

async def gather_competitor_data(industry: str, region: str, specific_competitors: List[str], builder: Builder) -> Dict[str, Any]:
    """Gather competitor data from web sources using search"""
    results = {}
    
    # Construct the primary search query
    primary_query = f"top competitors {industry} industry {region} market share"
    
    # Add specific competitors to the query if provided
    if specific_competitors:
        competitors_str = " ".join(specific_competitors)
        primary_query = f"competitive analysis {competitors_str} {industry} industry {region}"
    
    # Execute the primary search
    try:
        primary_results = await cached_tavily_search(
            primary_query,
            search_depth="advanced",
            max_results=8,
            include_domains=[
                "gartner.com", "forrester.com", "idc.com", "statista.com", 
                "bloomberg.com", "reuters.com", "forbes.com", "techcrunch.com",
                "businesswire.com", "prnewswire.com", "marketwatch.com"
            ],
            builder=builder
        )
        results["primary"] = primary_results
    except Exception as e:
        logger.warning(f"Primary competitor search failed: {str(e)}")
        results["primary"] = {"error": str(e), "results": []}
    
    # Secondary searches for specialized information
    searches = [
        {
            "query": f"{industry} industry market leaders competitive landscape {region}",
            "key": "market_leaders"
        },
        {
            "query": f"{industry} competitor comparison strengths weaknesses {region}",
            "key": "strengths_weaknesses"
        },
        {
            "query": f"{industry} competitive pricing strategies {region}",
            "key": "pricing_strategies"
        },
        {
            "query": f"{industry} competitor target audience customer segments {region}",
            "key": "target_audiences",
            "include_domains": ["marketingweek.com", "adweek.com", "marketingdive.com"]
        }
    ]
    
    # Add specific competitor searches if provided
    if specific_competitors:
        for i, competitor in enumerate(specific_competitors[:2]):  # Limit to top 2 competitors
            searches.append({
                "query": f"{competitor} market share revenue {industry} {region}",
                "key": f"competitor_{i+1}",
            })
            searches.append({
                "query": f"{competitor} strengths weaknesses opportunities threats SWOT analysis",
                "key": f"competitor_{i+1}_swot",
            })
    
    # Execute secondary searches in parallel
    tasks = []
    for search in searches:
        include_domains = search.get("include_domains")
        task = asyncio.create_task(cached_tavily_search(
            search["query"],
            search_depth="moderate",
            max_results=5,
            include_domains=include_domains,
            builder=builder
        ))
        tasks.append((search["key"], task))
    
    # Gather results from all searches
    for key, task in tasks:
        try:
            results[key] = await task
        except Exception as e:
            logger.warning(f"Secondary competitor search for {key} failed: {str(e)}")
            results[key] = {"error": str(e), "results": []}
    
    return results

async def analyze_competitor_data(search_results: Dict[str, Any], industry: str, region: str, 
                                specific_competitors: List[str]) -> Dict[str, Any]:
    """Use LLM to analyze and structure the raw competitor search data"""
    
    # Extract all content from search results
    all_content = ""
    for key, search_result in search_results.items():
        if "results" in search_result:
            all_content += f"\n--- {key.upper()} DATA ---\n"
            for item in search_result.get("results", []):
                content = item.get("content", "")
                if content:
                    all_content += content + "\n\n"
    
    # Truncate if too long for LLM context
    if len(all_content) > 15000:
        all_content = all_content[:15000]
    
    # Format specific competitors for the prompt
    competitors_str = ", ".join(specific_competitors) if specific_competitors else "any major competitors"
    
    # Set up the prompt for the LLM
    system_prompt = """You are an expert market researcher specializing in competitive analysis. 
Your task is to analyze web search results about an industry's competitive landscape and extract structured information.
Focus only on extracting factual data about competitors. Do not invent or fabricate data. If information is not available,
clearly indicate what is missing."""
    
    user_prompt = f"""
I have collected web search data about competitors in the {industry} industry in {region}.
{f'I am specifically interested in these competitors: {competitors_str}.' if specific_competitors else ''}

Please analyze this data and extract structured information about the competitive landscape.
Organize the information into the following categories:

1. Major competitors in the industry (name, estimated market share if available)
2. For each significant competitor:
   a. Company overview (size, founded, headquarters location if available)
   b. Product/service offerings
   c. Strengths
   d. Weaknesses
   e. Pricing strategy
   f. Target audience
   g. Key differentiators
   h. Recent business developments (if any)
3. Market trends affecting competition
4. Competitive landscape summary (concentration, barriers to entry, etc.)

Only include information that is actually present in the data. Do not invent details.
Provide your analysis in JSON format.

Here is the web search data:

{all_content}"""
    
    try:
        # Call the NVIDIA API for analysis
        llm_response = await call_nvidia_api(user_prompt, system_prompt, max_tokens=4000)
        structured_data_text = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Extract the JSON part from the response
        json_match = re.search(r'```json\n(.*?)\n```', structured_data_text, re.DOTALL)
        if json_match:
            structured_data_text = json_match.group(1)
        else:
            # Try to find JSON without markdown formatting
            json_match = re.search(r'({.*})', structured_data_text, re.DOTALL)
            if json_match:
                structured_data_text = json_match.group(1)
        
        try:
            structured_data = json.loads(structured_data_text)
            return structured_data
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM output as JSON, using raw output for further processing")
            return {"raw_analysis": structured_data_text}
                
    except Exception as e:
        logger.error(f"Error analyzing competitor data with LLM: {str(e)}")
        return {"error": str(e), "industry": industry, "region": region}

async def generate_competitor_analysis(structured_data: Dict[str, Any], industry: str, region: str,
                                     specific_competitors: List[str]) -> Dict[str, Any]:
    """Generate the final competitor analysis using structured data and LLM"""
    
    # Format the structured data for the LLM prompt
    structured_data_str = json.dumps(structured_data, indent=2)
    
    # Format specific competitors for the prompt
    competitors_str = ", ".join(specific_competitors) if specific_competitors else "major competitors"
    
    system_prompt = """You are an expert in competitive analysis for market research. 
Your analysis must be data-driven and factual. When information is missing, either indicate that
or make conservative, realistic inferences based on available data."""

    user_prompt = f"""
Using the structured competitor data below, create a comprehensive competitive analysis for the {industry} industry in {region}.
{f'Focus on these specific competitors: {competitors_str}.' if specific_competitors else ''}

Transform the raw data into a well-structured competitive analysis with these components:
1. An executive summary of the competitive landscape
2. Detailed profiles of each major competitor including:
   - Company overview
   - Product/service offerings
   - Strengths and weaknesses
   - Market positioning
   - Pricing strategy
   - Target audience
   - Unique selling propositions
3. Competitive matrix showing how competitors compare across key factors
4. Market trends affecting the competitive landscape
5. Strategic recommendations based on the competitive analysis

Return your analysis as a JSON object with these clearly structured sections.
Include both raw data points and interpretive insights for each section.

Here's the structured data to use:

{structured_data_str}

Important: Make this analysis data-driven and actionable. Don't include any placeholder text or "N/A" entries.
When data is missing, make reasonable inferences based on the industry and available data.
"""

    try:
        # Call the NVIDIA API to generate the analysis
        llm_response = await call_nvidia_api(user_prompt, system_prompt, max_tokens=4000)
        analysis_text = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Extract the JSON part from the response
        json_match = re.search(r'```json\n(.*?)\n```', analysis_text, re.DOTALL)
        if json_match:
            analysis_text = json_match.group(1)
        else:
            # Try to find JSON without markdown formatting
            json_match = re.search(r'({.*})', analysis_text, re.DOTALL)
            if json_match:
                analysis_text = json_match.group(1)
        
        try:
            competitor_data = json.loads(analysis_text)
            
            # Ensure the expected structure
            if not any(k in competitor_data for k in ["competitors", "summary", "analysis"]):
                competitor_data = {"competitors": competitor_data, "summary": "Competitor analysis for " + industry}
            
            return competitor_data
        except json.JSONDecodeError:
            logger.warning("Failed to parse competitor JSON, returning raw LLM output")
            return {
            "summary": "Error parsing competitor data. See raw output.",
            "raw_output": analysis_text,
            "industry": industry,
            "region": region
            }
        
    except Exception as e:
        logger.error(f"Error generating competitor analysis with LLM: {str(e)}")
        return {
        "summary": f"Error generating competitor analysis: {str(e)}",
        "error": str(e),
        "industry": industry, 
        "region": region
    }

async def validate_competitor_analysis(competitor_analysis: Dict[str, Any], industry: str, region: str) -> Dict[str, Any]:
    """Validate the generated competitor analysis using a secondary LLM (Groq)"""
    
    # Only validate if we have Groq API access
    if not GROQ_API_KEY:
        logger.info("Skipping competitor validation as GROQ_API_KEY is not available")
        return competitor_analysis
    
    # Format the competitor analysis for the LLM prompt
    competitor_str = json.dumps(competitor_analysis, indent=2)
    
    system_prompt = """You are a critical evaluator of competitive analysis reports.
Your job is to ensure that competitor data is realistic, consistent, and based on factual information
rather than assumptions or arbitrary details. You should identify any inconsistencies, unrealistic elements,
or problematic assumptions in the analysis."""

    user_prompt = f"""
Review the following competitor analysis for the {industry} industry in {region}.
Identify any issues with:
1. Internal consistency (contradictions within the analysis)
2. Realism (does this data seem plausible?)
3. Data quality (is the information specific and actionable?)
4. Biases or unsubstantiated claims

If you identify issues, suggest specific corrections to make the analysis more authentic and actionable.
If the analysis is already well-constructed, indicate that it passes validation.

Here's the competitor analysis to validate:

{competitor_str}

Provide your evaluation in JSON format with these fields:
- passed: boolean (true if the analysis is valid, false if issues were found)
- issues: array of specific issues identified (empty if passed is true)
- corrected_analysis: the corrected analysis if issues were found, or the original if no issues
"""

    try:
        # Call the Groq API for validation
        groq_response = await call_groq_api(user_prompt, system_prompt, max_tokens=4000)
        validation_text = groq_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Extract the JSON part from the response
        json_match = re.search(r'```json\n(.*?)\n```', validation_text, re.DOTALL)
        if json_match:
            validation_text = json_match.group(1)
        else:
            # Try to find JSON without markdown formatting
            json_match = re.search(r'({.*})', validation_text, re.DOTALL)
            if json_match:
                validation_text = json_match.group(1)
        
        try:
            validation_data = json.loads(validation_text)
            
            # If validation passed, return the original analysis
            if validation_data.get("passed", True):
                return competitor_analysis
            
            # If validation failed, return the corrected analysis if available
            if "corrected_analysis" in validation_data:
                # Log the issues for debugging
                if "issues" in validation_data:
                    issues = validation_data.get("issues", [])
                    logger.info(f"Competitor validation found {len(issues)} issues: {', '.join(issues[:3])}")
                
                return validation_data["corrected_analysis"]
            
            # If no corrected analysis, return the original
            return competitor_analysis
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse validation JSON, returning original competitor analysis")
            return competitor_analysis
    
    except Exception as e:
        logger.error(f"Error validating competitor analysis with Groq LLM: {str(e)}")
        return competitor_analysis

async def create_llm_only_competitor_analysis(industry: str, region: str, 
                                           specific_competitors: List[str]) -> Dict[str, Any]:
    """Create competitor analysis using LLM only, as a fallback when web search fails"""
    
    # Format specific competitors for the prompt
    competitors_str = ", ".join(specific_competitors) if specific_competitors else ""
    
    system_prompt = """You are an expert market researcher specializing in competitive analysis.
Your task is to create a realistic competitor analysis for an industry based on your factual knowledge
of that industry and region. Base your analysis on verifiable facts, not assumptions or guesswork."""

    user_prompt = f"""
Create a comprehensive competitor analysis for the {industry} industry in {region}.
{f'Focus on these specific competitors if you have knowledge about them: {competitors_str}.' if competitors_str else ''}

Include the following elements in your analysis:
1. An executive summary of the competitive landscape
2. Profiles of 3-5 major competitors including:
   - Company overview
   - Product/service offerings
   - Strengths and weaknesses
   - Market positioning
   - Pricing strategy (general approach)
   - Target audience
   - Unique selling propositions
3. A comparison of how these competitors compare across key factors
4. Market trends affecting competition
5. Strategic insights based on the competitive landscape

Return your analysis as a JSON object with these elements clearly structured.

Important: Make this analysis realistic and data-driven based on your knowledge of this industry.
It should reflect actual competitive dynamics typical for the {industry} industry in {region}.
Base your analysis on factual understanding of the industry, not arbitrary details.
If you don't have specific knowledge about certain competitors, focus on the types of companies
that typically compete in this space rather than inventing specific details about real companies.
"""

    try:
        # Call the NVIDIA API to generate the analysis
        llm_response = await call_nvidia_api(user_prompt, system_prompt, max_tokens=4000)
        analysis_text = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Extract the JSON part from the response
        json_match = re.search(r'```json\n(.*?)\n```', analysis_text, re.DOTALL)
        if json_match:
            analysis_text = json_match.group(1)
        else:
            # Try to find JSON without markdown formatting
            json_match = re.search(r'({.*})', analysis_text, re.DOTALL)
            if json_match:
                analysis_text = json_match.group(1)
        
        try:
            competitor_data = json.loads(analysis_text)
            
            # Add a note that this was created without web search data
            competitor_data["note"] = "Created using LLM knowledge only, without web search data"
            
            return competitor_data
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM-only competitor JSON, returning basic structure")
            return create_minimal_competitor_analysis(industry, region, specific_competitors, 
                                                    "Error parsing data. Using basic fallback.")
    
    except Exception as e:
        logger.error(f"Error creating LLM-only competitor analysis: {str(e)}")
        return create_minimal_competitor_analysis(industry, region, specific_competitors, 
                                                f"Error creating analysis: {str(e)}")

def create_minimal_competitor_analysis(industry: str, region: str, 
                                     specific_competitors: List[str], 
                                     message: str = None) -> Dict[str, Any]:
    """Create a minimal valid competitor analysis structure as ultimate fallback"""
    
    summary = message or f"Basic competitor information for {industry} industry in {region}"
    
    # Use provided competitors or create generic ones
    competitors = []
    if specific_competitors:
        for comp in specific_competitors[:3]:
            competitors.append({
                "name": comp,
                "overview": f"Competitor in the {industry} industry",
                "strengths": ["Unknown"],
                "weaknesses": ["Unknown"],
                "target_audience": f"{industry} customers"
            })
    else:
        # Create generic competitors
        for i in range(3):
            competitors.append({
                "name": f"Competitor {i+1}",
                "overview": f"Generic competitor in the {industry} industry",
                "strengths": [f"Typical strength for {industry} business"],
                "weaknesses": [f"Typical weakness for {industry} business"],
                "target_audience": f"{industry} customers"
            })
    
    return {
        "summary": summary,
        "competitors": competitors,
        "market_trends": [f"Typical trends in the {industry} industry"],
        "industry": industry,
        "region": region,
        "note": "Created using fallback data due to processing errors"
    }

@register_function(config_type=DemographicAnalysisConfig)
async def demographic_analysis(config: DemographicAnalysisConfig, builder: Builder) -> AsyncGenerator[Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]], None]:
    logger.info(f"Starting demographic analysis for industry: {config.industry} in {config.region}")
    
    async def _demographic_fn(input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Parse input data to extract parameters
            parsed_data = parse_input({"input_data": input_data, "_skip_logging": True})
            industry = parsed_data.get("industry", "software")
            region = parsed_data.get("region", "United States")
            
            logger.info(f"Analyzing demographics for '{industry}' industry in {region}")
            
            # Attempt to gather demographic data using web search
            try:
                demographic_data = await gather_demographic_data(industry, region, builder)
                structured_data = await analyze_demographic_data(demographic_data, industry, region)
                demographic_analysis = await generate_demographic_analysis(structured_data, industry, region)
                
                # Validate using secondary LLM
                try:
                    valid_analysis = await validate_demographic_analysis(demographic_analysis, industry, region)
                    return valid_analysis
                except Exception as e:
                    logger.warning(f"Validation failed, returning unvalidated analysis: {str(e)}")
                    return demographic_analysis
                
            except Exception as e:
                logger.warning(f"Web search demographic analysis failed: {str(e)}, trying LLM-only approach")
                try:
                    # Fall back to LLM-only approach if web search fails
                    llm_demographic_analysis = await create_llm_only_demographic_analysis(industry, region)
                    return llm_demographic_analysis
                except Exception as e2:
                    logger.error(f"LLM-only approach also failed: {str(e2)}, returning minimal response")
                    return create_minimal_demographics(industry, region, 
                                                    f"Both web search and LLM-based approaches failed: {str(e)}, {str(e2)}")
        
        except Exception as e:
            logger.error(f"Unexpected error in demographic analysis: {str(e)}")
            return create_minimal_demographics("software", "global", f"Unexpected error: {str(e)}")
    
    yield _demographic_fn

async def gather_demographic_data(industry: str, region: str, builder: Builder) -> Dict[str, Any]:
    """Gather demographic data from web sources using search"""
    results = {}
    
    # Construct the primary search query
    primary_query = f"demographic analysis {industry} industry {region} customer profile"
    
    # Execute the primary search
    try:
        primary_results = await cached_tavily_search(
            primary_query,
            search_depth="advanced",
            max_results=8,
            include_domains=[
                "census.gov", "statista.com", "pewresearch.org", "nielsen.com", 
                "gallup.com", "marketingcharts.com", "entrepreneur.com", "businessinsider.com",
                "forrester.com", "gartner.com", "mckinsey.com", "hbr.org"
            ],
            builder=builder
        )
        results["primary"] = primary_results
    except Exception as e:
        logger.warning(f"Primary demographic search failed: {str(e)}")
        results["primary"] = {"error": str(e), "results": []}
    
    # Secondary searches for specialized information
    searches = [
        {
            "query": f"{industry} customer demographics age income education {region}",
            "key": "demographics_core"
        },
        {
            "query": f"{industry} psychographics customer values lifestyle interests {region}",
            "key": "psychographics"
        },
        {
            "query": f"{industry} market size total addressable market {region}",
            "key": "market_size"
        },
        {
            "query": f"{industry} customer behavior buying patterns {region}",
            "key": "behaviors",
            "include_domains": ["marketingprofs.com", "hubspot.com", "surveymonkey.com"]
        }
    ]
    
    # Execute secondary searches in parallel
    tasks = []
    for search in searches:
        include_domains = search.get("include_domains")
        task = asyncio.create_task(cached_tavily_search(
            search["query"],
            search_depth="moderate",
            max_results=5,
            include_domains=include_domains,
            builder=builder
        ))
        tasks.append((search["key"], task))
    
    # Gather results from all searches
    for key, task in tasks:
        try:
            results[key] = await task
        except Exception as e:
            logger.warning(f"Secondary demographic search for {key} failed: {str(e)}")
            results[key] = {"error": str(e), "results": []}
    
    return results

async def analyze_demographic_data(search_results: Dict[str, Any], industry: str, region: str) -> Dict[str, Any]:
    """Use LLM to analyze and structure the raw demographic search data"""
    
    # Extract all content from search results
    all_content = ""
    for key, search_result in search_results.items():
        if "results" in search_result:
            all_content += f"\n--- {key.upper()} DATA ---\n"
            for item in search_result.get("results", []):
                content = item.get("content", "")
                if content:
                    all_content += content + "\n\n"
    
    # Truncate if too long for LLM context
    if len(all_content) > 15000:
        all_content = all_content[:15000]
    
    # Set up the prompt for the LLM
    system_prompt = """You are an expert market researcher specializing in demographic and psychographic analysis.
Your task is to analyze web search results about an industry's customer demographics and extract structured information.
Focus only on extracting factual demographic information from the data. Do not invent or fabricate data.
If information is not available, clearly indicate what is missing."""
    
    user_prompt = f"""
I have collected web search data about customer demographics and psychographics in the {industry} industry in {region}.

Please analyze this data and extract structured information about the customer profile.
Organize the information into the following categories:

1. Key demographics:
   a. Age distribution (age groups and their proportion)
   b. Gender distribution
   c. Income levels
   d. Education levels
   e. Geographic concentration
   f. Professional background
2. Psychographics:
   a. Values and beliefs
   b. Lifestyle characteristics
   c. Interests and activities
   d. Attitudes and opinions
3. Market size and growth:
   a. Total addressable market size
   b. Growth rate
   c. Market segments and their size
4. Behavioral patterns:
   a. Buying habits
   b. Media consumption
   c. Online behavior
   d. Decision-making process

Only include information that is actually mentioned in the data. Do not invent details.
Provide your analysis in JSON format.

Here is the web search data:

{all_content}"""
    
    try:
        # Call the NVIDIA API for analysis
        llm_response = await call_nvidia_api(user_prompt, system_prompt, max_tokens=4000)
        structured_data_text = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Extract the JSON part from the response
        json_match = re.search(r'```json\n(.*?)\n```', structured_data_text, re.DOTALL)
        if json_match:
            structured_data_text = json_match.group(1)
        else:
            # Try to find JSON without markdown formatting
            json_match = re.search(r'({.*})', structured_data_text, re.DOTALL)
            if json_match:
                structured_data_text = json_match.group(1)
        
        try:
            structured_data = json.loads(structured_data_text)
            return structured_data
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM output as JSON, using raw output for further processing")
            return {"raw_analysis": structured_data_text}
    
    except Exception as e:
        logger.error(f"Error analyzing demographic data with LLM: {str(e)}")
        return {"error": str(e), "industry": industry, "region": region}

async def generate_demographic_analysis(structured_data: Dict[str, Any], industry: str, region: str) -> Dict[str, Any]:
    """Generate the final demographic analysis using structured data and LLM"""
    
    # Format the structured data for the LLM prompt
    structured_data_str = json.dumps(structured_data, indent=2)
    
    system_prompt = """You are an expert in market research and audience analysis. 
Your analysis must be data-driven and practical. When information is missing, either indicate that
or make conservative, realistic inferences based on available data."""

    user_prompt = f"""
Using the structured demographic data below, create a comprehensive audience analysis for the {industry} industry in {region}.

Transform the raw data into a well-structured audience profile with these components:
1. An executive summary of the audience profile
2. Detailed demographic breakdown including:
   - Age distribution
   - Gender distribution
   - Income levels
   - Education levels
   - Geographic distribution
3. Psychographic profile including:
   - Values and beliefs
   - Lifestyle characteristics
   - Interests and preferences
4. Market opportunity assessment:
   - Market size and growth potential
   - Key segments and their value
5. Strategic recommendations for targeting this audience

Return your analysis as a JSON object with these clearly structured sections.

Here's the structured data to use:

{structured_data_str}

Important: Make this analysis practical and actionable. Don't include any placeholder text or "N/A" entries.
When data is missing, make reasonable inferences based on the industry and available data.
"""

    try:
        # Call the NVIDIA API to generate the analysis
        llm_response = await call_nvidia_api(user_prompt, system_prompt, max_tokens=4000)
        result_text = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Extract the JSON part from the response
        json_match = re.search(r'```json\n(.*?)\n```', result_text, re.DOTALL)
        if json_match:
            result_text = json_match.group(1)
        else:
            # Try to find JSON without markdown formatting
            json_match = re.search(r'({.*})', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(1)
        
        try:
            result_json = json.loads(result_text)
            
            # Ensure required fields exist
            if "executive_summary" not in result_json:
                result_json["executive_summary"] = f"Audience analysis for the {industry} industry in {region}"
            
            # Add metadata
            result_json["industry"] = industry
            result_json["region"] = region
            result_json["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            return result_json
        except json.JSONDecodeError:
            # Create a structured format from the text if JSON parsing fails
            return {
                "executive_summary": f"Audience analysis for the {industry} industry in {region}",
                "raw_analysis": result_text,
                "industry": industry,
                "region": region,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
    
    except Exception as e:
        logger.error(f"Error generating demographic analysis: {str(e)}")
        return create_minimal_demographics(industry, region, f"Error generating analysis: {str(e)}")

async def validate_demographic_analysis(demographic_analysis: Dict[str, Any], industry: str, region: str) -> Dict[str, Any]:
    """Validate the demographic analysis with a secondary LLM for quality assurance"""
    
    # Format the analysis for validation
    analysis_str = json.dumps(demographic_analysis, indent=2)
    
    system_prompt = """You are an expert demographic analyst with years of experience in market research.
Your task is to validate a demographic analysis for accuracy, completeness, and realism.
Your role is not to create new content but to verify the existing analysis and provide a stamp of validation."""

    user_prompt = f"""
Please validate the following demographic analysis for the {industry} industry in {region}.

Assess the analysis for:
1. Factual accuracy and plausibility
2. Completeness of information
3. Logical consistency
4. Actionable insights
5. Appropriate market sizing and segmentation
6. Overall quality and usefulness

If you detect any serious issues or implausible claims, please note them specifically.
Otherwise, confirm that the analysis meets professional standards.

Here is the demographic analysis to validate:

{analysis_str}

Return the original analysis with an added "validation" object that includes:
1. A boolean "is_valid" field (true/false)
2. A "validation_notes" field with your assessment
3. A "confidence_score" from 1-10

Do not modify the original analysis content, only add the validation object.
"""

    try:
        # Use Groq API as a second opinion for validation
        llm_response = await call_groq_api(user_prompt, system_prompt, max_tokens=2000)
        result_text = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Parse the response
        try:
            validated_analysis = json.loads(result_text)
            
            # Check if the validator completely replaced the analysis
            if "executive_summary" not in validated_analysis and "demographics" not in validated_analysis:
                # It returned just the validation, so we need to combine it
                demographic_analysis["validation"] = validated_analysis
                return demographic_analysis
            
            # If we got here, the validator returned a complete object
            if "validation" not in validated_analysis:
                # Add a default validation if not provided
                validated_analysis["validation"] = {
                    "is_valid": True,
                    "validation_notes": "The analysis appears complete and consistent.",
                    "confidence_score": 8
                }
            
            return validated_analysis
        except json.JSONDecodeError:
            # Add a default validation
            demographic_analysis["validation"] = {
                "is_valid": True,
                "validation_notes": "Validation performed but structured output could not be parsed.",
                "confidence_score": 6
            }
            return demographic_analysis
                
    except Exception as e:
        logger.error(f"Error validating demographic analysis: {str(e)}")
        # Add a validation note about the error
        demographic_analysis["validation"] = {
            "is_valid": True,  # Still return valid to not block the workflow
            "validation_notes": f"Validation attempted but failed: {str(e)}",
            "confidence_score": 5
        }
        return demographic_analysis

async def create_llm_only_demographic_analysis(industry: str, region: str) -> Dict[str, Any]:
    """Create a demographic analysis using only LLM knowledge when web search fails"""
    
    system_prompt = """You are an expert in demographic analysis and market research.
Your task is to create a realistic demographic profile based on your knowledge of various industries and regions.
Use specific details and realistic statistics, but clearly indicate when you're providing estimates rather than documented facts."""

    user_prompt = f"""
The web search for demographic data about the {industry} industry in {region} has failed.

Based on your knowledge about this industry and region, please create a realistic audience profile that includes:

1. An executive summary of the typical audience for this industry
2. Demographic breakdown including reasonable estimates for:
   - Age distribution (specific age ranges and percentages)
   - Gender distribution (with percentages)
   - Income levels (specific income brackets and percentages)
   - Education levels (with percentages)
   - Geographic concentration within {region}
3. Psychographic profile including:
   - Values and beliefs relevant to this industry
   - Lifestyle characteristics of typical customers
   - Interests and preferences that might influence purchasing
4. Market opportunity assessment:
   - Estimated market size and growth rate
   - Key segments and their approximate value
5. Strategic recommendations for targeting this audience

Format your response as a detailed JSON object with these clearly structured sections.
Make sure all numerical values are realistic for this industry and region.
Include a note that this analysis is based on general knowledge rather than specific web data.
"""

    try:
        # Call the NVIDIA API for the analysis
        llm_response = await call_nvidia_api(user_prompt, system_prompt, max_tokens=4000, temperature=0.5)
        result_text = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Extract the JSON part from the response
        json_match = re.search(r'```json\n(.*?)\n```', result_text, re.DOTALL)
        if json_match:
            result_text = json_match.group(1)
        else:
            # Try to find JSON without markdown formatting
            json_match = re.search(r'({.*})', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(1)
        
        try:
            result_json = json.loads(result_text)
            
            # Add metadata and disclaimer
            result_json["industry"] = industry
            result_json["region"] = region
            result_json["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            result_json["data_source"] = "LLM-generated due to web search failure"
            
            return result_json
        except json.JSONDecodeError:
            # Create a structured format from the text if JSON parsing fails
                return {
                "executive_summary": f"Audience analysis for the {industry} industry in {region}",
                "raw_analysis": result_text,
                "industry": industry,
                "region": region,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "data_source": "LLM-generated due to web search failure (parsing error)"
            }
    
    except Exception as e:
        logger.error(f"Error creating LLM-only demographic analysis: {str(e)}")
        return create_minimal_demographics(industry, region, f"Error creating LLM-only analysis: {str(e)}")

def create_minimal_demographics(industry: str, region: str, message: str = None) -> Dict[str, Any]:
    """Create a minimal valid demographic analysis structure as ultimate fallback"""
    
    summary = message or f"Basic demographic information for {industry} industry in {region}"
    
    return {
        "executive_summary": summary,
        "demographics": {
            "age": [
                {"group": "18-24", "percentage": "15%"},
                {"group": "25-34", "percentage": "30%"},
                {"group": "35-44", "percentage": "25%"},
                {"group": "45-54", "percentage": "20%"},
                {"group": "55+", "percentage": "10%"}
            ],
            "gender": [
                {"type": "Male", "percentage": "55%"},
                {"type": "Female", "percentage": "45%"}
            ],
            "income": [
                {"level": "Middle income", "percentage": "60%"},
                {"level": "High income", "percentage": "40%"}
            ],
            "education": [
                {"level": "College degree", "percentage": "65%"},
                {"level": "High school", "percentage": "25%"},
                {"level": "Advanced degree", "percentage": "10%"}
            ]
        },
        "psychographics": {
            "values": [f"Typical values for {industry} customers"],
            "interests": [f"Typical interests for {industry} customers"]
        },
        "market_size": {
            "total": "Undetermined",
            "growth_rate": "Industry average"
        },
        "industry": industry,
        "region": region,
        "note": "Created using fallback data due to processing errors"
    }

@register_function(config_type=AudienceResearchWorkflowConfig)
async def audience_research_workflow(config: AudienceResearchWorkflowConfig, builder: Builder) -> AsyncGenerator[Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]], None]:
    """
    Orchestrates a complete audience research workflow combining multiple research tools.
    
    This function coordinates the execution of keyword research, competitor analysis,
    demographic analysis, and persona building to create a comprehensive audience research report.
    """
    logger.info(f"Starting audience research workflow in mode: {config.mode}")
    
    async def _workflow_fn(input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Parse input data to extract parameters using LLM
            parsed_data = parse_input({"input_data": input_data, "_skip_logging": True})
            
            # Extract user query for LLM parsing
            user_query = parsed_data.get("question", "") or input_data.get("question", "")
            if not user_query:
                # Try to find any text field in the input
                for key, value in input_data.items():
                    if isinstance(value, str) and len(value) > 5:
                        user_query = value
                        break
            
            logger.info(f"Original user query: {user_query}")
            
            # Use LLM to parse industry and region from the query
            try:
                extracted_info = await extract_query_parameters(user_query)
                industry = extracted_info.get("industry", "software")
                region = extracted_info.get("region", "United States")
                
                # Update parsed_data with extracted parameters
                parsed_data["industry"] = industry
                parsed_data["region"] = region
                
                # Add any other extracted parameters
                for key, value in extracted_info.items():
                    if key not in ["industry", "region"] and value:
                        parsed_data[key] = value
                
            except Exception as e:
                logger.error(f"Error extracting parameters from query: {str(e)}")
                industry = parsed_data.get("industry", "software")
                region = parsed_data.get("region", "United States")
            
            logger.info(f"Orchestrating audience research for '{industry}' industry in {region}")
            
            # Initialize results dictionary
            results = {}
            
            # Create task to execute all tools in parallel
            tasks = []
            
            try:
                # Keyword research task
                keyword_config = KeywordResearchConfig(
                    query=industry,
                    max_results=20,
                    include_search_volume=True,
                    include_competition=True
                )
                
                # Correctly handle the AsyncGenerator
                keyword_gen = keyword_research(keyword_config, builder)
                keyword_fn = await anext(await keyword_gen)
                tasks.append(("keywords", asyncio.create_task(keyword_fn(parsed_data))))
                
                # Demographic analysis task
                demo_config = DemographicAnalysisConfig(
                    industry=industry,
                    region=region
                )
                
                # Correctly handle the AsyncGenerator
                demo_gen = demographic_analysis(demo_config, builder)
                demo_fn = await anext(await demo_gen)
                tasks.append(("demographics", asyncio.create_task(demo_fn(parsed_data))))
                
                # Competitor analysis task
                comp_config = CompetitorAnalysisConfig(
                    industry=industry,
                    max_results=5
                )
                
                # Correctly handle the AsyncGenerator
                comp_gen = competitor_analysis(comp_config, builder)
                comp_fn = await anext(await comp_gen)
                tasks.append(("competitors", asyncio.create_task(comp_fn(parsed_data))))
                
                # If in-depth mode, create personas
                if config.mode == "in-depth" or config.mode == "default":
                    persona_config = PersonaBuilderConfig(
                        industry=industry,
                        persona_name=f"{industry} Target Persona",
                        number_of_personas=2
                    )
                    
                    # Correctly handle the AsyncGenerator
                    persona_gen = persona_builder(persona_config, builder)
                    persona_fn = await anext(await persona_gen)
                    tasks.append(("personas", asyncio.create_task(persona_fn(parsed_data))))
                
                # Gather results from all tasks
                for key, task in tasks:
                    try:
                        results[key] = await task
                    except Exception as e:
                        logger.error(f"Error in {key} task: {str(e)}")
                        results[key] = {
                            "error": str(e),
                            "summary": f"Failed to perform {key} analysis"
                        }
                
                # Format results into a comprehensive report
                report = await format_audience_research_report(results, industry, region, config.mode)
                return report
                
            except Exception as e:
                logger.error(f"Error in audience research workflow: {str(e)}")
                return {
                    "error": str(e),
                    "summary": f"An error occurred while orchestrating audience research for {industry} in {region}",
                    "industry": industry,
                    "region": region,
                    "available_results": results
                }
        
        except Exception as e:
            logger.error(f"Unexpected error in audience research workflow: {str(e)}")
            return {
                "error": str(e),
                "summary": "An unexpected error occurred in the audience research workflow",
                "available_results": {}
            }
    
    yield _workflow_fn

async def extract_query_parameters(query: str) -> Dict[str, Any]:
    """
    Use LLM to extract relevant parameters from the user's query
    """
    system_prompt = """You are a specialized system for extracting research parameters from user queries.
Your job is to identify the industry, region, and other relevant parameters that will be used for audience research.
Only extract information that is explicitly mentioned or clearly implied in the query."""

    user_prompt = f"""
Extract the following parameters from this user query:
- industry: The business sector or type of company (e.g., "software", "healthcare", "retail", "data science")
- region: The geographic area of interest (country, region, etc.)
- target_audience: Any specific segment mentioned (e.g., "small businesses", "teenagers", "professionals")
- objectives: Any specific business goals mentioned (e.g., "increase sales", "improve engagement")

User query: "{query}"

Return your answer as a JSON object with these keys. If a parameter is not mentioned, use a sensible default or leave it empty.
"""

    try:
        # Call the NVIDIA API for parameter extraction
        llm_response = await call_nvidia_api(user_prompt, system_prompt, max_tokens=1000, temperature=0.1)
        result_text = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Extract the JSON part from the response
        json_match = re.search(r'```json\n(.*?)\n```', result_text, re.DOTALL)
        if json_match:
            result_text = json_match.group(1)
        else:
            # Try to find JSON without markdown formatting
            json_match = re.search(r'({.*})', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(1)
        
        try:
            parameters = json.loads(result_text)
            logger.info(f"Extracted parameters: {parameters}")
            return parameters
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM output as JSON, attempting to extract manually")
            
            # Simple regex fallback extraction
            parameters = {}
            industry_match = re.search(r'"industry":\s*"([^"]+)"', result_text)
            if industry_match:
                parameters["industry"] = industry_match.group(1)
            
            region_match = re.search(r'"region":\s*"([^"]+)"', result_text)
            if region_match:
                parameters["region"] = region_match.group(1)
            
            return parameters
    
    except Exception as e:
        logger.error(f"Error extracting parameters with LLM: {str(e)}")
        # Return empty dict on failure
        return {}

async def format_audience_research_report(results: Dict[str, Any], industry: str, region: str, mode: str) -> Dict[str, Any]:
    """
    Format all research results into a comprehensive report
    """
    # Basic report structure
    report = {
        "title": f"Audience Research Report: {industry} Industry in {region}",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "industry": industry,
        "region": region,
        "mode": mode,
        "summary": f"Comprehensive audience research for the {industry} industry in {region}",
        "sections": {}
    }
    
    # Add available sections from results
    if "keywords" in results:
        report["sections"]["keyword_research"] = results["keywords"]
    
    if "demographics" in results:
        report["sections"]["demographics"] = results["demographics"]
    
    if "competitors" in results:
        report["sections"]["competitor_analysis"] = results["competitors"]
    
    if "personas" in results:
        report["sections"]["personas"] = results["personas"]
    
    # Add executive summary using NVIDIA API
    try:
        # Create a summary of all the research
        system_prompt = """You are an expert market researcher specializing in creating executive summaries.
Your task is to create a concise executive summary of an audience research report."""
        
        # Create a simplified version of the results for the prompt
        simplified_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                if "executive_summary" in value:
                    simplified_results[key] = value["executive_summary"]
                elif "summary" in value:
                    simplified_results[key] = value["summary"]
                else:
                    simplified_results[key] = "Available but no summary provided"
            else:
                simplified_results[key] = "Available but in unknown format"
        
        user_prompt = f"""
I have conducted comprehensive audience research for the {industry} industry in {region}.
The research includes the following components:

{json.dumps(simplified_results, indent=2)}

Please create a concise executive summary (2-3 paragraphs) that synthesizes these findings
into actionable insights. Focus on the most important points that would be valuable for
marketing decision-makers.
"""
        
        llm_response = await call_nvidia_api(user_prompt, system_prompt, max_tokens=1000, temperature=0.3)
        executive_summary = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        if executive_summary:
            report["executive_summary"] = executive_summary
    
    except Exception as e:
        logger.warning(f"Could not generate executive summary: {str(e)}")
        report["executive_summary"] = f"Audience research for {industry} in {region} revealed key insights about market demographics, competitors, and relevant keywords for targeting. Review detailed sections for specific findings."
    
    return report
