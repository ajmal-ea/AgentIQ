from typing import List, Dict, Any, Optional
from aiq.data_models.function import FunctionBaseConfig
from aiq.cli.register_workflow import register_function
import json
import logging
from logging import getLogger
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _parse_input(input_data):
    """
    Parse the input data to handle different input formats.
    """
    message = ""
    
    # Handle both direct string input and dictionary with input_message
    if isinstance(input_data, str):
        message = input_data
    elif isinstance(input_data, dict):
        if "input_message" in input_data:
            message = input_data["input_message"]
        elif "question" in input_data:
            message = input_data["question"]
        else:
            # Try to use the entire input as a string
            message = str(input_data)
    else:
        message = str(input_data)
    
    logger.info(f"Processed input message: {message}")
    return message

# Define configuration schema for the keyword research tool
class KeywordResearchConfig(FunctionBaseConfig, name="keyword_research"):
    """Tool for performing keyword research"""
    query: str
    max_results: int = 20
    include_search_volume: bool = False
    include_competition: bool = False

@register_function(config_type=KeywordResearchConfig)
async def keyword_research(config: KeywordResearchConfig, builder) -> Dict[str, Any]:
    """
    Performs keyword research for the given query.
    In production, this would call an external API like Google Keyword Planner, SEMrush, or Ahrefs.
    """
    logger.info(f"Performing keyword research for query: {config.query}")
    
    # Placeholder: In a real implementation, this would make an API call
    # For now, we'll simulate some keywords related to the query
    query = config.query.lower()
    
    keywords = []
    # Generate some relevant keywords based on the query
    if "software" in query:
        base_keywords = ["software solutions", "enterprise software", "custom software", "saas", "cloud computing"]
    elif "marketing" in query:
        base_keywords = ["digital marketing", "content marketing", "social media marketing", "email campaigns", "seo"]
    elif "finance" in query:
        base_keywords = ["financial services", "investment", "banking", "wealth management", "retirement planning"]
    else:
        # Generic keywords for any other industry
        base_keywords = ["services", "solutions", "consulting", "experts", "professional"]
    
    # Generate variations
    keywords = []
    for base in base_keywords:
        keywords.append(f"{query} {base}")
        keywords.append(f"{base} for {query}")
        keywords.append(f"best {base} for {query}")
        keywords.append(f"affordable {base} {query}")
    
    # Include some long-tail keywords
    long_tail = [
        f"how to find {query} {base_keywords[0]}",
        f"top {query} {base_keywords[1]} providers",
        f"{query} {base_keywords[2]} near me",
        f"compare {query} {base_keywords[3]} options"
    ]
    keywords.extend(long_tail)
    
    # Add search volume and competition data if requested
    keyword_data = []
    for i, kw in enumerate(keywords[:config.max_results]):
        keyword_info = {"keyword": kw}
        
        if config.include_search_volume:
            # In production, this would be real data from an API
            keyword_info["search_volume"] = 100 + (i * 50)  # Simulated volume
            
        if config.include_competition:
            # In production, this would be real data from an API
            keyword_info["competition"] = round(0.1 + (i * 0.05), 2)  # Simulated competition score (0-1)
            
        keyword_data.append(keyword_info)
    
    logger.info(f"Keyword research complete. Found {len(keyword_data)} keywords")
    return {"keywords": keyword_data[:config.max_results]}


# Define configuration schema for the competitor analysis tool
class CompetitorAnalysisConfig(FunctionBaseConfig, name="competitor_analysis"):
    """Tool for analyzing competitors"""
    industry: str
    competitor: Optional[str] = None
    max_results: int = 5
    include_strategies: bool = True
    include_target_audience: bool = True


@register_function(config_type=CompetitorAnalysisConfig)
async def competitor_analysis(config: CompetitorAnalysisConfig, builder) -> Dict[str, Any]:
    """
    Analyzes competitors in the specified industry.
    In production, this would call external APIs or web scraping services.
    """
    logger.info(f"Performing competitor analysis for industry: {config.industry}, competitor: {config.competitor}")
    
    # Placeholder: In a real implementation, this would make API calls or use web scraping
    industry = config.industry.lower()
    
    # Simulated competitor data based on industry
    competitors_by_industry = {
        "software": [
            {"name": "Microsoft", "market_share": 0.25, "target_audience": "Enterprise businesses", "strategy": "Product ecosystem and integration"},
            {"name": "Oracle", "market_share": 0.18, "target_audience": "Large enterprises", "strategy": "Comprehensive database solutions"},
            {"name": "Salesforce", "market_share": 0.15, "target_audience": "Sales teams across industries", "strategy": "Cloud-based CRM focus"},
            {"name": "SAP", "market_share": 0.12, "target_audience": "Global enterprises", "strategy": "Integrated enterprise solutions"},
            {"name": "Adobe", "market_share": 0.10, "target_audience": "Creative professionals", "strategy": "Creative cloud subscription model"}
        ],
        "marketing": [
            {"name": "HubSpot", "market_share": 0.22, "target_audience": "SMBs", "strategy": "Inbound marketing focus"},
            {"name": "Mailchimp", "market_share": 0.18, "target_audience": "Small businesses", "strategy": "Email marketing simplicity"},
            {"name": "SEMrush", "market_share": 0.15, "target_audience": "Digital marketers", "strategy": "Comprehensive SEO tools"},
            {"name": "Moz", "market_share": 0.12, "target_audience": "SEO professionals", "strategy": "SEO education and tools"},
            {"name": "Constant Contact", "market_share": 0.10, "target_audience": "Local businesses", "strategy": "User-friendly email marketing"}
        ],
        "finance": [
            {"name": "JPMorgan Chase", "market_share": 0.25, "target_audience": "Diverse clientele", "strategy": "Full-service banking"},
            {"name": "Bank of America", "market_share": 0.20, "target_audience": "Mass market", "strategy": "Digital banking convenience"},
            {"name": "Wells Fargo", "market_share": 0.15, "target_audience": "Middle-class families", "strategy": "Community banking focus"},
            {"name": "Citibank", "market_share": 0.12, "target_audience": "Urban professionals", "strategy": "Global banking network"},
            {"name": "Capital One", "market_share": 0.10, "target_audience": "Credit card users", "strategy": "Rewards programs"}
        ]
    }
    
    # Default to a generic industry if the specified one isn't in our data
    selected_industry = industry if industry in competitors_by_industry else "software"
    competitors = competitors_by_industry.get(selected_industry, competitors_by_industry["software"])
    
    # Filter for a specific competitor if provided
    if config.competitor:
        competitors = [comp for comp in competitors if config.competitor.lower() in comp["name"].lower()]
    
    # Limit results
    competitors = competitors[:config.max_results]
    
    # Remove fields if not requested
    if not config.include_strategies:
        for comp in competitors:
            comp.pop("strategy", None)
    
    if not config.include_target_audience:
        for comp in competitors:
            comp.pop("target_audience", None)
    
    # Create a competitor report summary
    summary = f"Analysis of {len(competitors)} competitors in the {industry} industry."
    
    logger.info(f"Competitor analysis complete. Found {len(competitors)} competitors")
    
    # Modified return structure to match what's expected in the workflow
    return {
        "competitors": competitors
    }


# Define configuration schema for the demographic analysis tool
class DemographicAnalysisConfig(FunctionBaseConfig, name="demographic_analysis"):
    """Tool for analyzing audience demographics and psychographics"""
    industry: str
    region: Optional[str] = "United States"
    include_psychographics: bool = True


@register_function(config_type=DemographicAnalysisConfig)
async def demographic_analysis(config: DemographicAnalysisConfig, builder) -> Dict[str, Any]:
    """
    Analyzes demographic and psychographic data for the specified industry and region.
    In production, this would call external APIs like Google Analytics or market research data.
    """
    logger.info(f"Performing demographic analysis for industry: {config.industry}, region: {config.region}")
    
    # Placeholder: In a real implementation, this would make API calls to market research sources
    industry = config.industry.lower()
    region = config.region
    
    # Simulated demographic data based on industry
    demographics_by_industry = {
        "software": {
            "age_groups": [
                {"range": "18-24", "percentage": 0.15},
                {"range": "25-34", "percentage": 0.35},
                {"range": "35-44", "percentage": 0.30},
                {"range": "45-54", "percentage": 0.15},
                {"range": "55+", "percentage": 0.05}
            ],
            "gender": [
                {"group": "Male", "percentage": 0.65},
                {"group": "Female", "percentage": 0.35}
            ],
            "income_levels": [
                {"level": "Under $50k", "percentage": 0.10},
                {"level": "$50k-$100k", "percentage": 0.35},
                {"level": "$100k-$150k", "percentage": 0.30},
                {"level": "Over $150k", "percentage": 0.25}
            ],
            "education": [
                {"level": "High School", "percentage": 0.10},
                {"level": "Bachelor's", "percentage": 0.50},
                {"level": "Master's or higher", "percentage": 0.40}
            ],
            "psychographics": {
                "interests": ["Technology", "Innovation", "Efficiency", "Problem-solving"],
                "values": ["Productivity", "Reliability", "Security", "Innovation"],
                "pain_points": ["Complex systems", "Data security", "Integration issues", "High costs"]
            }
        },
        "marketing": {
            "age_groups": [
                {"range": "18-24", "percentage": 0.20},
                {"range": "25-34", "percentage": 0.40},
                {"range": "35-44", "percentage": 0.25},
                {"range": "45-54", "percentage": 0.10},
                {"range": "55+", "percentage": 0.05}
            ],
            "gender": [
                {"group": "Male", "percentage": 0.45},
                {"group": "Female", "percentage": 0.55}
            ],
            "income_levels": [
                {"level": "Under $50k", "percentage": 0.20},
                {"level": "$50k-$100k", "percentage": 0.40},
                {"level": "$100k-$150k", "percentage": 0.25},
                {"level": "Over $150k", "percentage": 0.15}
            ],
            "education": [
                {"level": "High School", "percentage": 0.15},
                {"level": "Bachelor's", "percentage": 0.60},
                {"level": "Master's or higher", "percentage": 0.25}
            ],
            "psychographics": {
                "interests": ["Social media", "Content creation", "Analytics", "Brand strategy"],
                "values": ["Creativity", "Communication", "Engagement", "Results"],
                "pain_points": ["ROI measurement", "Content saturation", "Algorithm changes", "Budget constraints"]
            }
        },
        "finance": {
            "age_groups": [
                {"range": "18-24", "percentage": 0.05},
                {"range": "25-34", "percentage": 0.15},
                {"range": "35-44", "percentage": 0.25},
                {"range": "45-54", "percentage": 0.30},
                {"range": "55+", "percentage": 0.25}
            ],
            "gender": [
                {"group": "Male", "percentage": 0.55},
                {"group": "Female", "percentage": 0.45}
            ],
            "income_levels": [
                {"level": "Under $50k", "percentage": 0.15},
                {"level": "$50k-$100k", "percentage": 0.30},
                {"level": "$100k-$150k", "percentage": 0.25},
                {"level": "Over $150k", "percentage": 0.30}
            ],
            "education": [
                {"level": "High School", "percentage": 0.10},
                {"level": "Bachelor's", "percentage": 0.45},
                {"level": "Master's or higher", "percentage": 0.45}
            ],
            "psychographics": {
                "interests": ["Investing", "Wealth management", "Financial planning", "Economics"],
                "values": ["Security", "Stability", "Growth", "Trust"],
                "pain_points": ["Market volatility", "Financial complexity", "Retirement planning", "Debt management"]
            }
        }
    }
    
    # Default to a generic industry if the specified one isn't in our data
    selected_industry = industry if industry in demographics_by_industry else "marketing"
    demographic_data = demographics_by_industry.get(selected_industry, demographics_by_industry["marketing"])
    
    # Create a copy to avoid modifying the original
    result = json.loads(json.dumps(demographic_data))
    
    # Remove psychographics if not requested
    if not config.include_psychographics:
        result.pop("psychographics", None)
    
    # Add regional context if specified
    regional_context = f"Data is specific to {region}. Regional variations may exist."
    
    logger.info(f"Demographic analysis complete for {industry} in {region}")
    
    # Modified return structure to match what's expected in the workflow
    return {
        "demographics": result
    }


# Define configuration schema for the persona builder tool
class PersonaBuilderConfig(FunctionBaseConfig, name="persona_builder"):
    """Tool for building audience personas based on research data"""
    industry: str
    persona_name: str = ""
    number_of_personas: int = 1
    include_markdown_format: bool = True


@register_function(config_type=PersonaBuilderConfig)
async def persona_builder(config: PersonaBuilderConfig, builder) -> Dict[str, Any]:
    """
    Builds detailed audience personas based on the outputs of the other research tools.
    This tool collates the data and formats it into actionable audience personas.
    """
    logger.info(f"Building audience personas for industry: {config.industry}")
    
    # Get outputs from previous tools via builder context
    # In a real implementation, we would handle missing data more gracefully
    try:
        keywords_output = builder.get_previous_output("keyword_research")
        keywords = keywords_output.get("keywords", [])
    except Exception as e:
        logger.warning(f"Could not get keyword research data: {e}")
        keywords = []
    
    try:
        competitor_output = builder.get_previous_output("competitor_analysis")
        competitor_report = competitor_output.get("competitor_report", {})
        competitors = competitor_report.get("competitors", [])
    except Exception as e:
        logger.warning(f"Could not get competitor analysis data: {e}")
        competitors = []
    
    try:
        demographic_output = builder.get_previous_output("demographic_analysis")
        demographic_insights = demographic_output.get("demographic_insights", {})
        demographic_data = demographic_insights.get("data", {})
    except Exception as e:
        logger.warning(f"Could not get demographic analysis data: {e}")
        demographic_data = {}
    
    # Generate personas based on the collected data
    personas = []
    
    # Use demographic data to create realistic personas
    age_groups = demographic_data.get("age_groups", [])
    gender_data = demographic_data.get("gender", [])
    income_levels = demographic_data.get("income_levels", [])
    education_data = demographic_data.get("education", [])
    psychographics = demographic_data.get("psychographics", {})
    
    # Default persona name if not provided
    persona_name_prefix = config.persona_name if config.persona_name else "Persona"
    
    for i in range(config.number_of_personas):
        # Select demographic characteristics based on the highest percentages
        age_group = max(age_groups, key=lambda x: x["percentage"])["range"] if age_groups else "25-34"
        gender = max(gender_data, key=lambda x: x["percentage"])["group"] if gender_data else "Not specified"
        income = max(income_levels, key=lambda x: x["percentage"])["level"] if income_levels else "$50k-$100k"
        education = max(education_data, key=lambda x: x["percentage"])["level"] if education_data else "Bachelor's"
        
        # Create a persona with a variation if creating multiple
        if config.number_of_personas > 1:
            # Simple variation for multiple personas
            age_groups_copy = age_groups.copy() if age_groups else []
            if age_groups_copy and i > 0 and len(age_groups_copy) > i:
                # Sort by percentage descending and take i-th item
                sorted_ages = sorted(age_groups_copy, key=lambda x: x["percentage"], reverse=True)
                age_group = sorted_ages[min(i, len(sorted_ages)-1)]["range"]
        
        # Get interests and pain points from psychographics
        interests = psychographics.get("interests", []) if psychographics else []
        values = psychographics.get("values", []) if psychographics else []
        pain_points = psychographics.get("pain_points", []) if psychographics else []
        
        # Select relevant keywords (simplified approach)
        relevant_keywords = [kw["keyword"] for kw in keywords[:5]] if keywords else []
        
        # Extract competitor strategies
        competitor_strategies = [comp.get("strategy", "") for comp in competitors if "strategy" in comp]
        competitor_audiences = [comp.get("target_audience", "") for comp in competitors if "target_audience" in comp]
        
        # Create persona object
        persona_name = f"{persona_name_prefix} {i+1}" if config.number_of_personas > 1 else persona_name_prefix
        persona = {
            "name": persona_name,
            "demographics": {
                "age_range": age_group,
                "gender": gender,
                "income_level": income,
                "education_level": education
            },
            "psychographics": {
                "interests": interests,
                "values": values,
                "pain_points": pain_points
            },
            "online_behavior": {
                "searched_keywords": relevant_keywords,
                "competitor_engagement": competitor_audiences
            },
            "marketing_approach": {
                "unique_value_proposition": f"Based on {config.industry} industry needs and identified pain points",
                "messaging_strategy": "Derived from competitor strategies and psychographic insights",
                "competitor_strategies": competitor_strategies,
                "recommended_keywords": relevant_keywords
            }
        }
        
        personas.append(persona)
    
    # Convert to markdown if requested
    if config.include_markdown_format:
        markdown_personas = []
        for persona in personas:
            markdown = f"# {persona['name']}\n\n"
            
            markdown += "## Demographics\n"
            markdown += f"- **Age Range:** {persona['demographics']['age_range']}\n"
            markdown += f"- **Gender:** {persona['demographics']['gender']}\n"
            markdown += f"- **Income Level:** {persona['demographics']['income_level']}\n"
            markdown += f"- **Education Level:** {persona['demographics']['education_level']}\n\n"
            
            markdown += "## Psychographics\n"
            markdown += "### Interests\n"
            for interest in persona['psychographics']['interests']:
                markdown += f"- {interest}\n"
            
            markdown += "\n### Values\n"
            for value in persona['psychographics']['values']:
                markdown += f"- {value}\n"
            
            markdown += "\n### Pain Points\n"
            for pain in persona['psychographics']['pain_points']:
                markdown += f"- {pain}\n"
            
            markdown += "\n## Online Behavior\n"
            markdown += "### Searched Keywords\n"
            for kw in persona['online_behavior']['searched_keywords']:
                markdown += f"- {kw}\n"
            
            markdown += "\n### Competitor Engagement\n"
            for audience in persona['online_behavior']['competitor_engagement']:
                markdown += f"- {audience}\n"
            
            markdown += "\n## Marketing Approach\n"
            markdown += f"- **Unique Value Proposition:** {persona['marketing_approach']['unique_value_proposition']}\n"
            markdown += f"- **Messaging Strategy:** {persona['marketing_approach']['messaging_strategy']}\n"
            
            markdown += "\n### Competitor Strategies\n"
            for strategy in persona['marketing_approach']['competitor_strategies']:
                markdown += f"- {strategy}\n"
            
            markdown += "\n### Recommended Keywords\n"
            for kw in persona['marketing_approach']['recommended_keywords']:
                markdown += f"- {kw}\n"
            
            markdown_personas.append(markdown)
        
        # Include both JSON and markdown in the result
        result = {
            "personas": personas,
            "personas_markdown": markdown_personas
        }
    else:
        result = {
            "personas": personas
        }
    
    logger.info(f"Created {len(personas)} audience personas for {config.industry}")
    return result 