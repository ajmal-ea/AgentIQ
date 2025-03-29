from typing import List, Dict, Any, Optional
from agentiq.tool import register_function, FunctionBaseConfig
import json
import logging
from pydantic import Field, validator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define configuration schema for the keyword research tool
class KeywordResearchConfig(FunctionBaseConfig, name="keyword_research"):
    """Tool for performing keyword research"""
    query: str = Field(..., description="The search query to perform keyword research on")
    max_results: int = Field(20, description="Maximum number of keywords to return", ge=1, le=100)
    include_search_volume: bool = Field(False, description="Whether to include search volume data")
    include_competition: bool = Field(False, description="Whether to include competition data")

    @validator('query')
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v

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
    elif "healthcare" in query:
        base_keywords = ["healthcare IT", "medical software", "electronic health records", "telemedicine", "patient management"]
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

# Define configuration schema for the audience research tool
class AudienceResearchConfig(FunctionBaseConfig, name="audience_research"):
    """Tool for coordinating comprehensive audience research"""
    input_data: Optional[Dict[str, Any]] = Field(None, description="Input data containing industry, company_type, target_market")
    input_message: str = Field("audience research for software industry", description="The research query or instructions")

    class Config:
        # Allow extra fields to be flexible with input
        extra = "allow"

@register_function(config_type=AudienceResearchConfig)
async def audience_research(config: AudienceResearchConfig, builder) -> Dict[str, Any]:
    """
    Coordinates audience research by orchestrating multiple research tools.
    Acts as a wrapper for the individual research tools to provide a comprehensive result.
    """
    logger.info(f"Processing input: {config.input_data}")
    
    input_data = config.input_data or {}
    input_message = config.input_message
    
    # Set default values
    industry = input_data.get("industry", "software")
    query = input_data.get("query", industry)
    competitor = input_data.get("competitor")
    region = input_data.get("region", "United States")
    
    # Special handling for healthcare industry
    if "healthcare" in industry.lower() or (input_data.get("target_market", "").lower() and "healthcare" in input_data.get("target_market", "").lower()):
        logger.info(f"Healthcare industry detected, applying specialized research parameters")
        if not query or query == industry:
            query = "healthcare software"
    
    # Parse from input_data if available
    if "company_type" in input_data:
        # Adjust industry/query based on company_type if provided
        company_type = input_data.get("company_type", "")
        if company_type and company_type.lower() != industry.lower():
            query = f"{company_type} for {industry}"
    
    if "target_market" in input_data:
        target_market = input_data.get("target_market", "")
        if target_market:
            query = f"{query} for {target_market}"
    
    logger.info(f"Parsed input data: {{'industry': '{industry}', 'query': '{query}', 'competitor': {competitor}, 'region': '{region}'}}")
    
    try:
        # Step 1: Perform keyword research
        keyword_config = KeywordResearchConfig(
            query=query,
            max_results=15,
            include_search_volume=True,
            include_competition=True
        )
        keyword_results = await keyword_research(keyword_config, builder)
        
        # Step 2: Perform competitor analysis
        competitor_config = CompetitorAnalysisConfig(
            industry=industry,
            competitor=competitor,
            max_results=3,
            include_strategies=True,
            include_target_audience=True
        )
        competitor_results = await competitor_analysis(competitor_config, builder)
        
        # Step 3: Perform demographic analysis
        demographic_config = DemographicAnalysisConfig(
            industry=industry,
            region=region,
            include_psychographics=True
        )
        demographic_results = await demographic_analysis(demographic_config, builder)
        
        # Step 4: Build personas
        persona_config = PersonaBuilderConfig(
            industry=industry,
            persona_name=f"{industry.title()} Audience",
            number_of_personas=2,
            include_demographics=True,
            include_psychographics=True,
            include_online_behavior=True
        )
        persona_results = await persona_builder(persona_config, builder)
        
        # Ensure we have some minimal results even if individual tools fail
        if not keyword_results.get("keywords"):
            keyword_results["keywords"] = [{"keyword": query, "search_volume": "medium", "competition": "medium"}]
        
        if not competitor_results.get("competitor_report", {}).get("competitors"):
            competitor_results["competitor_report"] = {
                "summary": f"Analysis of the {industry} industry.",
                "competitors": [{"name": "Generic Competitor", "market_share": 0.25, "target_audience": "Various businesses", "strategy": "Multiple approaches"}]
            }
        
        # Combine all results
        combined_results = {
            "keywords": keyword_results.get("keywords", []),
            "competitor_report": competitor_results.get("competitor_report", {}),
            "demographic_insights": demographic_results.get("demographic_insights", {}),
            "personas": persona_results.get("personas", []),
            "personas_markdown": persona_results.get("personas_markdown", []),
            "summary": f"Completed audience research for {industry} industry focused on {query}."
        }
        
        # Validate that we're not returning empty results
        if not combined_results["personas_markdown"]:
            # Generate at least one basic persona if the tool didn't return any
            combined_results["personas_markdown"] = [
                f"# {industry.title()} Target Persona\n\n" +
                f"## Demographics\n- Industry: {industry}\n- Region: {region}\n\n" +
                f"## Key Interests\n- {query}\n\n" +
                f"## Challenges\n- Finding reliable solutions\n\n" +
                f"## Goals\n- Implementing effective solutions"
            ]
        
        logger.info("Audience research completed successfully")
        return combined_results
        
    except Exception as e:
        logger.error(f"Error during audience research: {str(e)}")
        # Return a minimal valid response even when errors occur
        return {
            "error": str(e),
            "summary": f"Partial results for audience research on {industry}: {str(e)}",
            "keywords": [{"keyword": query, "search_volume": "unknown", "competition": "unknown"}],
            "competitor_report": {"summary": "Error during analysis", "competitors": []},
            "demographic_insights": {"summary": "Error during analysis"},
            "personas": [{"name": f"{industry} Audience", "description": "Error occurred during persona generation"}],
            "personas_markdown": [f"# {industry.title()} Target Persona\n\nError occurred during research: {str(e)}"]
        }

# Define configuration schema for the competitor analysis tool
class CompetitorAnalysisConfig(FunctionBaseConfig, name="competitor_analysis"):
    """Tool for analyzing competitors"""
    industry: str = Field(..., description="The industry to analyze competitors for")
    competitor: Optional[str] = Field(None, description="Specific competitor to analyze")
    max_results: int = Field(5, description="Maximum number of competitors to analyze", ge=1, le=10)
    include_strategies: bool = Field(True, description="Whether to include competitor strategies")
    include_target_audience: bool = Field(True, description="Whether to include competitor target audience")


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
    return {
        "competitor_report": {
            "summary": summary,
            "competitors": competitors
        }
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
    return {
        "demographic_insights": {
            "industry": industry,
            "region": region,
            "regional_context": regional_context,
            "data": result
        }
    }


# Define configuration schema for the persona builder tool
class PersonaBuilderConfig(FunctionBaseConfig, name="persona_builder"):
    """Tool for building audience personas"""
    industry: str = Field(..., description="The industry to create personas for")
    persona_name: str = Field("Target Persona", description="Name for the primary persona")
    number_of_personas: int = Field(1, description="Number of personas to generate", ge=1, le=5)
    include_demographics: bool = Field(True, description="Whether to include demographic information")
    include_psychographics: bool = Field(True, description="Whether to include psychographic information")
    include_online_behavior: bool = Field(True, description="Whether to include online behavior information")
    
    @validator('industry')
    def industry_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Industry cannot be empty")
        return v

@register_function(config_type=PersonaBuilderConfig)
async def persona_builder(config: PersonaBuilderConfig, builder) -> Dict[str, Any]:
    """
    Creates detailed audience personas based on industry data.
    This function integrates results from other tools to build comprehensive personas.
    In production, this would use real data from previous tool calls.
    """
    try:
        logger.info(f"Building {config.number_of_personas} persona(s) for industry: {config.industry}")
        
        # Validate inputs
        if not config.industry:
            raise ValueError("Industry must be specified")
        
        if config.number_of_personas < 1:
            logger.warning("Number of personas must be at least 1, setting to 1")
            config.number_of_personas = 1
        
        # In a real implementation, we would retrieve results from previous tools
        # For now, we'll simulate this with mock data
        industry = config.industry.lower()
        
        # Create personas based on the industry
        personas = []
        
        # Persona templates by industry
        persona_templates = {
            "software": [
                {
                    "name": f"{config.persona_name} - IT Decision Maker",
                    "demographics": {
                        "age_range": "35-44",
                        "gender": "Male",
                        "income_level": "$100k-$150k",
                        "education": "Bachelor's or higher",
                        "location": "Urban/Suburban",
                        "job_role": "IT Director/CIO"
                    },
                    "psychographics": {
                        "goals": ["Increase efficiency", "Reduce costs", "Implement cutting-edge technology"],
                        "pain_points": ["Legacy system integration", "Security concerns", "Budget constraints"],
                        "values": ["Reliability", "Innovation", "ROI"],
                        "interests": ["Technology trends", "Digital transformation", "Cloud computing"]
                    },
                    "online_behavior": {
                        "platforms_used": ["LinkedIn", "Tech blogs", "Industry forums"],
                        "content_preferences": ["Whitepapers", "Case studies", "Product demos"],
                        "searched_keywords": [
                            "enterprise software solutions",
                            "IT infrastructure modernization",
                            "cloud migration strategies",
                            "software integration tools",
                            "best enterprise security software"
                        ]
                    },
                    "purchasing_behavior": {
                        "buying_cycle": "3-6 months",
                        "decision_factors": ["ROI", "Integration capabilities", "Support services"],
                        "influences": ["Peer recommendations", "Analyst reports", "Vendor credibility"]
                    }
                },
                {
                    "name": f"{config.persona_name} - Business Executive",
                    "demographics": {
                        "age_range": "45-54",
                        "gender": "Mixed",
                        "income_level": "Over $150k",
                        "education": "Master's or higher",
                        "location": "Urban",
                        "job_role": "CEO/CFO/COO"
                    },
                    "psychographics": {
                        "goals": ["Increase revenue", "Gain competitive advantage", "Improve operational efficiency"],
                        "pain_points": ["Difficulty measuring ROI", "Implementation disruption", "Change management"],
                        "values": ["Growth", "Leadership", "Excellence"],
                        "interests": ["Business strategy", "Competitive intelligence", "Performance metrics"]
                    },
                    "online_behavior": {
                        "platforms_used": ["LinkedIn", "Financial news sites", "Executive forums"],
                        "content_preferences": ["Industry reports", "Success stories", "Executive summaries"],
                        "searched_keywords": [
                            "software ROI calculator",
                            "digital transformation strategy",
                            "business process automation",
                            "executive dashboard software",
                            "competitive advantage through technology"
                        ]
                    },
                    "purchasing_behavior": {
                        "buying_cycle": "6-12 months",
                        "decision_factors": ["Business impact", "Cost", "Competitive advantage"],
                        "influences": ["Board recommendations", "Consultant advice", "Competitor moves"]
                    }
                }
            ],
            "marketing": [
                {
                    "name": f"{config.persona_name} - Marketing Manager",
                    "demographics": {
                        "age_range": "25-34",
                        "gender": "Mixed",
                        "income_level": "$50k-$100k",
                        "education": "Bachelor's",
                        "location": "Urban",
                        "job_role": "Marketing Manager/Director"
                    },
                    "psychographics": {
                        "goals": ["Increase leads", "Improve campaign ROI", "Enhance brand visibility"],
                        "pain_points": ["Measuring campaign effectiveness", "Limited budget", "Keeping up with trends"],
                        "values": ["Creativity", "Results", "Innovation"],
                        "interests": ["Digital marketing", "Content strategy", "Marketing analytics"]
                    },
                    "online_behavior": {
                        "platforms_used": ["LinkedIn", "Twitter", "Marketing blogs"],
                        "content_preferences": ["How-to guides", "Case studies", "Trend reports"],
                        "searched_keywords": [
                            "marketing automation tools",
                            "social media campaign strategies",
                            "content marketing ROI",
                            "email marketing best practices",
                            "lead generation tactics"
                        ]
                    },
                    "purchasing_behavior": {
                        "buying_cycle": "1-3 months",
                        "decision_factors": ["Ease of use", "Features", "Price"],
                        "influences": ["Peer reviews", "Industry blogs", "Free trials"]
                    }
                }
            ],
            "finance": [
                {
                    "name": f"{config.persona_name} - Financial Advisor",
                    "demographics": {
                        "age_range": "35-44",
                        "gender": "Mixed",
                        "income_level": "$100k-$150k",
                        "education": "Bachelor's or higher",
                        "location": "Urban/Suburban",
                        "job_role": "Financial Advisor/Planner"
                    },
                    "psychographics": {
                        "goals": ["Grow client base", "Improve client outcomes", "Streamline operations"],
                        "pain_points": ["Regulatory compliance", "Client acquisition", "Portfolio management time"],
                        "values": ["Trust", "Accuracy", "Client service"],
                        "interests": ["Financial markets", "Wealth management", "Retirement planning"]
                    },
                    "online_behavior": {
                        "platforms_used": ["LinkedIn", "Financial news sites", "Industry forums"],
                        "content_preferences": ["Market analyses", "Regulatory updates", "Client resources"],
                        "searched_keywords": [
                            "financial planning software",
                            "wealth management tools",
                            "client portfolio management",
                            "financial advisor marketing",
                            "retirement planning calculators"
                        ]
                    },
                    "purchasing_behavior": {
                        "buying_cycle": "2-4 months",
                        "decision_factors": ["Compliance features", "Ease of use", "Client-facing tools"],
                        "influences": ["Industry regulations", "Firm requirements", "Client demands"]
                    }
                }
            ]
        }
        
        # Default to a generic industry if the specified one isn't in our data
        selected_industry = industry if industry in persona_templates else "software"
        available_personas = persona_templates.get(selected_industry, persona_templates["software"])
        
        # Generate the requested number of personas
        for i in range(min(config.number_of_personas, len(available_personas))):
            persona = available_personas[i].copy()
            
            # Remove sections if not requested
            if not config.include_demographics:
                persona.pop("demographics", None)
            
            if not config.include_psychographics:
                persona.pop("psychographics", None)
                
            if not config.include_online_behavior:
                persona.pop("online_behavior", None)
                
            personas.append(persona)
        
        # Generate markdown representations for easier human consumption
        personas_markdown = []
        for persona in personas:
            md = f"# {persona['name']}\n\n"
            
            if "demographics" in persona:
                md += "## Demographics\n\n"
                for key, value in persona["demographics"].items():
                    md += f"- **{key.replace('_', ' ').title()}**: {value}\n"
                md += "\n"
                
            if "psychographics" in persona:
                md += "## Psychographics\n\n"
                for key, value in persona["psychographics"].items():
                    if isinstance(value, list):
                        md += f"- **{key.replace('_', ' ').title()}**:\n"
                        for item in value:
                            md += f"  - {item}\n"
                    else:
                        md += f"- **{key.replace('_', ' ').title()}**: {value}\n"
                md += "\n"
                
            if "online_behavior" in persona:
                md += "## Online Behavior\n\n"
                for key, value in persona["online_behavior"].items():
                    if isinstance(value, list):
                        md += f"- **{key.replace('_', ' ').title()}**:\n"
                        for item in value:
                            md += f"  - {item}\n"
                    else:
                        md += f"- **{key.replace('_', ' ').title()}**: {value}\n"
                md += "\n"
                
            if "purchasing_behavior" in persona:
                md += "## Purchasing Behavior\n\n"
                for key, value in persona["purchasing_behavior"].items():
                    if isinstance(value, list):
                        md += f"- **{key.replace('_', ' ').title()}**:\n"
                        for item in value:
                            md += f"  - {item}\n"
                    else:
                        md += f"- **{key.replace('_', ' ').title()}**: {value}\n"
                        
            personas_markdown.append(md)
            
        logger.info(f"Successfully built {len(personas)} personas")
        return {
            "personas": personas,
            "personas_markdown": personas_markdown,
            "summary": f"Created {len(personas)} audience personas for the {industry} industry."
        }
        
    except Exception as e:
        logger.error(f"Error building personas: {str(e)}")
        # Return a partial result or error message
        return {
            "error": str(e),
            "personas": [],
            "personas_markdown": [],
            "summary": f"Error creating personas: {str(e)}"
        } 