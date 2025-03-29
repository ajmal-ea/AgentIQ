# Third-Party Services Used by Audience Research Agent

This document outlines all external services used by the Audience Research Agent.

## API Services

### Tavily AI
- **Purpose**: Web search API used for gathering real-time audience research data.
- **Website**: [Tavily](https://tavily.com/)
- **Documentation**: [Tavily API Documentation](https://docs.tavily.com/)
- **Pricing**: Free tier available (100 searches/day), paid plans for higher volume.
- **Implementation Notes**: Core integration point for all research functions.
- **Usage Notes**: Primary source for web-based data collection. Used for demographic research, persona research, keyword research, and competitor analysis.

### NVIDIA API (NeMo Models)
- **Purpose**: Primary LLM provider for analyzing web search data and generating structured insights.
- **Website**: [NVIDIA AI Foundation Models](https://www.nvidia.com/en-us/ai-data-science/foundation-models/)
- **Documentation**: [NVIDIA API Documentation](https://build.nvidia.com/explore/discover)
- **Pricing**: Free and paid tiers available.
- **Implementation Notes**: Used for deep analysis of web search results and structuring data into audience insights.
- **Usage Notes**: Primary LLM for analyzing demographic data, building personas, and competitor analysis.

### Groq API
- **Purpose**: Secondary LLM provider for validation and quality assurance.
- **Website**: [Groq](https://console.groq.com/)
- **Documentation**: [Groq API Documentation](https://console.groq.com/docs/quickstart)
- **Pricing**: Free tier available, paid plans for production use.
- **Implementation Notes**: Used for validating outputs from primary LLM to ensure accuracy and quality.
- **Usage Notes**: Validates demographic analysis, personas, and competitor insights before returning to users.

## Data Sources (accessed via Tavily API)

### Keyword Research
- Industry publications
- Marketing blogs
- SEO tools and websites
- Social media trends via news sites

### Competitor Analysis
- Company websites and blogs
- Financial news sources
- Market research reports
- Business news outlets
- Industry reviews and comparisons
- Review aggregators (Trustpilot, G2, Capterra)

### Demographic Analysis
- Market research sites
- Census data
- Industry reports
- Consumer behavior studies

### Persona Building
- Professional social networks
- Industry forums and communities
- Customer journey mapping resources
- Job boards and career sites

## Service Usage Notes

- All third-party services are accessed via API calls and do not involve direct user authentication.
- Tavily's free tier (100 searches/day) is sufficient for testing and moderate usage.
- NVIDIA and Groq's free tiers provide adequate capacity for analysis with rate limiting implemented.
- Error handling and rate limiting have been implemented for all services.
- A caching layer minimizes repeat API calls for similar searches.
- All services include fallback mechanisms for resilience. 