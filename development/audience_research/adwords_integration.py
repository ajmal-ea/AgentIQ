#!/usr/bin/env python
"""
Google AdWords Integration Script for Audience Research Agent

This script takes the output from the Audience Research Agent and prepares it
for integration with Google AdWords API. In a real implementation, you would use
the Google Ads API client library to create audiences, keywords, and campaigns.

For more information on the Google Ads API, see:
https://developers.google.com/google-ads/api/docs/start
"""
import json
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdWordsIntegrator:
    """
    Class that handles integration with Google AdWords API.
    
    This is a mockup class. In a real implementation, this would use
    the Google Ads API client library to interact with the AdWords API.
    """
    
    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None):
        """
        Initialize the AdWords integrator.
        
        Args:
            client_id: Google API client ID
            client_secret: Google API client secret
        """
        # In a real implementation, these would be used to authenticate with the Google Ads API
        self.client_id = client_id or os.environ.get("GOOGLE_ADS_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("GOOGLE_ADS_CLIENT_SECRET")
        
        # Check if credentials are available
        if not (self.client_id and self.client_secret):
            logger.warning("Google Ads API credentials not found. Running in simulation mode.")
    
    def prepare_audience_segments(self, personas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare audience segments for Google AdWords based on personas.
        
        Args:
            personas: List of persona data from audience research
            
        Returns:
            List of audience segment definitions for AdWords
        """
        if not personas:
            logger.warning("No personas provided to prepare audience segments")
            return []
            
        audience_segments = []
        
        for i, persona in enumerate(personas):
            try:
                # Extract demographic information
                demographics = persona.get("demographics", {})
                age_range = demographics.get("age_range", "25-34")
                gender = demographics.get("gender", "Not specified")
                income_level = demographics.get("income_level", "$50k-$100k")
                location = demographics.get("location", "United States")
                
                # Extract interests from psychographics
                psychographics = persona.get("psychographics", {})
                interests = psychographics.get("interests", [])
                pain_points = psychographics.get("pain_points", [])
                values = psychographics.get("values", [])
                
                # Extract online behavior
                online_behavior = persona.get("online_behavior", {})
                keywords = online_behavior.get("searched_keywords", [])
                platforms = online_behavior.get("platforms_used", [])
                
                # Extract purchasing behavior
                purchasing_behavior = persona.get("purchasing_behavior", {})
                buying_cycle = purchasing_behavior.get("buying_cycle", "Unknown")
                decision_factors = purchasing_behavior.get("decision_factors", [])
                
                # Create an audience segment
                segment = {
                    "name": f"Audience Segment - {persona.get('name', f'Persona {i+1}')}",
                    "demographics": {
                        "age_ranges": [self._normalize_age_range(age_range)],
                        "genders": [self._normalize_gender(gender)],
                        "income_brackets": [self._normalize_income(income_level)],
                        "locations": [location]
                    },
                    "interests": interests,
                    "custom_intent": {
                        "keywords": keywords,
                        "urls": [f"https://{platform.lower().replace(' ', '')}.com" for platform in platforms if platform],
                        "topics": interests + values
                    },
                    "in_market": list(set(pain_points + decision_factors)),
                    "buying_cycle": self._normalize_buying_cycle(buying_cycle)
                }
                
                audience_segments.append(segment)
                logger.info(f"Created audience segment for persona: {persona.get('name', f'Persona {i+1}')}")
            except Exception as e:
                logger.error(f"Error processing persona {i}: {str(e)}")
                continue
            
        return audience_segments
    
    def prepare_keywords(self, personas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare keywords for Google AdWords based on personas.
        
        Args:
            personas: List of persona data from audience research
            
        Returns:
            List of keyword definitions for AdWords
        """
        if not personas:
            logger.warning("No personas provided to prepare keywords")
            return []
            
        all_keywords = []
        
        for i, persona in enumerate(personas):
            try:
                # Get keywords from persona
                online_behavior = persona.get("online_behavior", {})
                keywords = online_behavior.get("searched_keywords", [])
                
                if not keywords:
                    logger.warning(f"No keywords found for persona {i}")
                    continue
                
                # Get persona details for bidding adjustments
                demographics = persona.get("demographics", {})
                persona_name = persona.get("name", f"Persona {i+1}")
                
                # Determine base bid based on income level
                income_level = demographics.get("income_level", "$50k-$100k")
                if "over $150k" in income_level.lower():
                    base_bid = 750000  # $0.75
                elif "$100k" in income_level.lower():
                    base_bid = 600000  # $0.60
                else:
                    base_bid = 500000  # $0.50
                
                # Create keyword entries for AdWords with different match types
                for keyword in keywords:
                    # Exact match - highest bid
                    exact_keyword = {
                        "keyword": keyword,
                        "match_type": "EXACT", 
                        "status": "ENABLED",
                        "bid_amount_micros": int(base_bid * 1.2),
                        "persona": persona_name
                    }
                    all_keywords.append(exact_keyword)
                    
                    # Phrase match - medium bid
                    phrase_keyword = {
                        "keyword": keyword,
                        "match_type": "PHRASE",
                        "status": "ENABLED",
                        "bid_amount_micros": base_bid,
                        "persona": persona_name
                    }
                    all_keywords.append(phrase_keyword)
                    
                    # Broad match - lower bid
                    if len(keyword.split()) > 1:  # Only use broad match for multi-word keywords
                        broad_keyword = {
                            "keyword": keyword,
                            "match_type": "BROAD",
                            "status": "ENABLED",
                            "bid_amount_micros": int(base_bid * 0.8),
                            "persona": persona_name
                        }
                        all_keywords.append(broad_keyword)
                
                logger.info(f"Created {len(keywords) * 2} keyword variations for persona: {persona_name}")
            except Exception as e:
                logger.error(f"Error preparing keywords for persona {i}: {str(e)}")
                continue
                
        return all_keywords
    
    def prepare_ad_copy_suggestions(self, personas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate ad copy suggestions based on personas.
        
        Args:
            personas: List of persona data from audience research
            
        Returns:
            List of ad copy suggestions
        """
        if not personas:
            logger.warning("No personas provided to prepare ad copy")
            return []
        
        ad_suggestions = []
        
        for i, persona in enumerate(personas):
            try:
                # Extract data for ad copy generation
                name = persona.get("name", f"Target Audience {i+1}")
                
                # Extract psychographics
                psychographics = persona.get("psychographics", {})
                pain_points = psychographics.get("pain_points", [])
                values = psychographics.get("values", [])
                goals = psychographics.get("goals", [])
                
                # Extract keywords
                online_behavior = persona.get("online_behavior", {})
                keywords = online_behavior.get("searched_keywords", [])
                platforms = online_behavior.get("platforms_used", [])
                
                # Generate headline ideas
                headlines = []
                
                # Use pain points for headlines
                if pain_points:
                    headlines.append(f"Solve Your {pain_points[0]} Problems")
                    if len(pain_points) > 1:
                        headlines.append(f"End {pain_points[1]} Struggles Today")
                
                # Use values + keywords
                if values and keywords:
                    headlines.append(f"{values[0].title()} {keywords[0].title()}")
                    
                # Use goals for headlines
                if goals:
                    headlines.append(f"Achieve {goals[0]}")
                    
                # Use keywords
                if keywords:
                    headlines.append(f"Premium {keywords[0].title()} Solutions")
                    if len(keywords) > 1:
                        headlines.append(f"Discover {keywords[1].title()}")
                    
                # Fallback headlines if none were generated
                if not headlines:
                    headlines = [
                        "Industry-Leading Solutions",
                        "Discover What Makes Us Different",
                        "Start Your Journey Today"
                    ]
                    
                # Generate description ideas
                descriptions = []
                
                # Combine pain points and values
                if pain_points and values:
                    descriptions.append(f"We understand {pain_points[0]}. Our solution delivers {values[0]}.")
                
                # Use keywords and goals    
                if keywords and goals:
                    descriptions.append(f"Industry-leading {keywords[0]} tailored to help you {goals[0].lower()}. Try it today!")
                
                # Use platforms and pain points
                if platforms and pain_points:
                    descriptions.append(f"Join our community on {platforms[0]} and learn how to overcome {pain_points[0].lower()}.")
                
                # Fallback descriptions if none were generated
                if not descriptions:
                    descriptions = [
                        "Our tailored solutions meet your unique needs. Contact us today for a personalized consultation.",
                        "Discover why our customers choose us. Satisfaction guaranteed or your money back."
                    ]
                    
                # Create ad suggestion
                ad_suggestion = {
                    "persona": name,
                    "headlines": headlines[:3],  # AdWords allows up to 3 headlines
                    "descriptions": descriptions[:2],  # AdWords allows up to 2 descriptions
                    "path1": keywords[0].split()[0].lower() if keywords else "solutions",
                    "path2": "features",
                    "demographic_targeting": {
                        "age_ranges": [persona.get("demographics", {}).get("age_range", "25-34")],
                        "genders": [persona.get("demographics", {}).get("gender", "Not specified")],
                        "locations": [persona.get("demographics", {}).get("location", "United States")]
                    }
                }
                
                ad_suggestions.append(ad_suggestion)
                logger.info(f"Created ad copy suggestions for persona: {name}")
            except Exception as e:
                logger.error(f"Error creating ad copy for persona {i}: {str(e)}")
                continue
                
        return ad_suggestions
    
    def create_adwords_campaign(self, 
                                campaign_name: str, 
                                audience_segments: List[Dict[str, Any]], 
                                keywords: List[Dict[str, Any]], 
                                ad_suggestions: List[Dict[str, Any]], 
                                budget_micros: int = 10000000) -> Dict[str, Any]:
        """
        Create a Google AdWords campaign (simulation).
        
        Args:
            campaign_name: Name of the campaign
            audience_segments: Audience segments to target
            keywords: Keywords to bid on
            ad_suggestions: Ad copy suggestions
            budget_micros: Daily budget in micros (millionths of the account currency)
            
        Returns:
            Campaign data that would be created in AdWords
        """
        # Validate inputs
        if not campaign_name:
            campaign_name = "Audience Research Campaign"
            
        if not audience_segments:
            logger.warning("No audience segments provided for campaign")
            
        if not keywords:
            logger.warning("No keywords provided for campaign")
            
        if not ad_suggestions:
            logger.warning("No ad suggestions provided for campaign")
        
        # In a real implementation, this would make API calls to create the campaign
        logger.info(f"Creating AdWords campaign: {campaign_name}")
        logger.info(f"- with {len(audience_segments)} audience segments")
        logger.info(f"- with {len(keywords)} keywords")
        logger.info(f"- with {len(ad_suggestions)} ad suggestions")
        
        # Simulate campaign creation
        campaign = {
            "name": campaign_name,
            "status": "PAUSED",  # Start as paused so it can be reviewed before going live
            "budget_micros": budget_micros,
            "audience_segments": audience_segments,
            "keywords": keywords,
            "ad_suggestions": ad_suggestions,
            "creation_time": self._get_current_timestamp(),
            "settings": {
                "bidding_strategy": "MAXIMIZE_CONVERSIONS",
                "language_targeting": ["ENGLISH"],
                "network_targeting": {
                    "search_network": True,
                    "display_network": True
                }
            }
        }
        
        return campaign
    
    def export_campaign_to_file(self, campaign: Dict[str, Any], output_file: str) -> None:
        """
        Export the campaign definition to a file.
        
        Args:
            campaign: Campaign data
            output_file: Path to the output file
        """
        # Create parent directories if they don't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_file, 'w') as f:
                json.dump(campaign, f, indent=2)
            logger.info(f"Exported campaign to {output_file}")
        except Exception as e:
            logger.error(f"Error exporting campaign to {output_file}: {str(e)}")
            raise
    
    def process_audience_research_results(self, 
                                          result_file: str, 
                                          campaign_name: str, 
                                          output_file: str) -> Dict[str, Any]:
        """
        Process audience research results from a file and create an AdWords campaign.
        
        Args:
            result_file: Path to the audience research result JSON file
            campaign_name: Name for the AdWords campaign
            output_file: Path to save the campaign definition
            
        Returns:
            Campaign data that was created
        """
        try:
            # Load the research results
            with open(result_file, 'r') as f:
                result_data = json.load(f)
                
            # Check if we have personas data
            personas = result_data.get("personas", [])
            if not personas:
                logger.warning("No personas found in the result file")
                return {"error": "No personas found in result file"}
                
            # Prepare AdWords components
            audience_segments = self.prepare_audience_segments(personas)
            keywords = self.prepare_keywords(personas)
            ad_suggestions = self.prepare_ad_copy_suggestions(personas)
            
            # Create the campaign
            campaign = self.create_adwords_campaign(
                campaign_name=campaign_name,
                audience_segments=audience_segments,
                keywords=keywords,
                ad_suggestions=ad_suggestions
            )
            
            # Export the campaign definition
            self.export_campaign_to_file(campaign, output_file)
            
            return campaign
        except Exception as e:
            logger.error(f"Error processing audience research results: {str(e)}")
            return {"error": str(e)}
    
    def _normalize_age_range(self, age_range: str) -> str:
        """
        Normalize age range to match Google Ads API format.
        
        Args:
            age_range: Age range string (e.g., "25-34")
            
        Returns:
            Normalized age range for Google Ads API
        """
        # In a real implementation, this would map to valid Google Ads API values
        age_mapping = {
            "18-24": "AGE_RANGE_18_24",
            "25-34": "AGE_RANGE_25_34",
            "35-44": "AGE_RANGE_35_44",
            "45-54": "AGE_RANGE_45_54",
            "55+": "AGE_RANGE_55_UP",
            "55-64": "AGE_RANGE_55_64",
            "65+": "AGE_RANGE_65_UP"
        }
        return age_mapping.get(age_range, "AGE_RANGE_UNDETERMINED")
    
    def _normalize_gender(self, gender: str) -> str:
        """
        Normalize gender to match Google Ads API format.
        
        Args:
            gender: Gender string (e.g., "Male")
            
        Returns:
            Normalized gender for Google Ads API
        """
        # In a real implementation, this would map to valid Google Ads API values
        gender = gender.lower()
        if "male" in gender and "female" not in gender:
            return "MALE"
        elif "female" in gender and "male" not in gender:
            return "FEMALE"
        else:
            return "UNDETERMINED"
    
    def _normalize_income(self, income_level: str) -> str:
        """
        Normalize income level to match Google Ads API format.
        
        Args:
            income_level: Income level string (e.g., "$50k-$100k")
            
        Returns:
            Normalized income level for Google Ads API
        """
        # In a real implementation, this would map to valid Google Ads API values
        income_level = income_level.lower()
        
        if "under $50k" in income_level or "< $50k" in income_level:
            return "INCOME_RANGE_1"
        elif "$50k-$100k" in income_level or "50-100" in income_level:
            return "INCOME_RANGE_2" 
        elif "$100k-$150k" in income_level or "100-150" in income_level:
            return "INCOME_RANGE_3"
        elif "over $150k" in income_level or "> $150k" in income_level:
            return "INCOME_RANGE_4"
        else:
            return "INCOME_RANGE_UNDETERMINED"
    
    def _normalize_buying_cycle(self, buying_cycle: str) -> str:
        """
        Normalize buying cycle to match Google Ads API format.
        
        Args:
            buying_cycle: Buying cycle string (e.g., "3-6 months")
            
        Returns:
            Normalized buying cycle for Google Ads API
        """
        # In a real implementation, this would map to valid Google Ads API values
        buying_cycle = buying_cycle.lower()
        
        if "1" in buying_cycle and "month" in buying_cycle:
            return "IN_MARKET"
        elif "2" in buying_cycle and "month" in buying_cycle:
            return "IN_MARKET" 
        elif "3" in buying_cycle and "month" in buying_cycle:
            return "IN_MARKET"
        elif "6" in buying_cycle and "month" in buying_cycle:
            return "CONSIDERATION"
        elif "12" in buying_cycle or "year" in buying_cycle:
            return "AWARENESS"
        else:
            return "GENERAL"
            
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        from datetime import datetime
        return datetime.now().isoformat()


def main():
    """
    Main function to run the AdWords integration from the command line.
    """
    parser = argparse.ArgumentParser(description="Process audience research results and create AdWords campaign")
    parser.add_argument("--result-file", type=str, required=True, help="Path to the audience research result JSON file")
    parser.add_argument("--campaign-name", type=str, default="Audience Research Campaign", help="Name for the AdWords campaign")
    parser.add_argument("--output-file", type=str, default="./adwords_campaign.json", help="Path to save the campaign definition")
    parser.add_argument("--client-id", type=str, help="Google API client ID (optional)")
    parser.add_argument("--client-secret", type=str, help="Google API client secret (optional)")
    args = parser.parse_args()
    
    # Create an AdWords integrator
    integrator = AdWordsIntegrator(
        client_id=args.client_id,
        client_secret=args.client_secret
    )
    
    # Process the audience research results
    try:
        campaign = integrator.process_audience_research_results(
            result_file=args.result_file,
            campaign_name=args.campaign_name,
            output_file=args.output_file
        )
        
        if "error" in campaign:
            logger.error(f"Error processing audience research results: {campaign['error']}")
            sys.exit(1)
        else:
            logger.info(f"Successfully created AdWords campaign: {args.campaign_name}")
            logger.info(f"Campaign definition saved to: {args.output_file}")
    except Exception as e:
        logger.error(f"Error processing audience research results: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 