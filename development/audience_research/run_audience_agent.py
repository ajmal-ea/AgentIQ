#!/usr/bin/env python
"""
Script to run the Audience Research Agent for Google AdWords

This script serves as a convenience wrapper around the aiq CLI commands,
specifically tailored for audience research.
"""
import asyncio
import argparse
import os
import sys
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dotenv import load_dotenv

load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_aiq_command(command: str, config_file: str, input_message: str, output_dir: str = "./output") -> Dict[str, Any]:
    """
    Run an AgentIQ command with the given parameters
    
    Args:
        command: The AgentIQ command to run (run, serve, eval)
        config_file: Path to the workflow configuration file
        input_message: The input message or query to process
        output_dir: Directory to save output files
        
    Returns:
        The command result
    """
    # Ensure the output directory exists
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare the base command
    base_cmd = ["aiq", command, "--config_file", config_file]
    
    if command == "run":
        # For run, we use the input message directly
        input_file = output_dir_path / "input.txt"
        with open(input_file, "w") as f:
            f.write(input_message)
        base_cmd.extend(["--input_file", str(input_file)])
        
        try:
            # Run the command and capture output
            logger.info(f"Running command: {' '.join(base_cmd)}")
            result = subprocess.run(base_cmd, capture_output=True, text=True, check=True)
            
            # Parse the output to extract the result
            output_text = result.stdout
            
            # Save raw output
            output_file = output_dir_path / "aiq_output.txt"
            with open(output_file, "w") as f:
                f.write(output_text)
            
            # Process and return the result
            try:
                # Try to extract JSON from the result if it exists
                json_start = output_text.find('{')
                json_end = output_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = output_text[json_start:json_end]
                    result_data = json.loads(json_str)
                else:
                    # No JSON found, use the text as is
                    result_data = {"response": output_text}
            except json.JSONDecodeError:
                # If JSON parsing fails, treat the result as plain text
                result_data = {"response": output_text}
            
            return {
                "success": True,
                "data": result_data,
                "output_file": str(output_file)
            }
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running aiq {command}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "stdout": e.stdout,
                "stderr": e.stderr
            }
    
    elif command == "serve":
        # For serve, start the server without input
        logger.info(f"Starting server with command: {' '.join(base_cmd)}")
        print(f"Starting AgentIQ server with config: {config_file}")
        print(f"You can access the server at http://localhost:8000")
        print(f"To interact with the server, use the following curl command:")
        print(f"curl --request POST \\")
        print(f"  --url http://localhost:8000/generate \\")
        print(f"  --header 'Content-Type: application/json' \\")
        print(f"  --data '{{\"input_message\": \"{input_message}\"}}'\n")
        
        # Start the server (non-blocking)
        process = subprocess.Popen(base_cmd)
        
        return {
            "success": True,
            "message": "Server started",
            "process": process,
            "config_file": config_file
        }
    
    elif command == "eval":
        # For eval, we need to provide a reference answer
        # This is just a basic implementation - in practice, you would want to provide real reference data
        ref_file = output_dir_path / "reference.txt"
        ref_text = "Detailed audience persona for targeted marketing."
        with open(ref_file, "w") as f:
            f.write(ref_text)
        
        eval_cmd = base_cmd + ["--input", input_message, "--reference", str(ref_file)]
        
        try:
            logger.info(f"Running evaluation command: {' '.join(eval_cmd)}")
            result = subprocess.run(eval_cmd, capture_output=True, text=True, check=True)
            
            # Save evaluation output
            eval_file = output_dir_path / "evaluation_result.txt"
            with open(eval_file, "w") as f:
                f.write(result.stdout)
            
            return {
                "success": True,
                "output": result.stdout,
                "eval_file": str(eval_file)
            }
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running aiq {command}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "stdout": e.stdout,
                "stderr": e.stderr
            }
    
    else:
        return {
            "success": False,
            "error": f"Unsupported command: {command}"
        }

def process_and_save_results(result: Union[Dict[str, Any], str], industry: str, query: str, output_dir: str) -> Dict[str, Any]:
    """
    Process the results from the AgentIQ command and save them to files
    
    Args:
        result: The result from the AgentIQ command
        industry: The industry used in the research
        query: The query used in the research
        output_dir: Directory to save output files
        
    Returns:
        Dictionary with paths to output files
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    if isinstance(result, dict) and not result.get("success", False):
        # Handle error case
        error_file = output_dir_path / f"{industry}_{query.replace(' ', '_')}_error_log.txt"
        with open(error_file, "w") as f:
            f.write(f"Error in audience research: {result.get('error', 'Unknown error')}\n")
            if "stdout" in result:
                f.write(f"Standard output: {result['stdout']}\n")
            if "stderr" in result:
                f.write(f"Standard error: {result['stderr']}\n")
        
        return {
            "success": False,
            "error_file": str(error_file),
            "error": result.get("error", "Unknown error")
        }
    
    # Handle successful case
    if isinstance(result, dict) and "data" in result:
        data = result["data"]
        
        # Save the full JSON result
        result_file = output_dir_path / f"{industry}_{query.replace(' ', '_')}_audience_research_result.json"
        with open(result_file, "w") as f:
            json.dump(data, f, indent=2)
        
        # Extract and save personas if available
        personas_saved = False
            persona_files = []
        
        if isinstance(data, dict) and "personas_markdown" in data:
            for i, persona_md in enumerate(data["personas_markdown"]):
                persona_file = output_dir_path / f"{industry}_{query.replace(' ', '_')}_persona_{i + 1}.md"
                with open(persona_file, "w") as f:
                    f.write(persona_md)
                persona_files.append(str(persona_file))
                personas_saved = True
        
        # If no personas were found, create a basic one
        if not personas_saved:
            response_text = ""
            
            if isinstance(data, dict):
                response_text = data.get("response", str(data))
            elif isinstance(data, str):
                response_text = data
            else:
                response_text = str(data)
            
            # Check if the response might contain persona information
            if "# " in response_text and "Persona" in response_text:
                persona_file = output_dir_path / f"{industry}_{query.replace(' ', '_')}_persona_extracted.md"
                with open(persona_file, "w") as f:
                    f.write(response_text)
                persona_files.append(str(persona_file))
        else:
                # Create a basic persona
            basic_persona = f"""# {industry.title()} Target Persona

## Overview
Target audience for {query} in the {industry} industry.

## Demographics
- Industry: {industry}

## Key Interests
- {query}
"""
                persona_file = output_dir_path / f"{industry}_{query.replace(' ', '_')}_basic_persona.md"
                with open(persona_file, "w") as f:
                f.write(basic_persona)
                persona_files.append(str(persona_file))
            
        return {
            "success": True,
            "result_file": str(result_file),
            "persona_files": persona_files,
            "raw_output": result.get("output_file")
        }
    
    # Handle string or other result types
    text_result = str(result)
    text_file = output_dir_path / f"{industry}_{query.replace(' ', '_')}_result.txt"
    with open(text_file, "w") as f:
        f.write(text_result)
    
    return {
        "success": True,
        "text_file": str(text_file)
    }

def main():
    """
    Main function to parse arguments and run the audience research agent
    """
    parser = argparse.ArgumentParser(description="Run the Audience Research Agent for Google AdWords")
    parser.add_argument("--industry", type=str, default="software", help="Industry to research (e.g., software, marketing, finance)")
    parser.add_argument("--query", type=str, help="Keyword query for research")
    parser.add_argument("--region", type=str, default="United States", help="Region to focus on")
    parser.add_argument("--competitor", type=str, help="Specific competitor to analyze (optional)")
    parser.add_argument("--personas", type=int, default=1, help="Number of personas to generate")
    parser.add_argument("--persona-name", type=str, default="Target Persona", help="Name for the persona")
    parser.add_argument("--output-dir", type=str, default="./output", help="Directory to save output files")
    parser.add_argument("--config", type=str, default="./workflow.yml", help="Path to workflow configuration file")
    parser.add_argument("--command", type=str, choices=["run", "serve", "eval"], default="run", 
                       help="AgentIQ command to run (run, serve, eval)")
    
    args = parser.parse_args()

    # Use the query if provided, otherwise use the industry
    query = args.query if args.query else args.industry
    
    # Format the input message
    if args.competitor:
        input_message = f"Research audience for {args.industry} industry, focusing on {query}, including competitor analysis for {args.competitor}"
    else:
        input_message = f"Research audience for {args.industry} industry, focusing on {query}"
    
    # Run the appropriate AgentIQ command
    result = run_aiq_command(
        command=args.command,
        config_file=args.config,
        input_message=input_message,
        output_dir=args.output_dir
    )
    
    # For 'run' command, process and save the results
    if args.command == "run":
        output = process_and_save_results(
            result=result,
            industry=args.industry,
            query=query,
            output_dir=args.output_dir
        )
        
        if output.get("success", False):
            logger.info("Audience research completed successfully")
            if "persona_files" in output:
                logger.info(f"Generated personas saved to: {', '.join(output['persona_files'])}")
            if "result_file" in output:
                logger.info(f"Complete results saved to: {output['result_file']}")
    else:
            logger.error(f"Audience research failed: {output.get('error', 'Unknown error')}")
            if "error_file" in output:
                logger.info(f"Error details saved to: {output['error_file']}")
    
    # For 'serve' command, keep the server running
    elif args.command == "serve":
        try:
            # Keep the script running while the server is active
            if "process" in result:
                logger.info("Press Ctrl+C to stop the server")
                result["process"].wait()
        except KeyboardInterrupt:
            if "process" in result:
                logger.info("Stopping server...")
                result["process"].terminate()
                result["process"].wait()
                logger.info("Server stopped")
    
    # For 'eval' command, just display the results
    elif args.command == "eval":
        if result.get("success", False):
            logger.info("Evaluation completed")
            if "eval_file" in result:
                logger.info(f"Evaluation results saved to: {result['eval_file']}")
        else:
            logger.error(f"Evaluation failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main() 