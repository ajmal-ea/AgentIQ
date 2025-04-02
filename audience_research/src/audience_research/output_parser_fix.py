"""
SPDX-FileCopyrightText: Copyright (c) 2025
SPDX-License-Identifier: Apache-2.0

This module provides a utility to handle ReAct agent output parsing failures.
"""

import logging
import re
from typing import Any, Dict, Optional, Union

from aiq.agent.react_agent.output_parser import (
    FINAL_ANSWER_ACTION,
    MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE,
    MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
    ReActOutputParser,
    ReActOutputParserException
)
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

REPARSER_PROMPT = """
You are an expert at formatting ReAct agent outputs. The input provided failed to be parsed correctly because it doesn't follow the required format.

The ReAct format must follow these rules:
1. After each "Thought:", there MUST be an "Action:" (unless it's a final answer)
2. After each "Action:", there MUST be an "Action Input:"
3. For final answers, use "Final Answer:" without any Action.

THE ORIGINAL INPUT:
{original_output}

THE ERROR:
{error_message}

Please reformat the input to match the ReAct format exactly. Preserve the original content and intent.
If it's meant to be a tool call, use this exact format:

Thought: <the reasoning>
Action: <tool_name>
Action Input: <the input for the tool>
Observation: <leave this as is>

If it's meant to be a final answer, use this exact format:

Thought: <the reasoning>
Final Answer: <the final answer>

IMPORTANT: 
- Remove any special formatting like **, ###, etc.
- Use plain text format only
- Preserve all the content but format it correctly
- If there's a section that looks like a final report, format it as a Final Answer

REFORMATTED OUTPUT:
"""

CLEANUP_PROMPT = """
You are helping clean up the output of an audience research report. The report contains some error messages and formatting issues that need to be removed.

Original report:
{original_output}

Please clean this up by:
1. Removing any error messages or warnings (like "Invalid Format: Missing 'Action:' after 'Thought:'")
2. Removing any debugging information or technical explanations
3. Preserving ONLY the final audience research report content
4. Making sure the report has proper formatting with headings and sections

Return ONLY the clean, formatted report without any technical errors or explanations.
"""


async def reparse_with_llm(
    original_output: str,
    llm: BaseChatModel,
    error_message: Optional[str] = None
) -> Union[AgentAction, AgentFinish]:
    """
    Attempt to reformat and reparse a failed ReAct agent output using an LLM.
    
    Args:
        original_output: The original output text that failed parsing
        llm: A language model to use for reformatting
        error_message: The error message from the parsing failure

    Returns:
        Either an AgentAction or AgentFinish object
    """
    if error_message is None:
        # Try to determine the error
        if "Action:" not in original_output and "Thought:" in original_output:
            error_message = MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE
        elif "Action:" in original_output and "Action Input:" not in original_output:
            error_message = MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE
        else:
            error_message = "Failed to parse output in ReAct format"

    logger.info(f"Attempting to reparse failed ReAct output with LLM: {error_message}")

    # Clean up special formatting like Markdown headers (###) that might interfere with parsing
    cleaned_output = re.sub(r'#{1,6}\s+', '', original_output)
    cleaned_output = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned_output)
    
    prompt = REPARSER_PROMPT.format(
        original_output=cleaned_output,
        error_message=error_message
    )

    try:
        # Send to LLM for reformatting
        reformatted_output = await llm.ainvoke(prompt)
        logger.debug(f"Reformatted output: {reformatted_output}")

        # Try parsing with the reformatted output
        parser = ReActOutputParser()
        return await parser.aparse(reformatted_output)
    except Exception as e:
        logger.error(f"Failed to reparse with LLM: {e}")
        
        # As fallback, if it looks like a final answer, try to create that
        if "Final Answer:" in original_output or "AUDIENCE RESEARCH REPORT" in original_output:
            logger.info("Fallback: Converting to AgentFinish as it appears to be a final answer")
            
            # Clean up the final answer if needed
            try:
                cleanup_prompt = CLEANUP_PROMPT.format(original_output=original_output)
                cleaned_answer = await llm.ainvoke(cleanup_prompt)
                return AgentFinish({"output": cleaned_answer}, original_output)
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up final answer: {cleanup_error}")
                
            # If cleaning fails, just use the original
            return AgentFinish({"output": original_output}, original_output)
        
        # Create a very simple action as last resort
        logger.info("Fallback: Creating a minimal AgentAction as last resort")
        # Look for anything that might be a tool name
        action_match = re.search(r"(?:Action|Using)\s*:?\s*[`']?([a-zA-Z_]+)", original_output, re.IGNORECASE)
        tool_name = action_match.group(1) if action_match else "invalid_action"
        return AgentAction(tool_name, "{}", original_output)


async def clean_output(
    output: str,
    llm: BaseChatModel
) -> str:
    """
    Clean up the output to remove error messages and formatting issues.
    
    Args:
        output: The output to clean
        llm: A language model to use for cleaning

    Returns:
        The cleaned output
    """
    cleanup_prompt = CLEANUP_PROMPT.format(original_output=output)
    try:
        cleaned_output = await llm.ainvoke(cleanup_prompt)
        return cleaned_output
    except Exception as e:
        logger.error(f"Failed to clean output: {e}")
        return output 