"""
SPDX-FileCopyrightText: Copyright (c) 2025
SPDX-License-Identifier: Apache-2.0

This module provides a custom ReAct Agent that handles output parsing failures.
"""

import logging
from typing import List, Optional

from aiq.agent.react_agent.agent import ReActAgentGraph
from aiq.agent.react_agent.output_parser import ReActOutputParserException
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool

from .output_parser_fix import reparse_with_llm

logger = logging.getLogger(__name__)


class EnhancedReActAgentGraph(ReActAgentGraph):
    """A ReAct Agent Graph with enhanced output parsing capabilities.
    
    This agent automatically retries parsing failures using an LLM to fix the format.
    """

    def __init__(self,
                 llm: BaseChatModel,
                 prompt: ChatPromptTemplate,
                 tools: list[BaseTool],
                 use_tool_schema: bool = True,
                 callbacks: Optional[List[AsyncCallbackHandler]] = None,
                 detailed_logs: bool = False,
                 retry_parsing_errors: bool = True,
                 max_retries: int = 3,
                 use_llm_for_reparsing: bool = True):
        """Initialize the enhanced ReAct agent.
        
        Args:
            llm: The language model to use for the agent
            prompt: The chat prompt template
            tools: List of tools available to the agent
            use_tool_schema: Whether to include tool schema in descriptions
            callbacks: List of callbacks for tracking execution
            detailed_logs: Whether to log detailed information
            retry_parsing_errors: Whether to retry parsing errors
            max_retries: Maximum number of retries for parsing errors
            use_llm_for_reparsing: Whether to use an LLM to fix parsing errors
        """
        super().__init__(llm=llm,
                         prompt=prompt,
                         tools=tools,
                         use_tool_schema=use_tool_schema,
                         callbacks=callbacks,
                         detailed_logs=detailed_logs,
                         retry_parsing_errors=retry_parsing_errors,
                         max_retries=max_retries)
        self.use_llm_for_reparsing = use_llm_for_reparsing

    async def agent_node(self, state):
        """Override the agent_node method to add LLM-based parsing error handling.
        
        This method extends the original agent_node to use an LLM to fix parsing errors
        when they occur, rather than simply retrying with the original error message.
        """
        try:
            logger.debug("Starting the Enhanced ReAct Agent Node")
            # keeping a working state allows us to resolve parsing errors without polluting the agent scratchpad
            # the agent "forgets" about the parsing error after solving it - prevents hallucinations in next cycles
            working_state = []
            for attempt in range(1, self.max_tries + 1):
                # the first time we are invoking the ReAct Agent, it won't have any intermediate steps / agent thoughts
                if len(state.agent_scratchpad) == 0 and len(working_state) == 0:
                    # the user input comes from the "messages" state channel
                    if len(state.messages) == 0:
                        raise RuntimeError('No input received in state: "messages"')
                    # to check is any human input passed or not, if no input passed Agent will return the state
                    if state.messages[0].content.strip() == "":
                        logger.error("No human input passed to the agent.")
                        state.messages += [AIMessage(content="No human input recieved to the agent, Please ask a valid question.")]
                        return state
                    question = state.messages[0].content
                    logger.info("Querying agent, attempt: %d", attempt)
                    output_message = ""
                    async for event in self.agent.astream({"question": question},
                                                          config=RunnableConfig(callbacks=self.callbacks)):
                        output_message += event.content
                    output_message = AIMessage(content=output_message)
                    if self.detailed_logs:
                        logger.info("The user's question was: %s", question)
                        logger.info("The agent's thoughts are:\n%s", output_message.content)
                else:
                    # ReAct Agents require agentic cycles
                    # in an agentic cycle, preserve the agent's thoughts from the previous cycles,
                    # and give the agent the response from the tool it called
                    agent_scratchpad = []
                    for index, intermediate_step in enumerate(state.agent_scratchpad):
                        agent_thoughts = AIMessage(content=intermediate_step.log)
                        agent_scratchpad.append(agent_thoughts)
                        tool_response = HumanMessage(content=state.tool_responses[index].content)
                        agent_scratchpad.append(tool_response)
                    agent_scratchpad += working_state
                    question = state.messages[0].content
                    logger.info("Querying agent, attempt: %d", attempt)
                    output_message = ""
                    async for event in self.agent.astream({
                            "question": question, "agent_scratchpad": agent_scratchpad
                    },
                                                          config=RunnableConfig(callbacks=self.callbacks)):
                        output_message += event.content
                    output_message = AIMessage(content=output_message)
                    if self.detailed_logs:
                        logger.debug("The user's question was: %s", question)
                        logger.debug("The agent's scratchpad (with tool result) was:\n%s", agent_scratchpad)
                        logger.info("\n\nThe agent's thoughts are:\n%s", output_message.content)
                try:
                    # check if the agent has the final answer yet
                    logger.debug("Successfully obtained agent response. Parsing agent's response")
                    from aiq.agent.react_agent.output_parser import ReActOutputParser
                    agent_output = await ReActOutputParser().aparse(output_message.content)
                    logger.debug("Successfully parsed agent's response")
                    if attempt > 1:
                        logger.info("Successfully parsed agent response after %d attempts", attempt)
                    if isinstance(agent_output, AgentFinish):
                        final_answer = agent_output.return_values.get('output', output_message.content)
                        logger.debug("The agent has finished, and has the final answer")
                        # this is where we handle the final output of the Agent, we can clean-up/format/postprocess here
                        # the final answer goes in the "messages" state channel
                        state.messages += [AIMessage(content=final_answer)]
                    else:
                        # the agent wants to call a tool, ensure the thoughts are preserved for the next agentic cycle
                        agent_output.log = output_message.content
                        logger.debug("The agent wants to call a tool: %s", agent_output.tool)
                        state.agent_scratchpad += [agent_output]
                    return state
                except ReActOutputParserException as ex:
                    # The original parser failed. Let's try using our LLM-based reparser
                    logger.warning("Error parsing agent output\nObservation:%s\nAgent Output:\n%s",
                                   ex.observation,
                                   output_message.content)
                    
                    if self.use_llm_for_reparsing:
                        logger.info("Attempting to fix the format using LLM")
                        try:
                            # Use our LLM-based reparser to fix the format issues
                            agent_output = await reparse_with_llm(
                                original_output=output_message.content,
                                llm=self.llm,
                                error_message=ex.observation
                            )
                            
                            # If we successfully reparsed, we can proceed
                            logger.info("Successfully reparsed using LLM")
                            if isinstance(agent_output, AgentFinish):
                                final_answer = agent_output.return_values.get('output', output_message.content)
                                logger.debug("The agent has finished, and has the final answer")
                                state.messages += [AIMessage(content=final_answer)]
                            else:
                                # the agent wants to call a tool
                                agent_output.log = output_message.content
                                logger.debug("The agent wants to call a tool: %s", agent_output.tool)
                                state.agent_scratchpad += [agent_output]
                            return state
                            
                        except Exception as reparse_ex:
                            logger.error(f"LLM-based reparsing also failed: {reparse_ex}")
                            # Continue with the standard retry approach as fallback
                    
                    # Standard retry approach (from original implementation)
                    if attempt == self.max_tries:
                        logger.exception(
                            "Failed to parse agent output after %d attempts, consider enabling or "
                            "increasing max_retries",
                            attempt,
                            exc_info=True)
                        # the final answer goes in the "messages" state channel
                        output_message.content = ex.observation + '\n' + output_message.content
                        state.messages += [output_message]
                        return state
                    # retry parsing errors, if configured
                    logger.info("Retrying ReAct Agent, including output parsing Observation")
                    working_state.append(output_message)
                    working_state.append(HumanMessage(content=ex.observation))
        except Exception as ex:
            logger.exception("Failed to call agent_node: %s", ex, exc_info=True)
            raise ex 