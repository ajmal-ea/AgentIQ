## Overview

**Objective:**  
Develop an autonomous Audience Research Agent that leverages LLMs and integrated web search/data tools to identify and analyze potential target audiences. The agent will execute a series of tasks—from keyword research and competitor analysis to building detailed audience personas—with minimal hardcoding by relying on dynamic prompts and data-driven processes.

**Key Outputs:**  
- Detailed audience persona documents  
- Keyword lists  
- Competitor analysis reports

---

## Architecture & Design Principles

1. **Multi-Agent Structure:**  
   - **Modularity:** Break the overall task into specialized sub-agents:
     - **Keyword Research Agent**
     - **Competitor Analysis Agent**
     - **Demographic Analysis Agent**
     - **Persona Builder Agent**
     - **Audience Research Coordinator (Master Agent)**
   - **Interoperability:** Use a connectivity layer (inspired by NVIDIA AgentIQ) to enable seamless communication and orchestration between agents.

2. **Data and LLM-Driven:**  
   - **Dynamic Prompts:** Leverage LLMs (e.g. GPT-4 variants) to generate queries, validate results, and adjust workflows dynamically.  
   - **Minimal Hardcoding:** Utilize configuration files and modular functions to allow agents to adapt to different data sources and tasks without rewriting code.

3. **Web Search & Information Integration:**  
   - Integrate necessary web search tools (via APIs for Google Search, or specialized marketing research tools) to gather real-time data.  
   - Ensure that the agents can parse, validate, and correlate data (e.g., using retrieval-augmented generation techniques) to support accurate decision-making.

---

## Detailed Task Breakdown

### 1. Keyword Research Agent
- **Objective:** Identify relevant keywords using data from tools such as Google Keyword Planner, SEMrush, and Ahrefs.
- **Subtasks:**
  - Query multiple keyword research APIs.
  - Filter and rank keywords based on search volume, competition, and relevancy.
- **LLM Role:** Parse results, generate follow-up queries, and validate keyword lists.

### 2. Competitor Analysis Agent
- **Objective:** Analyze competitor strategies by gathering data on their target audiences, ad copies, and engagement metrics.
- **Subtasks:**
  - Scrape competitors’ landing pages, ad creatives, and public data.
  - Compare competitor keyword usage, ad formats, and market positioning.
- **LLM Role:** Synthesize findings into a structured competitor analysis report.

### 3. Demographic Analysis Agent
- **Objective:** Gather demographic and psychographic insights from sources such as Google Analytics and market research reports.
- **Subtasks:**
  - Retrieve and parse demographic data.
  - Analyze trends in age, gender, interests, and geographic distribution.
- **LLM Role:** Interpret data to suggest segmentation and provide actionable insights.

### 4. Persona Builder Agent
- **Objective:** Create detailed audience personas combining keyword, competitor, and demographic insights.
- **Subtasks:**
  - Synthesize research findings into persona templates.
  - Include sections for demographics, psychographics, behavior patterns, and pain points.
- **LLM Role:** Generate narrative persona descriptions and validate consistency with the data.

### 5. Audience Research Coordinator (Master Agent)
- **Objective:** Orchestrate the entire workflow, ensure the sequence (either predetermined or dynamically optimized) and compile final outputs.
- **Subtasks:**
  - Invoke each sub-agent sequentially (or based on a dynamic decision process).
  - Integrate outputs from sub-agents into a unified report.
  - Validate the overall output using LLM cross-checks.
- **LLM Role:** Summarize and refine the compiled report, ensuring clarity and actionable recommendations.

---

## Integration & Implementation Guidelines

1. **Configuration & Orchestration:**
   - Define agent roles and tasks in a centralized configuration file (YAML or JSON) to enable easy updates.
   - Use an orchestration framework (similar to NVIDIA AgentIQ’s configuration builder) to manage agent handoffs and task dependencies.

2. **Data & API Integration:**
   - Integrate with marketing data tools via their APIs (Google Keyword Planner, SEMrush, Ahrefs, Google Analytics).
   - Implement fallback mechanisms (e.g., web search scraping) in case API data is incomplete.

3. **LLM Integration:**
   - Use LLMs to generate, parse, and validate data outputs at each stage.
   - Ensure that prompts for each agent are dynamic and can be updated based on real-time data inputs.
   - Implement self-evaluation loops where an agent (or the master agent) re-prompts the LLM if outputs don’t meet quality criteria.

4. **Validation & Minimal Hardcoding:**
   - Rely on LLM-driven parsing for output validation, using few-shot examples to define expected outputs.
   - Keep agent logic abstracted in functions, making the system configurable and scalable without deep hardcoding.

5. **Monitoring & Telemetry:**
   - Incorporate profiling and telemetry tools (inspired by AgentIQ’s observability features) to monitor agent performance, latency, and accuracy.
   - Log key decisions and outcomes to refine agent strategies over time.

---

## Final Outputs & Documentation

- **Audience Persona Documents:** Detailed profiles that include keyword insights, competitor strategies, and demographic analyses.
- **Keyword Lists:** Optimized lists of high-potential keywords for campaign targeting.
- **Competitor Analysis Reports:** Comprehensive reports that detail competitor market positioning and ad strategies.