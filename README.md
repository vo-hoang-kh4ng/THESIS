# Multi-AI-Agent Systems with CrewAI Guidelines

This repository demonstrates a multi-level AI Agent system built with CrewAI that performs comprehensive social media monitoring and brand analysis. The system applies key CrewAI principles—**Role Playing, Focus, Tools, Cooperation, Guardrails, Memory**—to create a robust chain-of-agents framework. It integrates several specialized agents and supporting agents to not only gather and analyze data but also aggregate, refine, and optimize the final output.

## Features

- **Multi-Level Agent Architecture:**
  - **Specialist Agents:**
    - **Social Media Researcher:** Gathers comprehensive information about the brand from diverse sources.
    - **Social Media Monitor:** Monitors social media platforms and extracts detailed engagement metrics.
    - **Sentiment Analyzer:** Performs in-depth sentiment analysis on social media mentions.
    - **Report Generator:** Generates structured reports based on the collected analyses.
  - **Coordinator Agent:** Aggregates and synthesizes outputs from all specialist agents to produce a final comprehensive analysis.
  - **Supporting Agents:**
    - **Support Agent:** Provides additional context and clarifications to ensure completeness of the analysis.
    - **Memory Agent:** Stores important insights and reasoning traces for future reference.
    - **Re-ranking Agent:** Evaluates and reorders outputs from other agents to produce the optimal final report.

- **CrewAI Guidelines Implementation:**
  - **Role Playing:** Each agent is assigned a clear role, goal, and backstory to ensure they "act" according to their designated responsibilities.
  - **Focus:** Agents are prompted to concentrate on their specific tasks, delivering complete and accurate results.
  - **Tools:** Agents can leverage integrated tools (e.g., `SerperDevTool`) for data retrieval and analysis.
  - **Cooperation:** Agents can cooperate and delegate tasks where necessary, ensuring a cohesive workflow.
  - **Guardrails:** Built-in guidelines ensure that all responses are thorough and free from assumptions.
  - **Memory:** The system supports storing and retrieving past interactions to provide context-aware decision-making.

## Pipeline Architecture

