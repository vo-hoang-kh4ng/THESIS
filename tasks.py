from crewai import Task
from typing import List, Optional, Dict
import logging
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.store.memory import InMemoryStore

logger = logging.getLogger(__name__)

# Set up memory store
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)

# Create memory tools
memory_tools = [
    create_manage_memory_tool(namespace="memories", store=store),
    create_search_memory_tool(namespace="memories", store=store),
]

def create_tasks(brand_name: str, agents: List, twitter_data: Optional[Dict] = None) -> List[Task]:
    """
    Create tasks for social media monitoring with enhanced memory management.
    
    Args:
        brand_name: Name of the brand to monitor
        agents: List of agents to assign tasks to
        twitter_data: Optional Twitter data for analysis
        
    Returns:
        List of configured tasks
    """
    tasks = []
    
    # Research Task
    research_task = Task(
        description=(
            f"Conduct a systematic investigation of {brand_name}'s online presence across multiple platforms. "
            f"Focus on recent social media activity, news coverage, and market trends. "
            f"Use memory tools to store and retrieve relevant historical data."
        ),
        agent=next(agent for agent in agents if agent.role == "Social Media Researcher"),
        expected_output=(
            "A comprehensive research report including:\n"
            "- Recent social media activity\n"
            "- News coverage analysis\n"
            "- Market trends and insights\n"
            "- Historical context from memory"
        ),
        async_execution=False
    )
    tasks.append(research_task)
    
    # Monitoring Task
    monitoring_task = Task(
        description=(
            f"Monitor and analyze social media metrics for {brand_name} across all platforms. "
            f"Track engagement, reach, and performance indicators. "
            f"Use memory tools to compare with historical performance."
        ),
        agent=next(agent for agent in agents if agent.role == "Social Media Monitor"),
        expected_output=(
            "A detailed monitoring report including:\n"
            "- Engagement metrics\n"
            "- Reach statistics\n"
            "- Performance indicators\n"
            "- Historical comparisons"
        ),
        async_execution=False
    )
    tasks.append(monitoring_task)
    
    # Sentiment Analysis Task
    sentiment_task = Task(
        description=(
            f"Perform sentiment analysis on social media mentions of {brand_name}. "
            f"Analyze emotions, tone, and sentiment trends. "
            f"Use memory tools to identify sentiment patterns and changes."
        ),
        agent=next(agent for agent in agents if agent.role == "Sentiment Analyzer"),
        expected_output=(
            "A comprehensive sentiment analysis including:\n"
            "- Sentiment distribution\n"
            "- Emotion analysis\n"
            "- Trend identification\n"
            "- Historical sentiment patterns"
        ),
        async_execution=False
    )
    tasks.append(sentiment_task)
    
    # Report Generation Task
    report_task = Task(
        description=(
            f"Generate a comprehensive analysis report for {brand_name} based on all collected data. "
            f"Include executive summary, detailed analysis, and recommendations. "
            f"Use memory tools to ensure consistency with historical reports."
        ),
        agent=next(agent for agent in agents if agent.role == "Report Generator"),
        expected_output=(
            "A detailed report including:\n"
            "- Executive summary\n"
            "- Data analysis\n"
            "- Recommendations\n"
            "- Historical context"
        ),
        async_execution=False
    )
    tasks.append(report_task)
    
    # Coordination Task
    coordinator_task = Task(
        description=(
            f"Coordinate and synthesize outputs from all specialist agents for {brand_name}. "
            f"Ensure consistency and completeness of the final analysis. "
            f"Use memory tools to maintain data coherence."
        ),
        agent=next(agent for agent in agents if agent.role == "Coordinator"),
        expected_output=(
            "A coordinated analysis including:\n"
            "- Synthesized insights\n"
            "- Cross-validation results\n"
            "- Consistency checks\n"
            "- Historical alignment"
        ),
        async_execution=False
    )
    tasks.append(coordinator_task)
    
    # Support Task
    support_task = Task(
        description=(
            f"Provide supplementary support and clarifications for {brand_name}'s analysis. "
            f"Ensure accuracy and completeness of all findings. "
            f"Use memory tools to provide historical context."
        ),
        agent=next(agent for agent in agents if agent.role == "Support Agent"),
        expected_output=(
            "Support documentation including:\n"
            "- Clarifications\n"
            "- Additional context\n"
            "- Historical references\n"
            "- Validation checks"
        ),
        async_execution=False
    )
    tasks.append(support_task)
    
    # Memory Task
    memory_task = Task(
        description=(
            f"Store and organize key insights from {brand_name}'s analysis. "
            f"Maintain a comprehensive knowledge base for future reference. "
            f"Use memory tools to ensure proper data organization."
        ),
        agent=next(agent for agent in agents if agent.role == "Memory Agent"),
        expected_output=(
            "Memory documentation including:\n"
            "- Stored insights\n"
            "- Pattern recognition\n"
            "- Knowledge organization\n"
            "- Historical connections"
        ),
        async_execution=False
    )
    tasks.append(memory_task)
    
    # Re-ranking Task
    reranking_task = Task(
        description=(
            f"Optimize and re-rank the final report for {brand_name}. "
            f"Ensure critical information is properly prioritized. "
            f"Use memory tools to maintain consistency with historical reports."
        ),
        agent=next(agent for agent in agents if agent.role == "Re-ranking Agent"),
        expected_output=(
            "Optimized report including:\n"
            "- Prioritized information\n"
            "- Critical insights\n"
            "- Historical alignment\n"
            "- Consistency checks"
        ),
        async_execution=False,
        output_file="final_report.md"
    )
    tasks.append(reranking_task)
    
    # Crisis Detection Task
    crisis_task = Task(
        description=(
            f"Monitor and analyze potential crisis situations for {brand_name}. "
            f"Identify early warning signs and recommend response strategies. "
            f"Use memory tools to compare with historical crisis patterns."
        ),
        agent=next(agent for agent in agents if agent.role == "Crisis Detector"),
        expected_output=(
            "Crisis analysis including:\n"
            "- Warning signs\n"
            "- Impact assessment\n"
            "- Response recommendations\n"
            "- Historical crisis patterns"
        ),
        async_execution=False
    )
    tasks.append(crisis_task)
    
    return tasks