from crewai.flow.flow import Flow, listen, start, router
from pydantic import BaseModel
from typing import Optional, List, Dict
import time
from datetime import datetime
from crewai import Task
from my_agents import create_agents, create_llm
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

class SocialMediaState(BaseModel):
    brand_name: str
    timestamp: str = time.strftime('%Y-%m-%d %H:%M:%S')
    date: str = time.strftime('%Y-%m-%d')
    research_data: Optional[Dict] = None
    monitoring_data: Optional[Dict] = None
    sentiment_data: Optional[Dict] = None
    report_data: Optional[Dict] = None
    crisis_detected: bool = False
    negative_percent: float = 0.0
    top_influencers: List = []
    top_opposers: List = []
    graph: Optional[object] = None
    sentiment_trend: Optional[object] = None
    word_cloud: Optional[object] = None
    response_templates: Optional[List] = None
    crisis_signals: List = []
    memory_data: Optional[Dict] = None

class SocialMediaFlow(Flow[SocialMediaState]):
    def _create_initial_state(self) -> SocialMediaState:
        return SocialMediaState(
            brand_name=self.brand_name,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            date=time.strftime('%Y-%m-%d'),
            crisis_detected=False,
            negative_percent=0.0,
            top_influencers=[],
            top_opposers=[],
            crisis_signals=[],
            memory_data={}
        )

    def __init__(self, brand_name: str):
        self.brand_name = brand_name
        self.llm = create_llm()
        self.agents = create_agents(brand_name, self.llm)
        super().__init__()

    @start
    def research_task(self):
        """Initial research task to gather data."""
        researcher = next(agent for agent in self.agents if agent.role == "Social Media Researcher")
        task = Task(
            description=(
                f"Conduct comprehensive research about {self.state.brand_name} using all available tools. "
                f"Focus on recent social media activity, news coverage, and market trends. "
                f"Use memory tools to store and retrieve relevant historical data."
            ),
            agent=researcher,
            expected_output="Detailed research findings with source links and key insights",
            async_execution=False
        )
        try:
            result = researcher.execute_task(task)
            self.state.research_data = result
            # Store research data in memory
            memory_tools[0].run(f"Research data for {self.state.brand_name}: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in research task: {str(e)}")
            return None

    @listen(research_task)
    def monitoring_task(self, research_result):
        """Monitor social media metrics and engagement."""
        if research_result is None:
            return None
            
        monitor = next(agent for agent in self.agents if agent.role == "Social Media Monitor")
        task = Task(
            description=(
                f"Analyze social media metrics and engagement patterns for {self.state.brand_name}. "
                f"Use the research data to focus on key areas of interest. "
                f"Retrieve and utilize historical monitoring data from memory."
            ),
            agent=monitor,
            expected_output="Comprehensive monitoring report with metrics and trends",
            async_execution=False
        )
        try:
            result = monitor.execute_task(task)
            self.state.monitoring_data = result
            # Store monitoring data in memory
            memory_tools[0].run(f"Monitoring data for {self.state.brand_name}: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in monitoring task: {str(e)}")
            return None

    @listen(monitoring_task)
    def sentiment_task(self, monitoring_result):
        """Perform sentiment analysis."""
        if monitoring_result is None:
            return None
            
        sentiment = next(agent for agent in self.agents if agent.role == "Sentiment Analyzer")
        task = Task(
            description=(
                f"Analyze sentiment and emotions in social media mentions of {self.state.brand_name}. "
                f"Focus on identifying potential crisis signals and sentiment trends. "
                f"Compare current sentiment with historical patterns from memory."
            ),
            agent=sentiment,
            expected_output="Detailed sentiment analysis with crisis indicators",
            async_execution=False
        )
        try:
            result = sentiment.execute_task(task)
            self.state.sentiment_data = result
            
            # Update crisis status
            if "negative_percentage" in result:
                self.state.negative_percent = float(result["negative_percentage"])
                self.state.crisis_detected = self.state.negative_percent > 50
                
            # Store sentiment data in memory
            memory_tools[0].run(f"Sentiment data for {self.state.brand_name}: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in sentiment task: {str(e)}")
            return None

    @router
    def crisis_router(self, sentiment_result):
        """Route to appropriate analysis based on crisis status."""
        if sentiment_result is None:
            return None
            
        # Search memory for similar crisis patterns
        crisis_patterns = memory_tools[1].run(f"Find similar crisis patterns for {self.state.brand_name}")
        if crisis_patterns:
            logger.info(f"Found similar crisis patterns in memory: {crisis_patterns}")
            
        if self.state.crisis_detected:
            return self.crisis_analysis(sentiment_result)
        else:
            return self.normal_analysis(sentiment_result)

    @listen(crisis_router)
    def crisis_analysis(self, sentiment_result):
        """Perform crisis-specific analysis."""
        crisis_detector = next(agent for agent in self.agents if agent.role == "Crisis Detector")
        task = Task(
            description=(
                f"Analyze potential crisis situation for {self.state.brand_name}. "
                f"Focus on identifying root causes, impact assessment, and response recommendations. "
                f"Use historical crisis data from memory to inform analysis."
            ),
            agent=crisis_detector,
            expected_output="Crisis analysis report with actionable recommendations",
            async_execution=False
        )
        try:
            result = crisis_detector.execute_task(task)
            self.state.crisis_signals.extend(result.get("crisis_signals", []))
            # Store crisis analysis in memory
            memory_tools[0].run(f"Crisis analysis for {self.state.brand_name}: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in crisis analysis: {str(e)}")
            return None

    @listen(crisis_router)
    def normal_analysis(self, sentiment_result):
        """Perform standard analysis."""
        report = next(agent for agent in self.agents if agent.role == "Report Generator")
        task = Task(
            description=(
                f"Generate standard analysis report for {self.state.brand_name}. "
                f"Focus on performance metrics and improvement opportunities. "
                f"Incorporate historical performance data from memory."
            ),
            agent=report,
            expected_output="Standard analysis report with performance insights",
            async_execution=False
        )
        try:
            result = report.execute_task(task)
            # Store normal analysis in memory
            memory_tools[0].run(f"Normal analysis for {self.state.brand_name}: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in normal analysis: {str(e)}")
            return None

    @listen(crisis_router)
    def analysis_aggregator(self, analysis_result):
        """Aggregate results from both crisis and normal analysis."""
        if analysis_result is None:
            return None
        return analysis_result

    @listen(analysis_aggregator)
    def report_task(self, analysis_result):
        """Generate final report based on all collected data."""
        if analysis_result is None:
            return None
            
        report = next(agent for agent in self.agents if agent.role == "Report Generator")
        task = Task(
            description=(
                f"Synthesize a comprehensive real-time report for {self.state.brand_name} using all collected data. "
                f"Include executive summary, data analysis, recommendations, and crisis status. "
                f"Reference historical reports from memory for trend analysis."
            ),
            agent=report,
            expected_output=(
                f"A concise report with executive summary, data analysis, recommendations, "
                f"and crisis status."
            ),
            async_execution=False
        )
        try:
            result = report.execute_task(task)
            self.state.report_data = result
            # Store report in memory
            memory_tools[0].run(f"Final report for {self.state.brand_name}: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in report task: {str(e)}")
            return None

    @listen(report_task)
    def coordinator_task(self, report_result):
        """Coordinate and validate final report."""
        if report_result is None:
            return None
            
        coordinator = next(agent for agent in self.agents if agent.role == "Coordinator")
        task = Task(
            description=(
                f"Aggregate and synthesize real-time outputs from all specialist agents into a final "
                f"comprehensive analysis for {self.state.brand_name}. "
                f"Use memory tools to ensure consistency with historical data."
            ),
            agent=coordinator,
            expected_output=(
                f"A final real-time aggregated report combining all insights and chain-of-thought reasoning "
                f"from specialist agents."
            ),
            async_execution=False
        )
        try:
            result = coordinator.execute_task(task)
            # Store coordinated report in memory
            memory_tools[0].run(f"Coordinated report for {self.state.brand_name}: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in coordinator task: {str(e)}")
            return None

    @listen(coordinator_task)
    def support_task(self, coordinator_result):
        """Provide additional support and clarifications."""
        if coordinator_result is None:
            return None
            
        try:
            support = next(agent for agent in self.agents if agent.role == "Support Agent")
            task = Task(
                description=(
                    f"Provide real-time supplementary support details and clarifications to ensure the final report "
                    f"on {self.state.brand_name} is complete and crisis-ready. "
                    f"Use memory tools to provide historical context and examples."
                ),
                agent=support,
                expected_output="Real-time support insights and clarifications that enrich the overall crisis report.",
                async_execution=False
            )
            result = support.execute_task(task)
            # Store support insights in memory
            memory_tools[0].run(f"Support insights for {self.state.brand_name}: {result}")
            return result
        except StopIteration:
            logger.warning("No Support Agent found; skipping support task.")
            return None
        except Exception as e:
            logger.error(f"Error in support task: {str(e)}")
            return None

    @listen(support_task)
    def memory_task(self, support_result):
        """Store key insights in memory."""
        if support_result is None:
            return None
            
        try:
            memory = next(agent for agent in self.agents if agent.role == "Memory Agent")
            task = Task(
                description=(
                    f"Store and organize key insights from the {self.state.brand_name} analysis "
                    f"for future reference and pattern recognition. "
                    f"Use memory tools to maintain a comprehensive knowledge base."
                ),
                agent=memory,
                expected_output="Memory log with stored insights and patterns",
                async_execution=False
            )
            result = memory.execute_task(task)
            self.state.memory_data = result
            # Store memory data
            memory_tools[0].run(f"Memory data for {self.state.brand_name}: {result}")
            return result
        except StopIteration:
            logger.warning("No Memory Agent found; skipping memory task.")
            return None
        except Exception as e:
            logger.error(f"Error in memory task: {str(e)}")
            return None

    @listen(memory_task)
    def reranking_task(self, memory_result):
        """Optimize and re-rank final report."""
        if memory_result is None:
            return None
            
        try:
            reranker = next(agent for agent in self.agents if agent.role == "Re-ranking Agent")
            task = Task(
                description=(
                    f"Optimize and re-rank the final report for {self.state.brand_name} "
                    f"to ensure the most critical information is prioritized. "
                    f"Use memory tools to maintain consistency with historical reports."
                ),
                agent=reranker,
                expected_output="Optimized and re-ranked final report",
                async_execution=False,
                output_file="final_report.md"
            )
            result = reranker.execute_task(task)
            # Store reranked report in memory
            memory_tools[0].run(f"Reranked report for {self.state.brand_name}: {result}")
            return result
        except StopIteration:
            logger.warning("No Re-ranking Agent found; skipping reranking task.")
            return None
        except Exception as e:
            logger.error(f"Error in reranking task: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    # Create and run the flow
    flow = SocialMediaFlow(brand_name="Example Brand")
    result = flow.kickoff()
    
    # Print the final result
    print("\nFinal Report:")
    print(result)
    
    # Generate flow plot
    flow.plot("social_media_flow") 