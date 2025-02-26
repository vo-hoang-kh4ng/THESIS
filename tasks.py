from crewai import Task
import time

def create_tasks(brand_name, agents):
    tasks = []
    
    # Nhiệm vụ cho Specialist Agents (4 Agent đầu tiên)
    researcher = next(agent for agent in agents if agent.role == "Social Media Researcher")
    research_task = Task(
        description=(
            f"Conduct a systematic real-time investigation of {brand_name}'s online presence, focusing on social media platforms, "
            f"news articles, and public websites as of {time.strftime('%Y-%m-%d %H:%M:%S')}. Use search tools to collect raw data "
            f"(e.g., posts, articles, mentions). Provide a detailed chain-of-thought (CoT) explanation documenting: (1) search queries used, "
            f"(2) sources accessed, (3) data verification steps, and (4) synthesis process. Aim for a reproducible methodology suitable "
            f"for scientific reporting and crisis monitoring."
        ),
        agent=researcher,
        expected_output=(
            f"A real-time structured summary containing key data points and insights about {brand_name}, along with a detailed "
            "chain-of-thought explanation of your reasoning process."
        ),
        async_execution=True
    )
    
    monitor = next(agent for agent in agents if agent.role == "Social Media Monitor")
    monitoring_task = Task(
        description=(
            f"Monitor social media platforms (e.g., X, Facebook) in real-time for quantitative engagement metrics about {brand_name} "
            f"as of {time.strftime('%Y-%m-%d %H:%M:%S')}. Use search tools to extract data such as mention counts, likes, shares, "
            f"and trends. Provide a step-by-step CoT explanation detailing: (1) platforms monitored, (2) metrics collected, "
            f"(3) data validation methods, and (4) trend identification process, with a focus on crisis signals."
        ),
        agent=monitor,
        expected_output=(
            f"A real-time detailed report with metrics and engagement data on {brand_name}, including a chain-of-thought explanation "
            "of your analysis process."
        ),
        async_execution=True
    )
    
    sentiment = next(agent for agent in agents if agent.role == "Sentiment Analyzer")
    sentiment_task = Task(
        description=(
            f"Perform a rigorous real-time sentiment analysis on social media mentions of {brand_name} using data from Researcher and Monitor. "
            f"Categorize sentiments into positive, neutral, and negative, compute distribution percentages, and detect crisis signals "
            f"(e.g., negative > 50%). Provide a detailed CoT explanation covering: (1) data inputs, (2) sentiment classification method, "
            f"(3) percentage calculation, and (4) validation steps."
        ),
        agent=sentiment,
        expected_output=(
            f"A real-time sentiment analysis report with percentages (e.g., 60% positive, 30% neutral, 10% negative), key themes, "
            "crisis alerts if detected, and a comprehensive chain-of-thought reasoning trace."
        )
    )
    
    report = next(agent for agent in agents if agent.role == "Report Generator")
    report_task = Task(
        description=(
            f"Synthesize a comprehensive real-time report for {brand_name} using outputs from Researcher, Monitor, and Sentiment Analyzer. "
            f"Include: (1) an executive summary, (2) data analysis section with metrics and sentiment, (3) actionable recommendations "
            f"based on findings, and (4) crisis alerts if detected. Provide a detailed CoT explanation of the synthesis process. "
            f"Structure the report for rapid crisis response."
        ),
        agent=report,
        expected_output=(
            f"A real-time comprehensive report including executive summary, data analysis, recommendations, and crisis status, "
            "with a detailed chain-of-thought explanation of your reasoning."
        )
    )
    
    tasks.extend([research_task, monitoring_task, sentiment_task, report_task])
    
    # Nhiệm vụ cho Coordinator Agent
    coordinator = next(agent for agent in agents if agent.role == "Coordinator")
    coordinator_task = Task(
        description=(
            f"Aggregate and synthesize real-time outputs from all specialist agents into a final comprehensive analysis for {brand_name}. "
            f"Integrate all CoT reasoning traces to ensure transparency and coherence, with a focus on crisis detection. "
            f"Validate the final report for clarity and accuracy in a crisis context."
        ),
        agent=coordinator,
        expected_output=(
            f"A final real-time aggregated report combining all insights and chain-of-thought reasoning from specialist agents, "
            f"delivering a coherent and actionable crisis analysis for {brand_name}."
        )
    )
    tasks.append(coordinator_task)
    
    # Nhiệm vụ cho Support Agent
    try:
        support = next(agent for agent in agents if agent.role == "Support Agent")
        support_task = Task(
            description=(
                f"Provide real-time supplementary support details and clarifications to ensure the final report on {brand_name} is complete "
                f"and crisis-ready. Include additional reasoning or context missed by other agents."
            ),
            agent=support,
            expected_output="Real-time support insights and clarifications that enrich the overall crisis report.",
            context=[research_task, monitoring_task, sentiment_task, report_task, coordinator_task],
            dependencies=[research_task, monitoring_task, sentiment_task, report_task, coordinator_task]
        )
        tasks.append(support_task)
    except StopIteration:
        print("No Support Agent found; skipping support task.")
    
    # Nhiệm vụ cho Memory Agent
    try:
        memory = next(agent for agent in agents if agent.role == "Memory Agent")
        memory_task = Task(
            description=(
                f"Store real-time key insights and detailed reasoning traces from the analysis on {brand_name} for future reference "
                f"and crisis trend tracking."
            ),
            agent=memory,
            expected_output="A real-time memory log containing all key insights and chain-of-thought reasoning traces."
        )
        tasks.append(memory_task)
    except StopIteration:
        print("No Memory Agent found; skipping memory task.")
    
    # Nhiệm vụ cho Re-ranking Agent
    try:
        reranker = next(agent for agent in agents if agent.role == "Re-ranking Agent")
        reranking_task = Task(
            description=(
                f"Re-rank and evaluate real-time candidate outputs from all agents to produce the most coherent and crisis-focused "
                f"final report for {brand_name}. Provide a refined report integrating the best reasoning."
            ),
            agent=reranker,
            expected_output=(
                "A refined real-time report that optimally integrates and reorders outputs and reasoning from all agents, "
                "with emphasis on crisis detection."
            ),
            context=[research_task, monitoring_task, sentiment_task, report_task, coordinator_task, support_task],
            dependencies=[support_task],
            output_file="final_report.md"
        )
        tasks.append(reranking_task)
    except StopIteration:
        print("No Re-ranking Agent found; skipping reranking task.")
    
    # Nhiệm vụ cho Crisis Detector Agent (mới)
    try:
        crisis_detector = next(agent for agent in agents if agent.role == "Crisis Detector")
        crisis_task = Task(
            description=(
                f"Monitor real-time sentiment trends for {brand_name} using outputs from Sentiment Analyzer and detect potential crises "
                f"(e.g., negative sentiment > 50% or rapid negative spike). Provide a CoT explanation: (1) data sources, "
                f"(2) crisis threshold, (3) detection process."
            ),
            agent=crisis_detector,
            expected_output=(
                "Crisis status: 'Detected' or 'Not Detected' with percentage evidence (e.g., 'Detected: 60% negative sentiment') "
                "and a chain-of-thought reasoning trace."
            ),
            context=[sentiment_task, monitoring_task],
            dependencies=[sentiment_task]
        )
        tasks.append(crisis_task)
    except StopIteration:
        print("No Crisis Detector Agent found; skipping crisis task.")
    
    return tasks
