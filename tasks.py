from crewai import Task

def create_tasks(brand_name, agents):
    tasks = []
    
    # Nhiệm vụ cho Specialist Agents (giả sử các Specialist là 4 Agent đầu tiên)
    researcher = next(agent for agent in agents if agent.role == "Social Media Researcher")
    research_task = Task(
        description=f"Research {brand_name} and provide a detailed summary of their online presence, key information, and recent activities.",
        agent=researcher,
        expected_output="A structured summary containing key data points and insights about {brand_name}."
    )
    
    monitor = next(agent for agent in agents if agent.role == "Social Media Monitor")
    monitoring_task = Task(
        description=f"Monitor social media platforms for detailed metrics and engagement data about {brand_name}.",
        agent=monitor,
        expected_output="A detailed report with metrics and engagement data on {brand_name}."
    )
    
    sentiment = next(agent for agent in agents if agent.role == "Sentiment Analyzer")
    sentiment_task = Task(
        description=f"Perform an in-depth sentiment analysis on the social media mentions of {brand_name}.",
        agent=sentiment,
        expected_output="A detailed sentiment analysis report with percentages and key themes."
    )
    
    report = next(agent for agent in agents if agent.role == "Report Generator")
    report_task = Task(
        description=f"Generate a comprehensive report about {brand_name} based on the gathered research and analysis.",
        agent=report,
        expected_output="A comprehensive report including executive summary, data analysis, and recommendations."
    )
    
    tasks.extend([research_task, monitoring_task, sentiment_task, report_task])
    
    # Nhiệm vụ cho Coordinator Agent
    coordinator = next(agent for agent in agents if agent.role == "Coordinator")
    coordinator_task = Task(
        description=f"Aggregate and synthesize the outputs from all specialist agents to produce a final comprehensive analysis on {brand_name}.",
        agent=coordinator,
        expected_output="A final aggregated report that combines all insights and provides a coherent overall analysis."
    )
    tasks.append(coordinator_task)
    
    # Nhiệm vụ cho Support Agent
    try:
        support = next(agent for agent in agents if agent.role == "Support Agent")
        support_task = Task(
            description=f"Provide supplementary support details and clarifications to ensure that the final report on {brand_name} is complete and addresses all relevant queries.",
            agent=support,
            expected_output="Additional support insights and clarifications that enrich the overall report."
        )
        tasks.append(support_task)
    except StopIteration:
        print("No Support Agent found; skipping support task.")
    
    # Nhiệm vụ cho Memory Agent
    try:
        memory = next(agent for agent in agents if agent.role == "Memory Agent")
        memory_task = Task(
            description=f"Store key insights and reasoning traces from the analysis on {brand_name} for future reference.",
            agent=memory,
            expected_output="A memory log containing all key insights and reasoning traces."
        )
        tasks.append(memory_task)
    except StopIteration:
        print("No Memory Agent found; skipping memory task.")
    
    # Nhiệm vụ cho Re-ranking Agent
    try:
        reranker = next(agent for agent in agents if agent.role == "Re-ranking Agent")
        reranking_task = Task(
            description=f"Re-rank and evaluate candidate outputs from all agents to produce the most coherent and detailed final report for {brand_name}.",
            agent=reranker,
            expected_output="A refined final report that integrates and optimizes the outputs from all agents."
        )
        tasks.append(reranking_task)
    except StopIteration:
        print("No Re-ranking Agent found; skipping reranking task.")
    
    return tasks
