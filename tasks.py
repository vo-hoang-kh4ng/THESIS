from crewai import Task

def create_tasks(brand_name, agents):
    tasks = []
    
    # Nhiệm vụ cho Specialist Agents (giả sử các Specialist là 4 Agent đầu tiên)
    researcher = next(agent for agent in agents if agent.role == "Social Media Researcher")
    research_task = Task(
        description=(
            f"Research {brand_name} and provide a detailed summary of their online presence, key information, and recent activities. "
            "Include a step-by-step explanation (chain-of-thought) outlining how you retrieved, verified, and synthesized the information."
        ),
        agent=researcher,
        expected_output=(
            "A structured summary containing key data points and insights about {brand_name}, along with a detailed chain-of-thought "
            "explanation of your reasoning process."
        )
    )
    
    monitor = next(agent for agent in agents if agent.role == "Social Media Monitor")
    monitoring_task = Task(
        description=(
            f"Monitor social media platforms for detailed metrics and engagement data about {brand_name}. "
            "Provide a step-by-step reasoning on how you identified key metrics and trends."
        ),
        agent=monitor,
        expected_output=(
            "A detailed report with metrics and engagement data on {brand_name}, including a chain-of-thought explanation of your analysis process."
        )
    )
    
    sentiment = next(agent for agent in agents if agent.role == "Sentiment Analyzer")
    sentiment_task = Task(
        description=(
            f"Perform an in-depth sentiment analysis on the social media mentions of {brand_name}. "
            "Explain your reasoning step-by-step, detailing how you categorized sentiments and computed the distributions."
        ),
        agent=sentiment,
        expected_output=(
            "A detailed sentiment analysis report with percentages and key themes, along with a comprehensive chain-of-thought reasoning trace."
        )
    )
    
    report = next(agent for agent in agents if agent.role == "Report Generator")
    report_task = Task(
        description=(
            f"Generate a comprehensive report about {brand_name} based on the gathered research and analysis. "
            "Include an executive summary, data analysis, and actionable recommendations, along with a detailed chain-of-thought explanation for your synthesis process."
        ),
        agent=report,
        expected_output=(
            "A comprehensive report including executive summary, data analysis, and recommendations, with a detailed explanation (chain-of-thought) of your reasoning."
        )
    )
    
    tasks.extend([research_task, monitoring_task, sentiment_task, report_task])
    
    # Nhiệm vụ cho Coordinator Agent
    coordinator = next(agent for agent in agents if agent.role == "Coordinator")
    coordinator_task = Task(
        description=(
            f"Aggregate and synthesize the outputs from all specialist agents to produce a final comprehensive analysis on {brand_name}. "
            "Ensure you integrate all chain-of-thought reasoning traces from the specialist agents into your final report."
        ),
        agent=coordinator,
        expected_output=(
            "A final aggregated report that combines all insights and detailed chain-of-thought reasoning from the specialist agents, delivering a coherent and actionable overall analysis on {brand_name}."
        )
    )
    tasks.append(coordinator_task)
    
    # Nhiệm vụ cho Support Agent
    try:
        support = next(agent for agent in agents if agent.role == "Support Agent")
        support_task = Task(
            description=(
                f"Provide supplementary support details and clarifications to ensure that the final report on {brand_name} is complete. "
                "Include any additional reasoning or context that might have been missed by the other agents."
            ),
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
            description=(
                f"Store key insights and detailed reasoning traces from the analysis on {brand_name} for future reference."
            ),
            agent=memory,
            expected_output="A memory log containing all key insights and chain-of-thought reasoning traces."
        )
        tasks.append(memory_task)
    except StopIteration:
        print("No Memory Agent found; skipping memory task.")
    
    # Nhiệm vụ cho Re-ranking Agent
    try:
        reranker = next(agent for agent in agents if agent.role == "Re-ranking Agent")
        reranking_task = Task(
            description=(
                f"Re-rank and evaluate candidate outputs from all agents to produce the most coherent and detailed final report for {brand_name}. "
                "Provide a refined final report that integrates the best chain-of-thought reasoning from all agents."
            ),
            agent=reranker,
            expected_output=(
                "A refined final report that optimally integrates and reorders the outputs and chain-of-thought reasoning from all agents."
            )
        )
        tasks.append(reranking_task)
    except StopIteration:
        print("No Re-ranking Agent found; skipping reranking task.")
    
    return tasks