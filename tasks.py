from crewai import Task
import time

# Improved task creation logic
def create_tasks(brand_name, agents, twitter_data=None):
    tasks = []
    
    # Task for Social Media Researcher
    researcher = next(agent for agent in agents if agent.role == "Social Media Researcher")
    research_task = Task(
        description=(
            f"Conduct a systematic real-time investigation of {brand_name}'s online presence as of {time.strftime('%Y-%m-%d %H:%M:%S')}. "
            f"Use the 'Fetch Twitter Data' tool to collect recent tweets about {brand_name}, ensuring hyperlinks to original tweets are included. "
            f"Use 'Fetch Web Data with Firecrawl' tool in 'search' mode to fetch posts, news, and events from web sources (e.g., news sites, forums), "
            f"Use search tools and the 'social_media_search' MCP tool to fetch posts, news, and events from platforms like Facebook, TikTok, and news sites, "
            f"Use 'YoutubeVideoSearchTool' to search for videos mentioning {brand_name} and extract key events or discussions from video content. "
            f"capturing hyperlinks to original sources (e.g., article URLs, post links). "
            f"Focus on identifying key events, partnerships, and trends. "
            f"Provide a chain-of-thought (CoT) explanation: (1) search queries (e.g., '{brand_name} legal issues'), "
            f"(2) sources accessed (e.g., Twitter, Thanh Niên), (3) data verification (e.g., cross-check dates), "
            f"(4) synthesis process (e.g., grouping by theme)."
        ),
        agent=researcher,
        expected_output=(
            f"A detailed report with: "
            f"- Key events, partnerships, and trends (e.g., legal issues, product launches). "
            f"- Raw text of social media posts and news articles, each with hyperlinks to original sources. "
            f"- Example: 'Thanh Niên article: [Hằng Du Mục bị phạt 140 triệu](https://thanhnien.vn/link)'. "
            f"- Chain-of-thought explanation."
        ),
        async_execution=False
    )
    
    # Task for Social Media Monitor
    monitor = next(agent for agent in agents if agent.role == "Social Media Monitor")
    monitoring_task = Task(
        description=(
            f"Monitor social media platforms for quantitative engagement metrics about {brand_name} as of {time.strftime('%Y-%m-%d %H:%M:%S')}. "
            f"Use the 'Fetch Twitter Data' tool to collect recent tweets about {brand_name}. "
            f"Use 'Fetch Web Data with Firecrawl' tool in 'scrape' mode if a specific URL is provided, or 'search' mode to fetch additional web data, "
            f"Fetch real-time metrics using TwitterFetchTool and FirecrawlTool (scrape mode for news articles)"
            f"Also use search tools and the 'social_media_search' MCP tool to fetch additional data. "
            f"Use 'YoutubeVideoSearchTool' to identify popular videos about {brand_name} and extract discussion metrics if available. "
            f"Focus on collecting metrics such as mentions, likes, and shares."
        ),
        agent=monitor,
        expected_output=(
            f"A detailed report with: "
            f"- Total mentions, likes, and shares on monitored platforms. "
            f"- Identified trends (e.g., increasing mentions of a specific event). "
            f"- Chain-of-thought explanation of the analysis process."
        ),
        async_execution=False,
        context=[research_task]  # Ensure this task depends on the research task
    )
    
    # Task for Sentiment Analyzer
    sentiment = next(agent for agent in agents if agent.role == "Sentiment Analyzer")
    sentiment_task = Task(
        description=(
        f"Perform a rigorous real-time sentiment analysis on social media mentions of {brand_name} using data from Researcher and Monitor. "
        f"Extract raw text from the outputs of the Researcher and Monitor tasks. "
        f"Run DynamicKeywordExtractorTool on raw text to identify top 5 keywords/themes"
        f"Use the 'distilbert_sentiment_tool' to categorize sentiments into positive, neutral, and negative. "
        f"Compute sentiment distribution percentages and detect crisis signals (e.g., negative > 50%). "
        f"Identify key themes or topics by analyzing frequent keywords or phrases in the mentions. "
        f"Provide a detailed chain-of-thought (CoT) explanation covering: "
        f"(1) data inputs (e.g., raw text from social media mentions), "
        f"(2) sentiment classification method, "
        f"(3) percentage calculation, "
        f"(4) theme identification process, and "
        f"(5) validation steps."
        ),
        agent=sentiment,
        expected_output=(
        f"A real-time sentiment analysis report with: "
        f"- Sentiment distribution percentages (e.g., 60% positive, 30% neutral, 10% negative). "
        f"- Key themes or topics identified (e.g., 'TM Roh', 'HBM3E', 'Galaxy S25 Edge'). "
        f"- Crisis alerts if detected (e.g., 'Crisis Detected: Negative sentiment > 50%'). "
        f"- A chain-of-thought reasoning trace explaining the analysis process."
        ),
        async_execution=False,
        context=[research_task, monitoring_task]  # Ensure this task depends on the research and monitoring tasks
    )

    report = next(agent for agent in agents if agent.role == "Report Generator")
    report_task = Task(
        description=(
        f"Synthesize a comprehensive real-time report for {brand_name} using outputs from Researcher, Monitor, and Sentiment Analyzer. "
        f"Use the 'Dynamic Keyword Extractor Tool' to identify additional themes from combined outputs if needed. "
        f"Use 'Fetch Web Data with Firecrawl' tool in 'crawl' mode to gather broad context from web sources if additional data is needed. "
        f"Include: "
        f"(1) an executive summary with key insights and crisis status, "
        f"(2) a data analysis section with: "
        f"   - Context: Key events, partnerships, and trends from Researcher and Monitor. "
        f"   - Themes: Key topics identified by Sentiment Analyzer. "
        f"   - Metrics: Sentiment percentages from Sentiment Analyzer and engagement metrics (mentions, likes, shares) from Monitor with links. "
        f"(3) actionable recommendations based on findings, "
        f"(4) crisis status with risk assessment, and "
        f"(5) a chain-of-thought explanation. "
        f"Structure the report as follows: "
        f"1. Executive Summary\n"
        f"2. Data Analysis\n   a. Context\n   b. Themes\n   c. Metrics\n"
        f"3. Recommendations\n"
        f"4. Crisis Status\n"
        f"5. Chain-of-Thought\n"
        f"Use real-time data as of {time.strftime('%Y-%m-%d')}."
        ),
        agent=report,
        expected_output=(
        f"A concise report with: "
        f"- Executive summary. "
        f"- Data analysis (e.g., 'Legal issues: [Thanh Niên](https://link)'). "
        f"- Recommendations (e.g., 'Refund via hotline 1900-XXXX'). "
        f"- Crisis status. "
        f"- Chain-of-thought."
        ),
        async_execution=False,
        context=[research_task, monitoring_task, sentiment_task]  # Ensure this task depends on the previous tasks
    )
    
    tasks.extend([research_task, monitoring_task, sentiment_task, report_task])
    
    # Task for Coordinator
    coordinator = next(agent for agent in agents if agent.role == "Coordinator")
    coordinator_task = Task(
        description=(
            f"Aggregate and synthesize real-time outputs from all specialist agents into a final comprehensive analysis for {brand_name}. "
            f"Integrate all CoT reasoning traces to ensure transparency and coherence, with a focus on crisis detection. "
            f"Validate the final report for clarity and accuracy in a crisis context."
        ),
        agent=coordinator,
        expected_output=(
            f"A final real-time aggregated report combining all insights and chain-of-thought reasoning from specialist agents,source links (e.g., '[Tweet](https://twitter.com/link)') "
            f"delivering a coherent and actionable crisis analysis for {brand_name}."
        ),
        async_execution=False,
        context=[research_task, monitoring_task, sentiment_task, report_task]  # Ensure this task depends on all previous tasks
    )
    tasks.append(coordinator_task)
    
    # Task for Support Agent
    try:
        support = next(agent for agent in agents if agent.role == "Support Agent")
        support_task = Task(
            description=(
                f"Provide real-time supplementary support details and clarifications to ensure the final report on {brand_name} is complete "
                f"and crisis-ready. Include additional reasoning or context missed by other agents."
                f"Support insights with links (e.g., '[Article](https://link)') and clarifications."
            ),
            agent=support,
            expected_output="Real-time support insights and clarifications that enrich the overall crisis report.",
            async_execution=False,
            context=[research_task, monitoring_task, sentiment_task, report_task, coordinator_task],
            dependencies=[research_task, monitoring_task, sentiment_task, report_task, coordinator_task]
        )
        tasks.append(support_task)
    except StopIteration:
        print("No Support Agent found; skipping support task.")
    
    # Task for Memory Agent
    try:
        memory = next(agent for agent in agents if agent.role == "Memory Agent")
        memory_task = Task(
            description=(
                f"Store real-time key insights and detailed reasoning traces from the analysis on {brand_name} for future reference "
                f"and crisis trend tracking."
            ),
            agent=memory,
            async_execution=False,
            expected_output="A real-time memory log containing all key insights and chain-of-thought reasoning traces."
        )
        tasks.append(memory_task)
    except StopIteration:
        print("No Memory Agent found; skipping memory task.")
    
    # Task for Re-ranking Agent
    try:
        reranker = next(agent for agent in agents if agent.role == "Re-ranking Agent")
        reranking_task = Task(
            description=(
                f"Re-rank and evaluate real-time candidate outputs from all agents to produce the most coherent and crisis-focused "
                f"final report for {brand_name}. Provide a refined report integrating the best reasoning."
                f"ensuring all hyperlinks are included and functional."
            ),
            agent=reranker,
            expected_output=(
                f"A refined real-time report that optimally integrates and reorders outputs and reasoning from all agents, "
                f"with emphasis on crisis detection."
                f"A refined report with integrated insights and links (e.g., '[News](https://link)'), "
            ),
            context=[research_task, monitoring_task, sentiment_task, report_task, coordinator_task, support_task],
            dependencies=[support_task],
            async_execution=False,
            output_file="final_report.md"
        )
        tasks.append(reranking_task)
    except StopIteration:
        print("No Re-ranking Agent found; skipping reranking task.")
    
    # Task for Crisis Detector Agent
    try:
        crisis_detector = next(agent for agent in agents if agent.role == "Crisis Detector")
        crisis_task = Task(
            description=(
                f"Monitor real-time sentiment trends for {brand_name} using outputs from Sentiment Analyzer and detect potential crises "
                f"(e.g., negative sentiment > 50% or rapid negative spike). Provide a CoT explanation: (1) data sources, "
                f"(2) crisis threshold, (3) detection process."
                f"Use 'Fetch Web Data with Firecrawl' tool in 'search' mode to fetch additional web data for trend validation. "
            ),
            agent=crisis_detector,
            expected_output=(
                "Crisis status: 'Detected' or 'Not Detected' with percentage evidence (e.g., 'Detected: 60% negative sentiment') "
                "and a chain-of-thought reasoning trace."
            ),
            context=[sentiment_task, monitoring_task],
            async_execution=False,
            dependencies=[sentiment_task]
        )
        tasks.append(crisis_task)
    except StopIteration:
        print("No Crisis Detector Agent found; skipping crisis task.")
    
    return tasks