from crewai import Task

def create_tasks(brand_name, agents):
    research_task = Task(
        description=f"Research {brand_name} and provide a summary of their online presence, key information, and recent activities.",
        agent=agents[0],
        expected_output="A structured summary containing: \n1. Brief overview of {brand_name}\n2. Key online platforms and follower counts\n3. Recent notable activities or campaigns\n4. Main products or services\n5. Any recent news or controversies"
    )

    monitoring_task = Task(
        description=f"Monitor social media platforms for mentions of '{brand_name}' in the last 24 hours. Provide a summary of the mentions.",
        agent=agents[1],
        expected_output="A structured report containing: \n1. Total number of mentions\n2. Breakdown by platform (e.g., Twitter, Instagram, Facebook)\n3. Top 5 most engaging posts or mentions\n4. Any trending hashtags associated with {brand_name}\n5. Notable influencers or accounts mentioning {brand_name}"
    )

    sentiment_analysis_task = Task(
        description=f"Analyze the sentiment of the social media mentions about {brand_name}. Categorize them as positive, negative, or neutral.",
        agent=agents[2],
        expected_output="A sentiment analysis report containing: \n1. Overall sentiment distribution (% positive, negative, neutral)\n2. Key positive themes or comments\n3. Key negative themes or comments\n4. Any notable changes in sentiment compared to previous periods\n5. Suggestions for sentiment improvement if necessary"
    )

    report_generation_task = Task(
        description=f"Generate a comprehensive report about {brand_name} based on the research, social media mentions, and sentiment analysis. Include key insights and recommendations.",
        agent=agents[3],
        expected_output="A comprehensive report structured as follows: \n1. Executive Summary\n2. Brand Overview\n3. Social Media Presence Analysis\n4. Sentiment Analysis\n5. Key Insights\n6. Recommendations for Improvement\n7. Conclusion"
    )

    return [research_task, monitoring_task, sentiment_analysis_task, report_generation_task]
