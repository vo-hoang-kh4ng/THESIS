import time
from crewai import Crew
from my_agents import create_llm, create_agents
from tasks import create_tasks

def run_social_media_monitoring(brand_name, max_retries=3):
    llm = create_llm()
    agents = create_agents(brand_name, llm)
    tasks = create_tasks(brand_name, agents)
    
    crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=True
    )

    for attempt in range(max_retries):
        try:
            result = crew.kickoff()
            return result
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(5)  # Chờ 5 giây trước khi retry
            else:
                print("Max retries reached. Unable to complete the task.")
                return None