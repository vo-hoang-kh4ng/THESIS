import time
from crewai import Crew,Process
from my_agents import create_llm, create_agents
from tasks import create_tasks
def run_social_media_monitoring(brand_name, max_retries=3):
    llm = create_llm()
    agents = create_agents(brand_name, llm)  # Danh sách Agent gồm Specialist, Coordinator, Support, Memory, Re-ranking
    tasks = create_tasks(brand_name, agents)
    
    crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=True,
        memory=False # Kích hoạt tính năng Memory cho toàn bộ crew
        # planning=True,
        # planning_llm= llm
        #process=Process.sequential
    )
    
    for attempt in range(max_retries):
        try:
            result = crew.kickoff()
            return result
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(5)
            else:
                print("Max retries reached. Unable to complete the task.")
                return None
