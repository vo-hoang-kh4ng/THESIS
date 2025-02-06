from crewai import Agent
from crewai_tools import SerperDevTool
from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
import os
from crewai.llm import LLM

os.environ["CREWAI_DISABLE_TELEMETRY"] = "1"

## Define a custom subclass to fix the search_query input type
class MySerperDevTool(SerperDevTool):
    def run(self, **kwargs):
        if 'search_query' in kwargs and isinstance(kwargs['search_query'], dict):
            kwargs['search_query'] = kwargs['search_query'].get('description', str(kwargs['search_query']))
        return super().run(**kwargs)

# Use the custom search tool
search_tool = MySerperDevTool()

def create_llm():
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    # Using Gemini model with the required format "provider/llm-name"
    return LLM(api_key=GEMINI_API_KEY, model="gemini/gemini-1.5-flash")

def create_specialist_agents(brand_name, llm):
    researcher = Agent(
        role="Social Media Researcher",
        goal=f"Research and gather information about {brand_name} from various sources",
        backstory="You are an expert researcher with a knack for finding relevant information quickly.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
        max_iter=2
    )

    social_media_monitor = Agent(
        role="Social Media Monitor",
        goal=f"Monitor social media platforms for mentions of {brand_name}",
        backstory="You are an experienced social media analyst with keen eyes for trends and mentions.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
        max_iter=2
    )

    sentiment_analyzer = Agent(
        role="Sentiment Analyzer",
        goal=f"Analyze the sentiment of social media mentions about {brand_name}",
        backstory="You are an expert in natural language processing and sentiment analysis.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=2
    )

    report_generator = Agent(
        role="Report Generator",
        goal=f"Generate comprehensive reports based on the analysis of {brand_name}",
        backstory="You are a skilled data analyst and report writer, adept at presenting insights clearly.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=2
    )

    return [researcher, social_media_monitor, sentiment_analyzer, report_generator]

# Tạo Coordinator Agent
def create_coordinator_agent(brand_name, llm):
    coordinator = Agent(
        role="Coordinator",
        goal=f"Aggregate and synthesize the outputs from all specialist agents to provide a final comprehensive analysis on {brand_name}",
        backstory="You are a high-level coordinator capable of merging and refining insights from various expert agents to produce an actionable final report.",
        verbose=True,
        allow_delegation=True,
        tools=[],  # Coordinator typically doesn't require external tools.
        llm=llm,
        max_iter=3
    )
    return coordinator

# Sửa hàm create_agents để trả về danh sách các Agent gồm các Agent chuyên môn và Coordinator Agent.
def create_agents(brand_name, llm):
    specialists = create_specialist_agents(brand_name, llm)
    coordinator = create_coordinator_agent(brand_name, llm)
    # Kết hợp danh sách các Agent chuyên môn với Coordinator (đặt Coordinator ở cuối danh sách)
    return specialists + [coordinator]
