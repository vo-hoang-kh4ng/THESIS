from crewai import Agent
import os
from crewai.llm import LLM
from dotenv import load_dotenv
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from langchain_community.utilities import GoogleSerperAPIWrapper
import litellm
from transformers import pipeline
from tools.sentiment_tool import SentimentAnalysisTool
from tools.exa_answer_tool import EXAAnswerTool
from tools.keyword_tool import DynamicKeywordExtractorTool
from tools.serper_dev_tool import MySerperDevTool
from tools.my_twitter_tool import TwitterFetchTool
from tools.firecrawl_tool import FirecrawlTool
from tools.youtube_search_tool import MyYoutubeSearchTool
from langgraph.store.memory import InMemoryStore
from crewai.tools import BaseTool
from langmem import create_manage_memory_tool, create_search_memory_tool
from pydantic import Field
load_dotenv()
os.environ["CREWAI_DISABLE_TELEMETRY"] = "1"

firecrawl_tool = FirecrawlTool()
search_tool = MySerperDevTool()
sentiment_tool = SentimentAnalysisTool()
key_word_tool = DynamicKeywordExtractorTool()
exa_tool = EXAAnswerTool()
twitter_fetch_tool = TwitterFetchTool()
youtube_search_tool = MyYoutubeSearchTool()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
store = InMemoryStore()
# class LangMemToolWrapper(BaseTool):
#     name: str
#     description: str
#     structured_tool: object = Field(..., description="The underlying StructuredTool from langmem")  # Khai báo trường

#     def __init__(self, structured_tool):
#         super().__init__(
#             name=structured_tool.name,
#             description=structured_tool.description or "A memory management tool",
#         )
#         self.structured_tool = structured_tool

#     def _run(self, *args, **kwargs):
#         return self.structured_tool.run(*args, **kwargs)

# # Memory tools from LangMem, bọc trong wrapper
# manage_memory_tool = LangMemToolWrapper(create_manage_memory_tool(namespace="brand_presence", store=store))
# search_memory_tool = LangMemToolWrapper(create_search_memory_tool(namespace="brand_presence", store=store))

# memory_tools = [manage_memory_tool, search_memory_tool]

# memory_tools = [manage_memory_tool, search_memory_tool]


def create_llm():
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    return LLM(api_key=GEMINI_API_KEY , model="gemini/gemini-2.0-flash-exp")

# Các Agent chuyên môn với các quy tắc CrewAI
def create_specialist_agents(brand_name, llm):
    researcher = Agent(
        role="Social Media Researcher",
        goal=f"Gather comprehensive and accurate information about {brand_name} from diverse sources.",
        backstory=(
            "You work as a Social Media Researcher dedicated to collecting up-to-date data about the brand. "
            "Focus: Provide a real-time overview with statistics to support crisis monitoring. "
            "Guardrails: Verify all data before inclusion; prioritize speed and precision. "
            "Role Playing: Embody an expert researcher with rapid response capabilities."
        ),
        verbose=True,
        allow_delegation=True,
        tools=[search_tool,exa_tool,twitter_fetch_tool,youtube_search_tool],
        llm=llm,
        max_iter=2
    )

    social_media_monitor = Agent(
        role="Social Media Monitor",
        goal=f"Monitor and extract detailed engagement metrics and trends about {brand_name} across multiple platforms.",
        backstory=(
            "You serve as the Social Media Monitor, tracking engagement data and trends in real-time. "
            "Focus: Extract metrics, hashtags, and influencer mentions with an eye on crisis detection. "
            "Guardrails: Report only verified, current data; avoid speculation. "
            "Role Playing: Act as a seasoned analyst with real-time social dynamics expertise."
        ),
        verbose=True,
        allow_delegation=True,
        tools=[search_tool,twitter_fetch_tool,youtube_search_tool],
        llm=llm,
        max_iter=2
    )

    sentiment_analyzer = Agent(
        role="Sentiment Analyzer",
        goal=f"Generate real-time, structured reports on {brand_name} with crisis alerts, including summary, analysis, and recommendations.",
        backstory=(
            "You are the Sentiment Analyzer, categorizing sentiments into positive, negative, or neutral in real-time. "
            "Focus: Provide rapid sentiment distributions with examples, flagging high negative sentiment as a crisis signal. "
            "Guardrails: Base analysis strictly on language cues; ensure speed and accuracy. "
            "Role Playing: Portray an NLP expert with a focus on crisis-sensitive analysis."
        ),
        verbose=True,
        allow_delegation=True,
        llm=llm,
        tools = [sentiment_tool,key_word_tool],
        max_iter=1

    )

    report_generator = Agent(
        role="Report Generator",
        goal=f"Generate a detailed, structured report on {brand_name} that includes an executive summary, data analysis, and actionable recommendations.",
        backstory=(
            "As the Report Generator, you synthesize real-time data into concise, actionable reports. "
            "Focus: Deliver fast, clear reports with crisis warnings if detected. "
            "Guardrails: Include all critical details; back conclusions with data. "
            "Role Playing: Embody a data analyst adept at real-time reporting."
        ),
        verbose=True,
        allow_delegation=True,
        llm=llm,
        max_iter=1
    )

    return [researcher, social_media_monitor, sentiment_analyzer, report_generator]

# Agent Điều phối (Coordinator) với quy tắc về tổng hợp và hợp nhất thông tin
def create_coordinator_agent(brand_name, llm):
    coordinator = Agent(
        role="Coordinator",
        goal=f"Aggregate and synthesize all outputs from specialist agents to produce a final, comprehensive analysis for {brand_name}.",
        backstory=(
           "You are the Coordinator, merging real-time insights from specialists. "
            "Focus: Ensure the analysis is cohesive, timely, and actionable for crisis management. "
            "Guardrails: Validate all inputs for accuracy and consistency. "
            "Role Playing: Act as a strategic manager with real-time oversight."
        ),
        verbose=True,
        allow_delegation=True,
        tools=[],  # Coordinator typically does not use external tools.
        llm=llm,
        max_iter=1
    )
    return coordinator
def create_support_agent(brand_name, llm):
    support_agent = Agent(
        role="Support Agent",
        goal="Provide supplementary support to ensure the overall system delivers complete and accurate analyses.",
        backstory=(
            "You are the Support Agent. Your role is to provide additional context, clarification, and help where necessary in real-time.. "
            "Focus: Offer comprehensive, error-free support. "
            "Guardrails: Do not deviate from the established guidelines; always ensure accuracy."
        ),
        verbose=True,
        allow_delegation=True,
        tools=[],
        llm=llm,
        max_iter=1
    )
    return support_agent

def create_memory_agent(brand_name, llm):
    
    memory_agent = Agent(
        role="Memory Agent",
        goal="Store and retrieve important data and reasoning traces from previous interactions.",
        backstory=(
            f"You are responsible for maintaining a long-term memory of all interactions and data related to {brand_name}. "
            "Focus: Collect data from multiple sources and ensure it is stored in an organized, retrievable format for future analysis. "
            "Guardrails: Verify data accuracy before storage; maintain consistency."
        ),
        verbose=True,
        allow_delegation=True,
        tools=[],
        llm=llm,
        max_iter=1
    )
    return memory_agent

def create_reranking_agent(brand_name, llm):
    reranker = Agent(
        role="Re-ranking Agent",
        goal="Evaluate and re-rank candidate outputs from other agents to produce the optimal final analysis,.",
        backstory=(
            "You are the Re-ranking Agent, tasked with assessing the quality, coherence, and completeness of real-time outputs from other agents. "
            "Focus: Reorder or merge outputs to achieve the best final report possible. "
            "Guardrails: Your final output must be logical, well-supported, and free of inconsistencies."
        ),
        verbose=True,
        allow_delegation=True,
        tools=[],
        llm=llm,
        max_iter=1
        
    )
    return reranker
# Agent mới: Crisis Detector
def create_crisis_detector_agent(brand_name, llm):
    crisis_detector = Agent(
        role="Crisis Detector",
        goal=f"Monitor real-time sentiment trends for {brand_name} and detect potential crises (e.g., high negative sentiment).",
        backstory=(
            "You are the Crisis Detector, specialized in identifying real-time threats to brand reputation. "
            "Focus: Flag crises based on rapid sentiment shifts (e.g., negative > 50%). "
            "Guardrails: Use data-driven thresholds; avoid false alarms. "
            "Role Playing: Act as a vigilant sentinel for brand safety."
        ),
        verbose=True,
        allow_delegation=True,
        tools=[search_tool,youtube_search_tool],
        llm=llm,
        max_iter=1 
    )
    return crisis_detector
# Tạo danh sách các Agent theo kiến trúc đa tầng
def create_agents(brand_name, llm):
    specialists = create_specialist_agents(brand_name, llm)
    coordinator = create_coordinator_agent(brand_name, llm)
    support = create_support_agent(brand_name, llm)
    memory = create_memory_agent(brand_name, llm)
    reranker = create_reranking_agent(brand_name, llm)
    crisis_detector = create_crisis_detector_agent(brand_name, llm)
    return specialists + [coordinator, support, memory, reranker, crisis_detector]