from crewai import Agent
from crewai_tools import SerperDevTool
import os
from crewai.llm import LLM
from dotenv import load_dotenv

load_dotenv()

# Thiết lập biến môi trường
os.environ["CREWAI_DISABLE_TELEMETRY"] = "1"
#os.environ["LITELLM_LOG"] = "DEBUG"  # Bật log debug cho LiteLLM

## Define a custom subclass to fix the search_query input type
class MySerperDevTool(SerperDevTool):
    def run(self, **kwargs):
        if 'search_query' in kwargs and isinstance(kwargs['search_query'], dict):
            kwargs['search_query'] = kwargs['search_query'].get('description', str(kwargs['search_query']))
        return super().run(**kwargs)

# Sử dụng custom search tool
search_tool = MySerperDevTool()

def create_llm():
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    # Sử dụng Gemini model với cú pháp "provider/llm-name"
    return LLM(api_key=GEMINI_API_KEY, model="gemini/gemini-2.0-flash-exp")

# Các Agent chuyên môn với các quy tắc CrewAI
def create_specialist_agents(brand_name, llm):
    researcher = Agent(
        role="Social Media Researcher",
        goal=f"Gather comprehensive and accurate information about {brand_name} from diverse sources.",
        backstory=(
            "You work as a Social Media Researcher and are dedicated to collecting detailed data about the brand. "
            "Focus: Provide a complete overview with all necessary details and statistics without making assumptions. "
            "Guardrails: Verify all data before including it in your summary; ensure clarity and precision in your findings. "
            "Role Playing: Embody the role of an expert researcher with thorough analytical skills."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
        max_iter=2
    )

    social_media_monitor = Agent(
        role="Social Media Monitor",
        goal=f"Monitor and extract detailed engagement metrics and trends about {brand_name} across multiple platforms.",
        backstory=(
            "You serve as the Social Media Monitor. Focus on tracking engagement data, identifying trending hashtags, and noting key influencer mentions. "
            "Guardrails: Do not infer or speculate beyond the available data. Report only verified metrics. "
            "Role Playing: Act as a seasoned analyst with sharp attention to social dynamics."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
        max_iter=2
    )

    sentiment_analyzer = Agent(
        role="Sentiment Analyzer",
        goal=f"Conduct an in-depth sentiment analysis on all social media mentions regarding {brand_name}.",
        backstory=(
            "You are the Sentiment Analyzer tasked with categorizing social media sentiments into positive, negative, or neutral. "
            "Focus: Provide detailed sentiment distributions with examples. "
            "Guardrails: Ensure that your analysis is based strictly on the language cues without personal bias. "
            "Role Playing: Portray an expert in natural language processing with a keen sense of emotional nuance."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=2
    )

    report_generator = Agent(
        role="Report Generator",
        goal=f"Generate a detailed, structured report on {brand_name} that includes an executive summary, data analysis, and actionable recommendations.",
        backstory=(
            "As the Report Generator, your task is to synthesize all gathered information into a coherent, comprehensive report. "
            "Focus: Ensure the final report is complete, logically structured, and covers all aspects of the analysis. "
            "Guardrails: Avoid omitting any crucial details; all conclusions must be backed by data. "
            "Role Playing: Embody a seasoned data analyst and report writer who communicates insights clearly and effectively."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=2
    )

    return [researcher, social_media_monitor, sentiment_analyzer, report_generator]

# Agent Điều phối (Coordinator) với quy tắc về tổng hợp và hợp nhất thông tin
def create_coordinator_agent(brand_name, llm):
    coordinator = Agent(
        role="Coordinator",
        goal=f"Aggregate and synthesize all outputs from specialist agents to produce a final, comprehensive analysis for {brand_name}.",
        backstory=(
            "You are the Coordinator, responsible for merging the insights from all specialist agents. "
            "Focus: Ensure that the final analysis is cohesive, comprehensive, and actionable. "
            "Guardrails: Validate that all key information from subordinate agents is accurately represented without contradictions. "
            "Role Playing: Act as an experienced manager with a strategic vision, ensuring clarity and thoroughness."
        ),
        verbose=True,
        allow_delegation=True,
        tools=[],  # Coordinator typically does not use external tools.
        llm=llm,
        max_iter=3
    )
    return coordinator

# Tạo các Agent bổ sung theo quy tắc CrewAI (ví dụ: Support, Memory, Re-ranking)
def create_support_agent(brand_name, llm):
    support_agent = Agent(
        role="Support Agent",
        goal="Provide supplementary support to ensure the overall system delivers complete and accurate analyses.",
        backstory=(
            "You are the Support Agent. Your role is to provide additional context, clarification, and help where necessary. "
            "Focus: Offer comprehensive, error-free support. "
            "Guardrails: Do not deviate from the established guidelines; always ensure accuracy."
        ),
        verbose=True,
        allow_delegation=False,
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
            "You are responsible for maintaining a long-term memory of all interactions. "
            "Focus: Ensure that no critical information is lost, and that context is preserved for future decisions. "
            "Guardrails: Data must be stored in an organized and retrievable format."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[],
        llm=llm,
        max_iter=1
    )
    return memory_agent

def create_reranking_agent(brand_name, llm):
    reranker = Agent(
        role="Re-ranking Agent",
        goal="Evaluate and re-rank candidate outputs from other agents to produce the optimal final analysis.",
        backstory=(
            "You are the Re-ranking Agent, tasked with assessing the quality, coherence, and completeness of outputs from other agents. "
            "Focus: Reorder or merge outputs to achieve the best final report possible. "
            "Guardrails: Your final output must be logical, well-supported, and free of inconsistencies."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[],
        llm=llm,
        max_iter=2
    )
    return reranker

# Tạo danh sách các Agent theo kiến trúc đa tầng
def create_agents(brand_name, llm):
    specialists = create_specialist_agents(brand_name, llm)
    coordinator = create_coordinator_agent(brand_name, llm)
    support = create_support_agent(brand_name, llm)
    memory = create_memory_agent(brand_name, llm)
    reranker = create_reranking_agent(brand_name, llm)
    # Sắp xếp thứ tự: Specialist Agents -> Coordinator Agent -> Support Agent -> Memory Agent -> Re-ranking Agent
    return specialists + [coordinator, support, memory, reranker]
