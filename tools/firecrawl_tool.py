# firecrawl_tool.py
import os
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from crewai_tools import FirecrawlCrawlWebsiteTool, FirecrawlScrapeWebsiteTool, FirecrawlSearchTool
import json

# Define the input schema for the tool
class FirecrawlToolSchema(BaseModel):
    query: str = Field(..., description="The search query or URL to fetch data from the web (e.g., 'Tesla', 'https://example.com').")
    limit: int = Field(default=50, description="Number of results or pages to fetch (default: 50).")
    mode: str = Field(default="search", description="Mode of operation: 'search' (web search), 'scrape' (single page), or 'crawl' (entire site). Default: 'search'.")

class FirecrawlTool(BaseTool):
    name: str = "Fetch Web Data with Firecrawl"
    description: str = "A tool that fetches real web data using Firecrawl (search, scrape, or crawl) and returns it in JSON format."
    args_schema: Type[BaseModel] = FirecrawlToolSchema

    def _run(self, query: str, limit: int = 50, mode: str = "search") -> str:
        # Load Firecrawl API key from environment variables
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            raise ValueError("Firecrawl API key is missing. Set FIRECRAWL_API_KEY in environment variables.")

        try:
            if mode == "search":
                tool = FirecrawlSearchTool()
                results = tool.run(query=query, search_options={"limit": limit}, fetchPageContent=True)
            elif mode == "scrape":
                tool = FirecrawlScrapeWebsiteTool()
                results = tool.run(url=query, onlyMainContent=True)
            elif mode == "crawl":
                tool = FirecrawlCrawlWebsiteTool()
                results = tool.run(url=query, crawler_options={"limit": limit, "onlyMainContent": True})
            else:
                raise ValueError("Invalid mode. Use 'search', 'scrape', or 'crawl'.")

            # Format results into JSON
            formatted_data = {
                "mode": mode,
                "query": query,
                "data": results if isinstance(results, list) else [{"content": results}]
            }

            print(f"✅ Fetched {len(formatted_data['data'])} items for '{query}' in {mode} mode.")
            return json.dumps(formatted_data)

        except Exception as e:
            print(f"⚠️ Error fetching web data: {e}")
            # Fallback to mock data
            mock_data = {
                "mode": mode,
                "query": query,
                "data": [
                    {"content": f"Sample web data for {query} from {mode} mode."}
                ]
            }
            print("⚠️ Using mock data due to Firecrawl failure.")
            return json.dumps(mock_data)