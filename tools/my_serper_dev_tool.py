import os
import requests
import json
import logging
from crewai.tools import BaseTool
from langchain_community.document_loaders import WebBaseLoader
from typing import Optional, Dict, List
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SearchToolBase(BaseTool):
    """Base class for search tools to share common functionality."""
    name: str = "base_search_tool"
    description: str = "Base tool for search operations."

    max_retries: int = 3
    retry_delay: int = 2

    @property
    def api_key(self) -> Optional[str]:
        """Get API key from environment variables."""
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            logger.error("SERPER_API_KEY is missing in environment variables.")
            raise ValueError("SERPER_API_KEY is required.")
        return api_key

    def _api_request(self, url: str, payload: Dict, retries: int = 0) -> Dict:
        """Perform API request with retry logic."""
        try:
            headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if retries < self.max_retries:
                logger.warning(f"API request failed (attempt {retries + 1}/{self.max_retries}): {str(e)}. Retrying...")
                time.sleep(self.retry_delay)
                return self._api_request(url, payload, retries + 1)
            logger.error(f"API request failed after {self.max_retries} retries: {str(e)}")
            return {"organic": []}

class InternetSearchTool(SearchToolBase):
    name: str = "search_internet"
    description: str = (
        "Search the internet using Google for information. "
        "Returns up to 5 results with titles, snippets, and links."
    )

    def _run(self, query: str) -> str:
        """
        Search the internet for information.

        Args:
            query (str): The search query.

        Returns:
            str: Formatted search results.
        """
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid search query.")
            return "Error: Invalid search query."
        
        url = "https://google.serper.dev/search"
        payload = {"q": query, "num": 5}
        response = self._api_request(url, payload)
        results = response.get("organic", [])

        formatted_results = []
        for result in results:
            formatted_results.append(
                f"{result.get('title', 'No title')}\n{result.get('snippet', 'No snippet')}\n{result.get('link', 'No link')}\n\n"
            )
        return f"Search results for '{query}':\n\n" + "".join(formatted_results) if formatted_results else "No results found."

class InstagramSearchTool(SearchToolBase):
    name: str = "search_instagram"
    description: str = (
        "Search Instagram pages using Google by filtering site:instagram.com. "
        "Returns up to 5 results with titles, snippets, and links."
    )

    def _run(self, query: str) -> str:
        """
        Search Instagram for information.

        Args:
            query (str): The search query.

        Returns:
            str: Formatted search results.
        """
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid search query.")
            return "Error: Invalid search query."
        
        url = "https://google.serper.dev/search"
        payload = {"q": f"site:instagram.com {query}", "num": 5}
        response = self._api_request(url, payload)
        results = response.get("organic", [])

        formatted_results = []
        for result in results:
            formatted_results.append(
                f"{result.get('title', 'No title')}\n{result.get('snippet', 'No snippet')}\n{result.get('link', 'No link')}\n\n"
            )
        return f"Instagram search results for '{query}':\n\n" + "".join(formatted_results) if formatted_results else "No results found."

class OpenPageTool(BaseTool):
    name: str = "open_page"
    description: str = (
        "Open a webpage and return its content."
    )

    def _run(self, url: str) -> str:
        """
        Open a webpage and return its content.

        Args:
            url (str): The URL of the webpage to open.

        Returns:
            str: Content of the webpage.
        """
        if not isinstance(url, str) or not url.strip():
            logger.error("Invalid URL.")
            return "Error: Invalid URL."
        
        try:
            loader = WebBaseLoader(url)
            content = loader.load()
            return content[0].page_content if content else "No content available."
        except Exception as e:
            logger.error(f"Failed to open page {url}: {str(e)}")
            return f"Error: Failed to load page - {str(e)}"