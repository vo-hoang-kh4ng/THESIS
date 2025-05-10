
import os
import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class EXAAnswerToolSchema(BaseModel):
    query: str = Field(..., description="The question you want to ask Exa.")

class EXAAnswerTool(BaseTool):
    name: str = "Ask Exa a question"
    description: str = "A tool that asks Exa a question and returns the answer."
    args_schema: Type[BaseModel] = EXAAnswerToolSchema
    answer_url: str = "https://api.exa.ai/answer"

    def _run(self, query: str):
        api_key = os.getenv("EXA_API_KEY")
        if not api_key:
            return "Error: EXA_API_KEY environment variable is not set."

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "x-api-key": api_key
        }
        
        try:
            response = requests.post(
                self.answer_url,
                json={"query": query, "text": True},
                headers=headers,
                timeout=10  
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return f"Error: Failed to fetch answer from Exa - {str(e)}"

        try:
            response_data = response.json()
            answer = response_data.get("answer", "No answer found.")
            citations = response_data.get("citations", [])
            output = f"Answer: {answer}\n\n"
            if citations:
                output += "Citations:\n"
                for citation in citations:
                    output += f"- {citation.get('title', 'No title')} ({citation.get('url', '')})\n"
            return output
        except ValueError:
            return "Error: Invalid response format from Exa."