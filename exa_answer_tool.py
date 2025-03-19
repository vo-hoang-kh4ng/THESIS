# exa_answer_tool.py
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
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "x-api-key": os.getenv("EXA_API_KEY")
        }
        
        try:
            response = requests.post(
                self.answer_url,
                json={"query": query, "text": True},
                headers=headers,
            )
            response.raise_for_status() 
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            raise
        except Exception as err:
            print(f"Other error occurred: {err}")
            raise

        response_data = response.json()
        answer = response_data.get("answer", "No answer found.")
        citations = response_data.get("citations", [])
        output = f"Answer: {answer}\n\n"
        if citations:
            output += "Citations:\n"
            for citation in citations:
                output += f"- {citation.get('title', 'No title')} ({citation.get('url', '')})\n"

        return output
