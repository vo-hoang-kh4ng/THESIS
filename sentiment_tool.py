import os
import logging
from transformers import pipeline
from crewai.tools import BaseTool
from pydantic import Field
from pydantic.config import ConfigDict
from typing import ClassVar

# Fix TensorFlow CUDA issue by forcing CPU execution
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Suppress TensorFlow warnings about duplicate registrations
logging.getLogger("tensorflow").setLevel(logging.ERROR)

class SentimentAnalysisTool(BaseTool):
    name: str = "distilbert_sentiment_tool"
    description: str = "Analyze sentiment of a given text using DistilBERT."
    model_config = ConfigDict(extra='allow')
    sentiment_pipeline: ClassVar = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def _run(self, text: str = None, **kwargs) -> str:
        """
        Perform sentiment analysis on the given text.
        """
        try:
            if not text:
                return "No text provided."

            result = self.sentiment_pipeline(text)[0]
            return f"Label: {result['label']}, Score: {result['score']:.4f}"
        except Exception as e:
            return f"Error: {e}"
