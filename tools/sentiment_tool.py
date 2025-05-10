import os
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from crewai.tools import BaseTool
from pydantic import Field, BaseModel
from pydantic.config import ConfigDict
from typing import Union, List, Dict, Optional, Any
from collections import Counter
import numpy as np
from functools import lru_cache
import asyncio
import time

# Configure logging
logging.basicConfig(
    filename="sentiment_tool.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Fix TensorFlow CUDA issue by forcing CPU execution
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Suppress TensorFlow warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)

class SentimentAnalysisTool(BaseTool):
    name: str = "advanced_sentiment_tool"
    description: str = (
        "Advanced sentiment analysis tool using ensemble of models for improved accuracy. "
        "Features include aspect-based sentiment analysis, multilingual support, "
        "emotion detection, and crisis signal identification with temporal analysis."
    )
    model_config = ConfigDict(extra='allow')

    def __init__(self, crisis_threshold: float = 50.0, custom_keywords: Optional[List[str]] = None):
        super().__init__()
        self.crisis_threshold = crisis_threshold
        self.custom_keywords = custom_keywords or [
            "crisis", "urgent", "emergency", "problem", "issue",
            "complaint", "negative", "bad", "terrible", "worst",
            "excellent", "great", "good", "positive", "amazing"
        ]
        self.primary_pipeline = None
        self.fallback_pipeline = None
        self.emotion_pipeline = None
        self.aspect_pipeline = None
        self.temporal_window = 24  # hours for temporal analysis
        self.sentiment_history = []  # Store historical sentiment data
        self._initialize_pipelines()

    def _initialize_pipelines(self):
        """Initialize all sentiment analysis pipelines."""
        try:
            # Primary sentiment pipeline (DistilBERT)
            self.primary_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1  # Force CPU
            )

            # Fallback multilingual pipeline
            self.fallback_pipeline = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=-1
            )

            # Emotion detection pipeline
            self.emotion_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=-1
            )

            # Aspect-based sentiment analysis
            self.aspect_pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1
            )

            logger.info("Successfully initialized all sentiment analysis pipelines")
        except Exception as e:
            logger.error(f"Failed to initialize pipelines: {str(e)}")
            raise

    def _extract_aspects(self, text: str) -> Dict[str, Dict]:
        """Extract aspects and their sentiment from text."""
        aspects = {
            "product": ["quality", "features", "design", "performance"],
            "service": ["support", "delivery", "response", "help"],
            "brand": ["reputation", "image", "trust", "value"],
            "price": ["cost", "value", "affordability", "pricing"]
        }
        
        results = {}
        for category, terms in aspects.items():
            for term in terms:
                result = self.aspect_pipeline(
                    text,
                    candidate_labels=[f"positive {term}", f"negative {term}", f"neutral {term}"],
                    multi_label=True
                )
                if result["scores"][0] > 0.5:  # Confidence threshold
                    sentiment = result["labels"][0].split()[0]
                    results[f"{category}_{term}"] = {
                        "sentiment": sentiment,
                        "confidence": result["scores"][0]
                    }
        return results

    def _detect_emotions(self, text: str) -> Dict[str, float]:
        """Detect emotions in text."""
        try:
            result = self.emotion_pipeline(text)
            return {result[0]["label"]: result[0]["score"]}
        except Exception as e:
            logger.warning(f"Emotion detection failed: {str(e)}")
            return {}

    def _detect_temporal_patterns(self, current_sentiment: float) -> Dict[str, Any]:
        """Detect temporal patterns in sentiment data."""
        if not self.sentiment_history:
            return {"pattern": "initial", "trend": "stable", "velocity": 0.0}
        
        # Calculate sentiment velocity (rate of change)
        recent_sentiments = self.sentiment_history[-self.temporal_window:]
        if len(recent_sentiments) > 1:
            velocity = (current_sentiment - recent_sentiments[0]) / len(recent_sentiments)
        else:
            velocity = 0.0
            
        # Detect patterns
        if velocity > 0.1:
            pattern = "accelerating_negative"
        elif velocity < -0.1:
            pattern = "accelerating_positive"
        elif abs(velocity) <= 0.1:
            pattern = "stable"
        else:
            pattern = "fluctuating"
            
        return {
            "pattern": pattern,
            "trend": "increasing" if velocity > 0 else "decreasing" if velocity < 0 else "stable",
            "velocity": velocity
        }

    def _detect_crisis_signals(self, sentiment_data: Dict[str, Any], temporal_data: Dict[str, Any]) -> List[str]:
        """Enhanced crisis signal detection using multiple indicators."""
        crisis_signals = []
        
        # Check sentiment threshold
        if sentiment_data["negative_percent"] > self.crisis_threshold:
            crisis_signals.append(f"High negative sentiment ({sentiment_data['negative_percent']:.1f}%)")
            
        # Check temporal patterns
        if temporal_data["pattern"] == "accelerating_negative":
            crisis_signals.append("Rapid negative sentiment acceleration")
            
        # Check emotion intensity
        if any(score > 0.8 for score in sentiment_data["emotions"].values()):
            crisis_signals.append("High emotion intensity detected")
            
        # Check aspect-based signals
        negative_aspects = [aspect for aspect, data in sentiment_data["aspects"].items() 
                          if data["sentiment"] == "negative" and data["confidence"] > 0.7]
        if negative_aspects:
            crisis_signals.append(f"Negative sentiment in key aspects: {', '.join(negative_aspects)}")
            
        return crisis_signals

    def _run(self, text: Union[str, List[str]] = None, **kwargs) -> str:
        """Run enhanced sentiment analysis with temporal patterns and crisis detection."""
        try:
            if not text:
                return "Error: No text provided for analysis"

            # Convert single text to list
            texts = [text] if isinstance(text, str) else text
            total = len(texts)

            # Initialize results storage
            results = []
            distribution = {"positive": 0, "neutral": 0, "negative": 0}
            emotions = {}
            aspects = {}

            # Process each text
            for t in texts:
                # Basic sentiment
                try:
                    result = self.primary_pipeline(t)[0]
                except Exception:
                    result = self.fallback_pipeline(t)[0]

                # Categorize sentiment
                if result["label"] == "POSITIVE" and result["score"] > 0.7:
                    category = "positive"
                elif result["label"] == "NEGATIVE" and result["score"] > 0.7:
                    category = "negative"
                else:
                    category = "neutral"

                # Update distribution
                distribution[category] += 1

                # Extract aspects and emotions
                text_aspects = self._extract_aspects(t)
                aspects.update(text_aspects)
                text_emotions = self._detect_emotions(t)
                for emotion, score in text_emotions.items():
                    emotions[emotion] = emotions.get(emotion, 0) + score

                results.append(result)

            # Calculate percentages
            for category in distribution:
                distribution[category] = (distribution[category] / total) * 100

            # Normalize emotions
            total_emotions = sum(emotions.values())
            if total_emotions > 0:
                emotions = {k: v/total_emotions for k, v in emotions.items()}

            # Update sentiment history
            self.sentiment_history.append(distribution["negative"])
            if len(self.sentiment_history) > self.temporal_window:
                self.sentiment_history.pop(0)

            # Detect temporal patterns
            temporal_data = self._detect_temporal_patterns(distribution["negative"])

            # Detect crisis signals
            sentiment_data = {
                "negative_percent": distribution["negative"],
                "emotions": emotions,
                "aspects": aspects
            }
            crisis_signals = self._detect_crisis_signals(sentiment_data, temporal_data)
            crisis_detected = len(crisis_signals) > 0

            # Format output
            output = [
                "Advanced Sentiment Analysis Report:",
                f"- Sentiment Distribution:",
                f"  - Positive: {distribution['positive']:.2f}%",
                f"  - Neutral: {distribution['neutral']:.2f}%",
                f"  - Negative: {distribution['negative']:.2f}%",
                f"- Temporal Analysis:",
                f"  - Pattern: {temporal_data['pattern']}",
                f"  - Trend: {temporal_data['trend']}",
                f"  - Velocity: {temporal_data['velocity']:.2f}",
                f"- Top Emotions:"
            ]
            
            # Add top emotions
            for emotion, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]:
                output.append(f"  - {emotion}: {score:.2f}")
            
            # Add key aspects
            output.append(f"- Key Aspects:")
            for aspect, data in sorted(aspects.items(), key=lambda x: x[1]["confidence"], reverse=True)[:3]:
                output.append(f"  - {aspect}: {data['sentiment']} ({data['confidence']:.2f})")
            
            # Add crisis status
            if crisis_detected:
                output.append(f"- Crisis Status: DETECTED")
                output.append(f"- Crisis Signals:")
                for signal in crisis_signals:
                    output.append(f"  - {signal}")
            else:
                output.append(f"- Crisis Status: No crisis detected")

            final_output = "\n".join(output)
            logger.info(f"Advanced sentiment analysis completed: {final_output}")
            return final_output

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}", exc_info=True)
            return f"Error: Sentiment analysis failed - {str(e)}"

    async def _arun(self, text: Union[str, List[str]] = None, **kwargs) -> str:
        """Asynchronous implementation using a thread pool for I/O-bound tasks."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._run(text, **kwargs))

    def test_sentiment_tool(self) -> Dict:
        """Automated test for sentiment analysis tool functionality."""
        test_cases = [
            {"text": "I love using CrewAI!", "expected": "positive"},
            {"text": "CrewAI is okay, nothing special.", "expected": "neutral"},
            {"text": "CrewAI is terrible and broken!", "expected": "negative"}
        ]
        results = []
        for case in test_cases:
            try:
                output = self._run(case["text"])
                sentiment = "positive" if "Positive" in output else "negative" if "Negative" in output else "neutral"
                passed = sentiment == case["expected"]
                results.append({
                    "text": case["text"],
                    "expected": case["expected"],
                    "actual": sentiment,
                    "passed": passed,
                    "output": output
                })
                logger.info(f"Test case '{case['text']}': {'Passed' if passed else 'Failed'}")
            except Exception as e:
                logger.error(f"Test case '{case['text']}' failed with error: {str(e)}")
                results.append({
                    "text": case["text"],
                    "expected": case["expected"],
                    "actual": "error",
                    "passed": False,
                    "output": str(e)
                })
        return {"test_results": results}