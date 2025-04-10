import os
import logging
from transformers import pipeline
from crewai.tools import BaseTool
from pydantic import Field
from pydantic.config import ConfigDict
from typing import Union, List, Dict
from collections import Counter

# Configure logging to write to a file for debugging
logging.basicConfig(
    filename="sentiment_tool.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Fix TensorFlow CUDA issue by forcing CPU execution
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Suppress TensorFlow warnings about duplicate registrations
logging.getLogger("tensorflow").setLevel(logging.ERROR)

class SentimentAnalysisTool(BaseTool):
    name: str = "distilbert_sentiment_tool"
    description: str = "Analyze sentiment of a given text or list of texts using DistilBERT, categorize into positive/neutral/negative, compute distribution percentages, and detect crisis signals."
    model_config = ConfigDict(extra='allow')

    def __init__(self):
        super().__init__()
        try:
            # Khởi tạo pipeline trong __init__ thay vì ClassVar
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1,  # Sử dụng CPU; nếu có GPU, có thể đặt device=0
                batch_size=16  # Xử lý hàng loạt để tăng hiệu suất
            )
            logging.info("Sentiment pipeline initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize sentiment pipeline: {str(e)}")
            raise RuntimeError(f"Failed to initialize sentiment pipeline: {str(e)}")

    def _run(self, text: Union[str, List[str]] = None, **kwargs) -> str:
        """
        Perform sentiment analysis on a single text or a list of texts.
        Categorize sentiments, compute distribution percentages, detect crisis signals,
        and provide a chain-of-thought explanation.
        """
        logging.info(f"SentimentAnalysisTool received input: {text}")
        try:
            # Step 1: Validate input
            if not text:
                logging.warning("No text provided.")
                return "No text provided."

            # Convert single text to a list for uniform processing
            texts = [text] if isinstance(text, str) else text
            if not isinstance(texts, list):
                logging.error(f"Invalid input type: {type(texts)}")
                return f"Error: Input must be a string or list of strings, got {type(texts)}"

            # Step 2: Run sentiment analysis on each text
            # Xử lý hàng loạt để tăng hiệu suất
            results = self.sentiment_pipeline(texts, truncation=True, max_length=512)
            logging.info(f"Raw sentiment results: {results}")

            # Step 3: Map DistilBERT labels to expected categories (positive, neutral, negative)
            categorized_results = []
            for result in results:
                label = result["label"]
                score = result["score"]
                # DistilBERT outputs POSITIVE or NEGATIVE; we'll map to positive/neutral/negative
                if label == "POSITIVE":
                    sentiment = "positive" if score > 0.7 else "neutral"
                else:  # NEGATIVE
                    sentiment = "negative" if score > 0.7 else "neutral"
                categorized_results.append({"label": sentiment, "score": score})

            # Step 4: Compute sentiment distribution percentages
            total = len(categorized_results)
            sentiment_counts = Counter(result["label"] for result in categorized_results)
            distribution = {
                "positive": (sentiment_counts.get("positive", 0) / total) * 100 if total > 0 else 0,
                "neutral": (sentiment_counts.get("neutral", 0) / total) * 100 if total > 0 else 0,
                "negative": (sentiment_counts.get("negative", 0) / total) * 100 if total > 0 else 0
            }

            # Step 5: Detect crisis signals (negative > 50%)
            crisis_detected = distribution["negative"] > 50
            crisis_alert = (
                f"Crisis Detected: Negative sentiment = {distribution['negative']:.2f}% (> 50%)"
                if crisis_detected
                else f"No Crisis: Negative sentiment = {distribution['negative']:.2f}%"
            )

            # Step 6: Identify key themes (basic keyword extraction for now)
            all_text = " ".join(texts).lower()
            common_words = ["good", "bad", "love", "hate", "great", "terrible", "crisis", "problem", "excellent"]
            themes = {word: all_text.count(word) for word in common_words if word in all_text}
            key_themes = ", ".join(f"{word} ({count})" for word, count in themes.items()) or "None identified"

            # Step 7: Build chain-of-thought explanation
            cot = [
                "Chain-of-Thought Explanation:",
                "1. Data Inputs:",
                f"   - Received {total} text(s): {texts[:3]}{'...' if len(texts) > 3 else ''}",
                "2. Sentiment Classification Method:",
                "   - Used DistilBERT model (distilbert-base-uncased-finetuned-sst-2-english).",
                "   - Mapped labels: POSITIVE (score > 0.7) -> positive, NEGATIVE (score > 0.7) -> negative, else neutral.",
                f"   - Raw results: {results[:3]}{'...' if len(results) > 3 else ''}",
                f"   - Categorized results: {categorized_results[:3]}{'...' if len(categorized_results) > 3 else ''}",
                "3. Percentage Calculation:",
                f"   - Positive: {distribution['positive']:.2f}% ({sentiment_counts.get('positive', 0)}/{total})",
                f"   - Neutral: {distribution['neutral']:.2f}% ({sentiment_counts.get('neutral', 0)}/{total})",
                f"   - Negative: {distribution['negative']:.2f}% ({sentiment_counts.get('negative', 0)}/{total})",
                "4. Theme Identification Process:",
                f"   - Extracted common words: {common_words}",
                f"   - Identified themes: {themes}",
                "5. Validation Steps:",
                "   - Cross-checked scores with confidence thresholds (0.7 for positive/negative).",
                "   - Ensured all texts were processed without errors."
            ]

            # Step 8: Format the final output as per expected_output
            output = [
                "Sentiment Analysis Report:",
                f"- Sentiment Distribution Percentages:",
                f"  - Positive: {distribution['positive']:.2f}%",
                f"  - Neutral: {distribution['neutral']:.2f}%",
                f"  - Negative: {distribution['negative']:.2f}%",
                f"- Key Themes or Topics Identified: {key_themes}",
                f"- Crisis Alert: {crisis_alert}",
                "\n".join(cot)
            ]

            final_output = "\n".join(output)
            logging.info(f"Sentiment analysis completed successfully: {final_output}")
            return final_output

        except Exception as e:
            logging.error(f"Sentiment analysis failed: {str(e)}", exc_info=True)
            return f"Error: Sentiment analysis failed with exception: {str(e)}"

    async def _arun(self, text: Union[str, List[str]] = None, **kwargs) -> str:
        """
        Asynchronous version of the sentiment analysis tool (not implemented).
        """
        raise NotImplementedError("Async run not implemented for SentimentAnalysisTool")