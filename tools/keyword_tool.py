import spacy
from collections import Counter
from crewai.tools import BaseTool
from typing import Union, List

# Load spaCy’s English model
nlp = spacy.load("en_core_web_sm")

class DynamicKeywordExtractorTool(BaseTool):
    name: str = "Dynamic Keyword Extractor Tool"
    description: str = "Extracts dynamic keywords from text using entity recognition and frequency analysis."

    def _run(self, texts: Union[str, List[str]], top_n: int = 5) -> list:
        """
        Extract dynamic keywords from texts using entity recognition and frequency.

        Args:
            texts (Union[str, List[str]]): Text or list of texts to process.
            top_n (int): Number of keywords to return (default: 5).

        Returns:
            list: Top N keywords extracted.
        """
        if isinstance(texts, str):
            texts = [texts]
        all_text = " ".join(texts)
        doc = nlp(all_text)
        entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "PRODUCT", "GPE", "EVENT"]]
        entity_freq = Counter(entities)
        return [keyword for keyword, freq in entity_freq.most_common(top_n)]