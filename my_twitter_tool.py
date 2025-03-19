# my_twitter_tool.py
import os
import tweepy
from crewai.tools import BaseTool
from pydantic import Field
from pydantic.config import ConfigDict

# Improved TwitterIngestionTool class
class TwitterIngestionTool(BaseTool):
    name: str = "twitter_ingestion_tool"
    description: str = "Fetch tweets or user data from Twitter (X) via official API."
    model_config = ConfigDict(extra='allow')
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("TWITTER_API_KEY", "")
        self.api_secret = os.getenv("TWITTER_API_SECRET", "")
        self.access_token = os.getenv("TWITTER_ACCESS_TOKEN", "")
        self.access_token_secret = os.getenv("TWITTER_ACCESS_SECRET", "")

        auth = tweepy.OAuth1UserHandler(
            self.api_key,
            self.api_secret,
            self.access_token,
            self.access_token_secret
        )
        self.api = tweepy.API(auth)

    def _run(self, query: str = None, **kwargs) -> str:
        """
        Override phương thức abstract _run(...) thay vì run(...).
        """
        try:
            if not query:
                return "No query provided."

            tweets = self.api.search_tweets(q=query, count=10, tweet_mode='extended', lang='en')
            results = []
            for t in tweets:
                results.append({
                    "id": t.id_str,
                    "created_at": str(t.created_at),
                    "text": t.full_text,
                    "user": t.user.screen_name,
                    "retweet_count": t.retweet_count,
                    "favorite_count": t.favorite_count
                })
            return str(results)
        except tweepy.TweepError as e:
            return f"Twitter API error: {e}"
        except Exception as e:
            return f"Error: {e}"
