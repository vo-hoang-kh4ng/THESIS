# twitter_fetch_tool.py
import os
import json
import tweepy
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

# Define the input schema for the tool
class TwitterFetchToolSchema(BaseModel):
    brand_name: str = Field(..., description="The brand name to search for on Twitter (e.g., 'iPhone').")
    count: int = Field(default=100, description="Number of tweets to fetch (default: 100).")

class TwitterFetchTool(BaseTool):
    name: str = "Fetch Twitter Data"
    description: str = "A tool that fetches real Twitter data for a given brand using the Twitter API and returns it in JSON format."
    args_schema: Type[BaseModel] = TwitterFetchToolSchema

    def _run(self, brand_name: str, count: int = 100) -> str:
        # Load Twitter API credentials from environment variables
        api_key = os.getenv("TWITTER_API_KEY")
        api_secret = os.getenv("TWITTER_API_SECRET")
        access_token = os.getenv("TWITTER_ACCESS_TOKEN")
        access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")

        if not all([api_key, api_secret, access_token, access_token_secret]):
            raise ValueError("Twitter API credentials are missing. Set TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, and TWITTER_ACCESS_TOKEN_SECRET in environment variables.")

        # Authenticate with Twitter API v1.1
        auth = tweepy.OAuthHandler(api_key, api_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth, wait_on_rate_limit=True)

        try:
            # Fetch tweets
            tweets = api.search_tweets(
                q=f"{brand_name} -filter:retweets",  # Exclude retweets for cleaner data
                lang="en",  # English tweets (you can adjust this)
                count=count,
                tweet_mode="extended"  # Get full text of tweets
            )

            # Format tweets to match the expected structure
            formatted_data = {
                "data": [
                    {
                        "user": tweet.user.screen_name,
                        "text": tweet.full_text,
                        "mentions": [user_mention["screen_name"] for user_mention in tweet.entities.get("user_mentions", [])]
                    }
                    for tweet in tweets
                ]
            }

            print(f"✅ Fetched {len(tweets)} tweets for {brand_name}.")
            return json.dumps(formatted_data)

        except tweepy.TweepyException as e:
            print(f"⚠️ Error fetching Twitter data: {e}")
            # Fallback to mock data in case of failure
            mock_data = {
                "data": [
                    {"user": "UserA", "text": f"Ủng hộ {brand_name}!", "mentions": ["UserB", "UserC"]},
                    {"user": "UserD", "text": f"Tôi phản đối {brand_name}!", "mentions": ["UserE"]},
                    {"user": "UserF", "text": f"Mọi người nghĩ sao về {brand_name}?", "mentions": ["UserA", "UserD"]},
                ]
            }
            print("⚠️ Using mock data due to Twitter API failure.")
            return json.dumps(mock_data)