import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime
import re
import streamlit as st
import networkx as nx
import json
from crewai import Crew, Process
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from my_agents import create_llm, create_agents
from tasks import create_tasks
import agentops
import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
agentops.init(api_key=os.environ["AGENTOPS_API_KEY"], auto_start_session=False)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hàm tìm top influencers từ tweets
def find_top_influencers(tweets):
    """
    Tìm influencers dựa trên dữ liệu tweets.
    Input: Danh sách tweets (list of dict)
    Output: (top_influencers, top_opposers, G)
    """
    G = nx.DiGraph()  # Đồ thị quan hệ

    for tweet in tweets:
        user = tweet["user"]
        mentions = tweet.get("mentions", [])

        for mentioned_user in mentions:
            G.add_edge(user, mentioned_user)

    top_influencers = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)[:3]
    top_opposers = sorted(G.out_degree(), key=lambda x: x[1], reverse=True)[:3]

    return top_influencers, top_opposers, G

# Xây dựng mạng lưới ảnh hưởng
def build_influencer_network(tweets):
    """
    Xây dựng mạng lưới ảnh hưởng từ danh sách bài đăng.
    """
    G = nx.DiGraph()

    for tweet in tweets:
        user = tweet["user"]
        mentions = tweet.get("mentions", [])
        
        if user not in G:
            G.add_node(user, mentions=[], tweets=1)
        else:
            G.nodes[user]["tweets"] += 1

        for mentioned_user in mentions:
            if mentioned_user not in G:
                G.add_node(mentioned_user, mentions=[], tweets=0)
            G.add_edge(user, mentioned_user)
            G.nodes[user]["mentions"].append(mentioned_user)

    return G

# Phân tích mạng lưới ảnh hưởng
def analyze_influencer_network(G):
    """Phân tích mạng lưới ảnh hưởng"""
    pagerank = nx.pagerank(G)
    betweenness = nx.betweenness_centrality(G)

    influencers = []
    for user in G.nodes():
        influencers.append({
            "user": user,
            "pagerank": pagerank[user],
            "betweenness": betweenness[user],
            "tweets": G.nodes[user]["tweets"],
            "mentions": G.nodes[user]["mentions"]
        })

    influencers = sorted(influencers, key=lambda x: x["pagerank"], reverse=True)
    return influencers

# Vẽ biểu đồ mạng lưới ảnh hưởng
def display_influencer_graph(G):
    """
    Vẽ biểu đồ mạng lưới ảnh hưởng.
    """
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="gray", font_size=8, edge_color="gray")
    plt.title("Influencer Network")
    st.pyplot(plt)

# Phân tích phần trăm tiêu cực
def parse_negative_percentage(result):
    try:
        matches = re.findall(r'negative sentiment[^%]*?(\d+)%%?', str(result), re.IGNORECASE)
        return int(matches[-1]) if matches else 0
    except Exception as e:
        logger.error(f"⚠️ Parse error: {e}")
        return 0

# Vẽ biểu đồ xu hướng mentions và sentiment
def plot_mentions_and_sentiment(time_series_data):
    if len(time_series_data) < 2:
        st.warning("⚠️ Not enough data points to generate a trend chart.")
        return

    # Convert time_series_data to DataFrame
    df = pd.DataFrame(time_series_data)
    df["Date"] = pd.to_datetime(df["Date"])  # Ensure Date is in datetime format

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(8, 4))
    
    # Plot Mentions on the left y-axis
    ax1.plot(df["Date"], df["Mentions"], label="Mentions", color="blue")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Mentions Count", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    # Plot Sentiment Score on the right y-axis
    ax2 = ax1.twinx()
    ax2.plot(df["Date"], df["Sentiment Score"], label="Sentiment Score", color="green")
    ax2.set_ylabel("Sentiment Score (100 - % Negative)", color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    # Add title and adjust layout
    fig.suptitle("Mentions & Sentiment Trend Over Time")
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to prevent overlap
    st.pyplot(fig)

def run_social_media_monitoring(brand_name, max_retries=3):
    llm = create_llm()
    agents = create_agents(brand_name, llm)
    tasks = create_tasks(brand_name, agents)

    crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=True,
        process=Process.sequential,
        memory=True,
        long_term_memory=LongTermMemory(
            storage=LTMSQLiteStorage(
                db_path="THESIS-master/db/ltm_storage.db"
            )
        )
    )
    crisis_detected = False
    time_series_data = []

    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}: Starting crew kickoff for brand: {brand_name}")
            result = crew.kickoff()
            result_str = str(result)

            # Log the raw result to identify potential formatting issues
            logger.debug(f"Raw result: {result_str}")

            # Extract the report
            if "Agent: Re-ranking Agent" in result_str:
                re_ranking_start = result_str.index("Agent: Re-ranking Agent")
                full_report = result_str[re_ranking_start:]
            else:
                full_report = result_str

            # Parse negative sentiment
            negative_percentage = parse_negative_percentage(result_str)
            crisis_detected = negative_percentage > 50

            # Update time series data for trend plotting
            time_series_data.append({
                "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Mentions": len(result_str.split()),
                "Sentiment Score": 100 - negative_percentage
            })

            # Extract tweets from the researcher's output
            tweets = []
            try:
                tweets_start = result_str.find('{"data":')
                if tweets_start != -1:
                    tweets_end = result_str.rfind('}')
                    tweets_data = json.loads(result_str[tweets_start:tweets_end + 1])
                    tweets = tweets_data.get("data", [])
            except json.JSONDecodeError as e:
                logger.error(f"⚠️ Could not extract tweets from result for influencer analysis: {e}")
                # Fallback to mock data
                tweets = [
                    {"user": "UserA", "text": f"Ủng hộ {brand_name}!", "mentions": ["UserB", "UserC"]},
                    {"user": "UserD", "text": f"Tôi phản đối {brand_name}!", "mentions": ["UserE"]},
                    {"user": "UserF", "text": f"Mọi người nghĩ sao về {brand_name}?", "mentions": ["UserA", "UserD"]},
                ]
                logger.warning("⚠️ Using mock data for influencer analysis due to extraction failure.")

            # Perform influencer analysis
            if tweets:
                top_influencers, top_opposers, G = find_top_influencers(tweets)
                G = build_influencer_network(tweets)
                influencers = analyze_influencer_network(G)

                # Add influencer analysis to the report
                influencer_section = "\n\n**Influencer Analysis:**\n"
                influencer_section += "- **Top Influencers (Most Mentioned):**\n"
                for influencer, degree in top_influencers:
                    # Sanitize influencer name to avoid % or ) issues
                    influencer = str(influencer).replace("%", "%%").replace(")", "\\)")
                    influencer_section += f"  - {influencer}: Mentioned by {degree} users\n"
                influencer_section += "- **Top Opposers (Most Mentioning):**\n"
                for opposer, degree in top_opposers:
                    opposer = str(opposer).replace("%", "%%").replace(")", "\\)")
                    influencer_section += f"  - {opposer}: Mentioned {degree} users\n"
                influencer_section += "- **Top Influencers by PageRank:**\n"
                for influencer in influencers[:3]:
                    user = str(influencer['user']).replace("%", "%%").replace(")", "\\)")
                    influencer_section += f"  - {user}: PageRank {influencer['pagerank']:.4f}, Betweenness {influencer['betweenness']:.4f}, Tweets {influencer['tweets']}\n"

                full_report += influencer_section

                # Display the influencer network graph
                st.subheader("Influencer Network Visualization")
                display_influencer_graph(G)
            else:
                full_report += "\n\n**Influencer Analysis:**\n- No tweets available for influencer analysis."

            # Plot mentions and sentiment trends
            plot_mentions_and_sentiment(time_series_data)

            # Use f-strings and escape % properly
            logger.info(f"🔍 Parsed Negative Sentiment: {negative_percentage} percent")
            if crisis_detected:
                logger.warning(f"🚨 Crisis Detected! Negative Sentiment: {negative_percentage} percent")
            else:
                logger.info(f"✅ No Crisis. Negative Sentiment: {negative_percentage} percent")

            return full_report, crisis_detected, negative_percentage

        except RuntimeError as e:
            if "cannot schedule new futures after shutdown" in str(e):
                logger.warning(f"⚠️ ThreadPoolExecutor shutdown detected. Attempting to recover... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(5)
                continue
            else:
                raise e
        except Exception as e:
            logger.error(f"Error during monitoring attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                logger.error("Max retries reached. Exiting.")
                st.error(f"Failed to generate report after {max_retries} attempts. Error: {str(e)}")
                break
            time.sleep(10)

    return None, False, 0

try: 
    logger.info("\n\nStarting the task...\n\n")
    agentops.start_session()
    agentops.end_session("Success")
    logger.info("\n\nTask completed.\n\n")
except Exception as e:
    agentops.end_session("Fail", str(e))