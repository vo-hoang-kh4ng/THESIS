import time
import threading
import queue
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime
import re
import streamlit as st
from my_twitter_tool import TwitterIngestionTool
import networkx as nx
import json
from queue import Queue
from crewai import Crew, Process
from my_agents import create_llm, create_agents
from tasks import create_tasks

# Hàng đợi lưu dữ liệu Twitter
data_queue = Queue()

# Hàm giả lập lấy dữ liệu từ Twitter API
def fetch_data_twitter(brand_name):
    """
    Lấy dữ liệu Twitter từ hệ thống giám sát MXH.
    Giả lập dữ liệu trả về dưới dạng JSON.
    """
    data = {
        "data": [
            {"user": "UserA", "text": f"Ủng hộ {brand_name}!", "mentions": ["UserB", "UserC"]},
            {"user": "UserD", "text": f"Tôi phản đối {brand_name}!", "mentions": ["UserE"]},
            {"user": "UserF", "text": f"Mọi người nghĩ sao về {brand_name}?", "mentions": ["UserA", "UserD"]},
        ]
    }
    data_queue.put(json.dumps(data))  # Đưa dữ liệu vào hàng đợi dưới dạng chuỗi JSON

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
        matches = re.findall(r'negative sentiment[^%]*?(\d+)%', str(result), re.IGNORECASE)
        return int(matches[-1]) if matches else 0
    except Exception as e:
        print(f"⚠️ Parse error: {e}")
        return 0

# Vẽ biểu đồ xu hướng mentions và sentiment
def plot_mentions_and_sentiment(time_series_data):
    if len(time_series_data) < 2:
        st.warning("⚠️ Not enough data points to generate a trend chart.")
        return

    df = pd.DataFrame(time_series_data)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(df["Date"], df["Mentions"], label="Mentions", color="blue")
    ax1.set_ylabel("Mentions Count")
    ax1.set_xticklabels(df["Date"], rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(df["Date"], df["Sentiment Score"], label="Sentiment Score", color="green")
    ax2.set_ylabel("Sentiment Score (100 - % Negative)")

    fig.suptitle("Mentions & Sentiment Trend Over Time")
    st.pyplot(fig)

# Hàm chính chạy giám sát mạng xã hội
def run_social_media_monitoring(brand_name, twitter_data=None, max_retries=3):
    llm = create_llm()
    agents = create_agents(brand_name, llm)
    tasks = create_tasks(brand_name, agents, twitter_data)

    crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=True,
        process=Process.sequential
    )

    crisis_detected = False
    time_series_data = []

    for attempt in range(max_retries):
        try:
            latest_data = data_queue.get() if not data_queue.empty() else "No new data yet."
            tasks[0].description = f"Analyze latest social media data for {brand_name}: {latest_data}"

            result = crew.kickoff()
            result_str = str(result)

            if "Agent: Re-ranking Agent" in result_str:
                re_ranking_start = result_str.index("Agent: Re-ranking Agent")
                full_report = result_str[re_ranking_start:]
            else:
                full_report = result_str

            negative_percentage = parse_negative_percentage(result_str)
            crisis_detected = negative_percentage > 50

            time_series_data.append({
                "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Mentions": len(result_str.split()),
                "Sentiment Score": 100 - negative_percentage
            })

            print(f"🔍 Parsed Negative Sentiment: {negative_percentage}%")
            if crisis_detected:
                print(f"🚨 Crisis Detected! Negative Sentiment: {negative_percentage}%")
            else:
                print(f"✅ No Crisis. Negative Sentiment: {negative_percentage}%")

            plot_mentions_and_sentiment(time_series_data)
            return full_report, crisis_detected, negative_percentage
        except Exception as e:
            print(f"Error during monitoring attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                print("Max retries reached. Exiting.")
                break
            time.sleep(10)

    return None, False, 0