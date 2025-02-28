import time
from crewai import Crew, Process
from my_agents import create_llm, create_agents
from tasks import create_tasks
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

# Hàng đợi để lưu dữ liệu mới nhất từ mạng xã hội
data_queue = queue.Queue()

def fetch_data_twitter(brand_name: str, interval: int = 60):
    """
    Hàm chạy vòng lặp, mỗi interval giây gọi TwitterIngestionTool để lấy tweet.
    """
    tool = TwitterIngestionTool()
    while True:
        try:
            result = tool.run(query=brand_name)
            # Lưu vào queue
            data_queue.put({"platform": "twitter", "data": result})
            time.sleep(interval)
        except Exception as e:
            print(f"Twitter fetch error: {e}")
            time.sleep(interval)
def build_influencer_network(tweets):
    """
    Xây dựng mạng lưới ảnh hưởng từ danh sách bài đăng.
    - Tạo node cho mỗi user.
    - Kết nối các user dựa trên mentions.
    """
    G = nx.DiGraph()  # Tạo đồ thị có hướng

    for tweet in tweets:
        user = tweet["user"]  # Người đăng bài
        mentions = tweet.get("mentions", [])  # Danh sách tài khoản được nhắc đến
        
        # Thêm user vào mạng lưới nếu chưa tồn tại
        if user not in G:
            G.add_node(user, mentions=[], tweets=1)
        else:
            G.nodes[user]["tweets"] += 1  # Cập nhật số bài đăng của user

        # Kết nối user với các mentions
        for mentioned_user in mentions:
            if mentioned_user not in G:
                G.add_node(mentioned_user, mentions=[], tweets=0)  # Nếu chưa có, thêm vào
            G.add_edge(user, mentioned_user)  # Tạo kết nối user → mentioned_user
            
            # Lưu lại danh sách mentions cho mỗi user
            G.nodes[user]["mentions"].append(mentioned_user)

    return G

def analyze_influencer_network(G):
    """Phân tích mạng lưới ảnh hưởng"""
    pagerank = nx.pagerank(G)  # Tính PageRank
    betweenness = nx.betweenness_centrality(G)  # Tính betweenness centrality

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
def display_influencer_graph(G):
    """
    Vẽ biểu đồ mạng lưới ảnh hưởng.
    - Không sử dụng sentiment (tất cả node có màu giống nhau).
    """
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)  # Tự động căn chỉnh vị trí nodes
    
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="gray", font_size=8, edge_color="gray")
    plt.title("Influencer Network")
    st.pyplot(plt)
def find_top_influencers(tweets):
    """Xác định danh sách top influencers và top opposers từ mạng xã hội"""
    G = build_influencer_network(tweets)  # Xây dựng mạng lưới
    influencers = analyze_influencer_network(G)  # Phân tích mạng

    # Lọc ra top influencers (nếu ít hơn 5, trả về tất cả)
    top_influencers = influencers[:5] if len(influencers) >= 5 else influencers

    # Lọc ra top opposers (nếu ít hơn 5, trả về tất cả)
    top_opposers = influencers[-5:] if len(influencers) >= 5 else influencers

    return top_influencers, top_opposers, G  # ✅ Đảm bảo luôn trả về 3 giá trị

def parse_negative_percentage(result):
    try:
        matches = re.findall(r'negative sentiment[^%]*?(\d+)%', str(result), re.IGNORECASE)
        return int(matches[-1]) if matches else 0  # ✅ Lấy giá trị cuối cùng (mới nhất)
    except Exception as e:
        print(f"⚠️ Parse error: {e}")
        return 0

def plot_mentions_and_sentiment(time_series_data):
    if len(time_series_data) < 2:
        st.warning("⚠️ Not enough data points to generate a trend chart.")
        return

    df = pd.DataFrame(time_series_data)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    
    # Vẽ Mentions
    ax1.plot(df["Date"], df["Mentions"], label="Mentions", color="blue")
    ax1.set_ylabel("Mentions Count")
    ax1.set_xticklabels(df["Date"], rotation=45, ha="right")

    # Vẽ Sentiment Score trên cùng biểu đồ
    ax2 = ax1.twinx()
    ax2.plot(df["Date"], df["Sentiment Score"], label="Sentiment Score", color="green")
    ax2.set_ylabel("Sentiment Score (100 - % Negative)")

    fig.suptitle("Mentions & Sentiment Trend Over Time")
    
    # ✅ Thay plt.show() bằng st.pyplot()
    st.pyplot(fig)

def run_social_media_monitoring(brand_name, max_retries=3):
    llm = create_llm()
    agents = create_agents(brand_name, llm)
    tasks = create_tasks(brand_name, agents)

    crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=True,
        process=Process.sequential
    )

    # SỬA ĐOẠN DƯỚI ĐÂY: Đổi fetch_social_media_data → fetch_data_twitter
    if not hasattr(run_social_media_monitoring, "data_thread"):
        data_thread = threading.Thread(target=fetch_data_twitter, args=(brand_name,), daemon=True)
        data_thread.start()
        run_social_media_monitoring.data_thread = data_thread

    crisis_detected = False
    time_series_data = []
    last_result = None

    for attempt in range(max_retries):
        try:
            latest_data = data_queue.get() if not data_queue.empty() else "No new data yet."
            tasks[0].description = f"Analyze latest social media data for {brand_name}: {latest_data}"

            # CHẠY PHÂN TÍCH
            result = crew.kickoff()
            result_str = str(result)

            # ✅ FIX: Đảm bảo lấy đúng báo cáo từ Re-ranking Agent
            if "Agent: Re-ranking Agent" in result_str:
                re_ranking_start = result_str.index("Agent: Re-ranking Agent")
                full_report = result_str[re_ranking_start:]
            else:
                full_report = result_str

            # ✅ FIX: Lấy đúng số % negative sentiment
            negative_percentage = parse_negative_percentage(result_str)
            crisis_detected = negative_percentage > 50

            # ✅ Lưu dữ liệu time-series để vẽ biểu đồ
            time_series_data.append({
                "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Mentions": len(result_str.split()),  # Giả định mentions = số lượng từ trong kết quả
                "Sentiment Score": 100 - negative_percentage  # Giả định điểm sentiment là 100 - % negative
            })

            print(f"🔍 Parsed Negative Sentiment: {negative_percentage}%")
            if crisis_detected:
                print(f"🚨 Crisis Detected! Negative Sentiment: {negative_percentage}%")
            else:
                print(f"✅ No Crisis. Negative Sentiment: {negative_percentage}%")

            # ✅ Hiển thị biểu đồ sau khi lấy dữ liệu
            plot_mentions_and_sentiment(time_series_data)

            return full_report, crisis_detected, negative_percentage
        except Exception as e:
            print(f"Error during monitoring attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                print("Max retries reached. Exiting.")
                break
            time.sleep(10)

    return None, False, 0  # Trả về giá trị mặc định nếu có lỗi
