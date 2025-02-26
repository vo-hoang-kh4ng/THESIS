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

# Hàng đợi để lưu dữ liệu mới nhất từ mạng xã hội
data_queue = queue.Queue()

def fetch_social_media_data(brand_name):
    """Lấy dữ liệu thời gian thực từ mạng xã hội bằng SerperDevTool."""
    from crewai_tools import SerperDevTool
    search_tool = SerperDevTool()
    while True:
        try:
            # Tìm kiếm dữ liệu thực tế từ web liên quan đến brand_name
            search_query = f"{brand_name} site:twitter.com OR site:instagram.com OR site:facebook.com -inurl:(login)"
            result = search_tool.run(search_query=search_query)
            data_queue.put(result)
            time.sleep(60)  # Cập nhật mỗi 60 giây
        except Exception as e:
            print(f"Error fetching data: {e}")
            time.sleep(60)
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

    if not hasattr(run_social_media_monitoring, "data_thread"):
        data_thread = threading.Thread(target=fetch_social_media_data, args=(brand_name,), daemon=True)
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
                full_report = result_str[re_ranking_start:]  # Lấy báo cáo từ Re-ranking Agent
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
