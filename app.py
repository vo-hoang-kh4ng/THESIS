import streamlit as st
from my_utils import (
    run_social_media_monitoring, parse_negative_percentage,
    find_top_influencers, display_influencer_graph, data_queue, fetch_data_twitter
)
import time
import os
import json
import networkx as nx

os.environ["OTEL_SDK_DISABLED"] = "true"
st.title("📊 Real-Time Crisis Sentiment Analysis Crew")
st.write("This app monitors social media sentiment for a brand in real-time, detects potential crises, and identifies key influencers.")

# Nhập tên thương hiệu
brand = st.text_input("🔍 Enter the brand or influencer name:")

# Khởi tạo session state
if "result" not in st.session_state:
    st.session_state.result = None
if "crisis_detected" not in st.session_state:
    st.session_state.crisis_detected = False
if "negative_percent" not in st.session_state:
    st.session_state.negative_percent = 0
if "show_report" not in st.session_state:
    st.session_state.show_report = False
if "top_influencers" not in st.session_state:
    st.session_state.top_influencers = []
if "top_opposers" not in st.session_state:
    st.session_state.top_opposers = []
if "graph" not in st.session_state:
    st.session_state.graph = None

# Nút lấy dữ liệu Twitter
if st.button("📥 Fetch Twitter Data"):
    with st.spinner("Fetching data from Twitter..."):
        try:
            fetch_data_twitter(brand)
            st.success("✅ Data fetched successfully!")
        except Exception as e:
            st.error(f"⚠️ Error fetching data: {e}")
# Improved run_analysis button logic
if st.button("⚡ Run Analysis"):
    with st.spinner("Running analysis..."):
        if not data_queue.empty():
            try:
                twitter_data = data_queue.get()
                result, crisis_detected, negative_percent = run_social_media_monitoring(brand, twitter_data)
                st.session_state.result = result
                st.session_state.crisis_detected = crisis_detected
                st.session_state.negative_percent = negative_percent
                st.session_state.show_report = True
            except Exception as e:
                st.error(f"⚠️ Error during analysis: {e}")
        else:
            st.error("⚠️ Please fetch Twitter data first!")

# Hiển thị báo cáo từ final_report.md
if st.session_state.show_report and os.path.exists("final_report.md"):
    with open("final_report.md", "r", encoding="utf-8") as file:
        report_content = file.read()
    st.markdown("## 📄 Full Report from final_report.md")
    st.markdown(report_content, unsafe_allow_html=True)

# Hiển thị báo cáo khủng hoảng
if st.session_state.result:
    st.markdown("## 📊 Full Crisis Report")
    st.markdown(st.session_state.result, unsafe_allow_html=True)

    if st.session_state.crisis_detected:
        st.error(f"## 🔥 CRISIS DETECTED! Negative: {st.session_state.negative_percent}%")
    else:
        st.success(f"## ✅ No Crisis. Negative: {st.session_state.negative_percent}%")

# Phân tích mạng lưới ảnh hưởng
st.markdown("---")
st.markdown("## 🏆 Influencer Network Analysis")

if st.button("🔍 Analyze Influencer Network"):
    with st.spinner("Analyzing influencer impact..."):
        try:
            if not data_queue.empty():
                latest_data = data_queue.get()
                if isinstance(latest_data, str):
                    latest_data = json.loads(latest_data)
                tweets = latest_data.get("data", [])
            else:
                st.warning("⚠️ No new Twitter data available.")
                tweets = []

            if tweets:
                top_influencers, top_opposers, G = find_top_influencers(tweets)
                st.session_state.top_influencers = top_influencers
                st.session_state.top_opposers = top_opposers
                st.session_state.graph = G
                st.success("✅ Influencer analysis complete!")
            else:
                st.session_state.top_influencers = []
                st.session_state.top_opposers = []
                st.session_state.graph = nx.DiGraph()

        except Exception as e:
            st.error(f"⚠️ Error analyzing influencers: {e}")
            st.session_state.graph = nx.DiGraph()

# Hiển thị danh sách influencers
if st.session_state.top_influencers:
    st.markdown("### 🌟 Top Influencers:")
    for influencer in st.session_state.top_influencers:
        st.write(f"- {influencer[0]} (Influence Score: {influencer[1]})")

if st.session_state.top_opposers:
    st.markdown("### 🔥 Top Opposers:")
    for opposer in st.session_state.top_opposers:
        st.write(f"- {opposer[0]} (Opposition Score: {opposer[1]})")

# Hiển thị biểu đồ mạng lưới
if st.session_state.graph is not None:
    display_influencer_graph(st.session_state.graph)
else:
    st.warning("⚠️ No influencer network available to display.")