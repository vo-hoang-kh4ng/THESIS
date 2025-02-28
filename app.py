import streamlit as st
from my_utils import run_social_media_monitoring, parse_negative_percentage, find_top_influencers, display_influencer_graph
import time
import os
from my_utils import display_influencer_graph

st.title("Real-Time Crisis Sentiment Analysis Crew")
st.write("This app monitors social media sentiment for a brand in real-time, detects potential crises, and identifies key influencers.")

# Nhập thương hiệu hoặc nhân vật cần theo dõi
brand = st.text_input("Enter the brand or influencer name:")

# Session state để lưu trạng thái giữa các lần chạy
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

# **Chạy phân tích chính**
if st.button("Run Analysis"):
    with st.spinner("Running analysis..."):
        # Gọi hàm phân tích và lấy kết quả đầy đủ
        result, crisis_detected, negative_percent = run_social_media_monitoring(brand)

        # Cập nhật Session State
        st.session_state.result = result
        st.session_state.crisis_detected = crisis_detected
        st.session_state.negative_percent = negative_percent
        st.session_state.show_report = True  # Kích hoạt hiển thị báo cáo
        time.sleep(1)  # Giả lập độ trễ để giao diện cập nhật

# **Kiểm tra nếu báo cáo tổng hợp `final_report.md` tồn tại**
if st.session_state.show_report and os.path.exists("final_report.md"):
    with open("final_report.md", "r", encoding="utf-8") as file:
        report_content = file.read()
    
    st.markdown("## 📄 Full Report from `final_report.md`")
    st.markdown(report_content, unsafe_allow_html=True)  # Hiển thị nội dung báo cáo

# **Hiển thị báo cáo phân tích khủng hoảng**
if st.session_state.result:
    st.markdown("## Full Crisis Report")
    st.markdown(st.session_state.result, unsafe_allow_html=True)  
    print(f"DEBUG: Displaying Negative Sentiment: {st.session_state.negative_percent}%")

    # Hiển thị trạng thái khủng hoảng
    if st.session_state.negative_percent > 50:
        st.error(f"## 🔥 CRISIS DETECTED! Negative: {st.session_state.negative_percent}%")
    else:
        st.success(f"## ✅ No Crisis. Negative: {st.session_state.negative_percent}%")

# **Phân tích tác nhân gây ảnh hưởng (Influencer Network Analysis)**
st.markdown("---")
st.markdown("## 🏆 Influencer Network Analysis")

if st.button("Analyze Influencer Network"):
    with st.spinner("Analyzing influencer impact..."):
        try:
            # **Lấy danh sách tweet/bài đăng từ hệ thống giám sát MXH**
            tweets = [
                {"user": "UserA", "text": "Ủng hộ thương hiệu này, rất tốt!", "mentions": ["UserB", "UserC"]},
                {"user": "UserD", "text": "Tôi phản đối chiến dịch này, không rõ ràng!", "mentions": ["UserE"]},
                {"user": "UserF", "text": "Mọi người nghĩ sao về scandal này?", "mentions": ["UserA", "UserD"]},
                {"user": "UserC", "text": "Mạng xã hội đang bùng nổ tranh luận!", "mentions": []},
                {"user": "UserE", "text": "Câu chuyện này rất nhạy cảm, cần theo dõi thêm.", "mentions": []},
            ]

            # **Gọi hàm phân tích mạng lưới influencer**
            result = find_top_influencers(tweets)

            # **Xử lý số lượng giá trị trả về để tránh lỗi unpacking**
            if len(result) == 3:
                top_influencers, top_opposers, G = result  # ✅ Nếu có đủ 3 giá trị
            elif len(result) == 2:
                top_influencers, top_opposers = result
                G = nx.DiGraph()  # ✅ Tạo một đồ thị rỗng nếu không có dữ liệu

            # **Lưu vào session_state**
            st.session_state.top_influencers = top_influencers
            st.session_state.top_opposers = top_opposers
            st.session_state.graph = G

        except Exception as e:
            st.error(f"⚠️ Error analyzing influencers: {e}")
            st.session_state.graph = nx.DiGraph()  # ✅ Đảm bảo G tồn tại để tránh lỗi NameError

# **Hiển thị danh sách influencers với chi tiết**
if "top_influencers" in st.session_state and st.session_state.top_influencers:
    st.markdown("### 🌟 Top 5 Influencers:")
    for user in st.session_state.top_influencers:
        with st.expander(f"🔹 {user['user']} (Ảnh hưởng: {user['pagerank']:.4f}, Tweets: {user['tweets']})"):
            st.write(f"- **User:** {user['user']}")
            st.write(f"- **Mức độ ảnh hưởng (PageRank):** {user['pagerank']:.4f}")
            st.write(f"- **Số bài đăng:** {user['tweets']}")
            st.write(f"- **Mentions:** {', '.join(user.get('mentions', [])) if user.get('mentions') else 'Không có'}")

if "top_opposers" in st.session_state and st.session_state.top_opposers:
    st.markdown("### 🔥 Top 5 Opposers:")
    for user in st.session_state.top_opposers:
        with st.expander(f"🔻 {user['user']} (Ảnh hưởng: {user['pagerank']:.4f}, Tweets: {user['tweets']})"):
            st.write(f"- **User:** {user['user']}")
            st.write(f"- **Mức độ ảnh hưởng (PageRank):** {user['pagerank']:.4f}")
            st.write(f"- **Số bài đăng:** {user['tweets']}")
            st.write(f"- **Mentions:** {', '.join(user.get('mentions', [])) if user.get('mentions') else 'Không có'}")

# **Hiển thị biểu đồ mạng xã hội**
if "graph" in st.session_state and st.session_state.graph is not None:
    display_influencer_graph(st.session_state.graph)
else:
    st.warning("⚠️ No influencer network available to display.")
