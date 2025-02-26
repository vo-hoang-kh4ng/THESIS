import streamlit as st
from my_utils import run_social_media_monitoring, parse_negative_percentage
import time
import os
st.title("Real-Time Crisis Sentiment Analysis Crew")
st.write("This app monitors social media sentiment for a brand in real-time and detects potential crises.")

# Nhập thương hiệu
brand = st.text_input("Enter the brand or influencer name:")

# Session state để lưu trạng thái
if "result" not in st.session_state:
    st.session_state.result = None
if "crisis_detected" not in st.session_state:
    st.session_state.crisis_detected = False
if "negative_percent" not in st.session_state:
    st.session_state.negative_percent = 0
if "show_report" not in st.session_state:
    st.session_state.show_report = False

if st.button("Run Analysis"):
    with st.spinner("Running analysis..."):
        # Gọi hàm phân tích và lấy kết quả đầy đủ
        result, crisis_detected, negative_percent = run_social_media_monitoring(brand)

        # ✅ FIX: Đảm bảo dữ liệu hiển thị đúng
        st.session_state.result = result
        st.session_state.crisis_detected = crisis_detected
        st.session_state.negative_percent = negative_percent
        st.session_state.show_report = True  # ✅ Kích hoạt hiển thị báo cáo
        time.sleep(1)  # Giả lập độ trễ để giao diện cập nhật
# ✅ FIX: Kiểm tra nếu file `final_report.md` tồn tại
if st.session_state.show_report and os.path.exists("final_report.md"):
    with open("final_report.md", "r", encoding="utf-8") as file:
        report_content = file.read()
    
    st.markdown("## 📄 Full Report from `final_report.md`")
    st.markdown(report_content, unsafe_allow_html=True)  # ✅ HIỂN THỊ FILE MD
# ✅ FIX: Hiển thị toàn bộ báo cáo từ Re-ranking Agent
if st.session_state.result:
    st.markdown("## Full Crisis Report")
    st.markdown(st.session_state.result, unsafe_allow_html=True)  
    print(f"DEBUG: Displaying Negative Sentiment: {st.session_state.negative_percent}%")
    # ✅ FIX: Đảm bảo Crisis Status hiển thị đúng
    if st.session_state.negative_percent > 50:
        st.error(f"## 🔥 CRISIS DETECTED! Negative: {st.session_state.negative_percent}%")
    else:
        st.success(f"## ✅ No Crisis. Negative: {st.session_state.negative_percent}%")
