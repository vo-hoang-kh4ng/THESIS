import streamlit as st
from my_utils import run_social_media_monitoring
import time

st.title("Social Media Sentiment Analysis Crew")
st.write("Welcome! This app uses an autonomous multi-agent system to analyze social media sentiment for a given brand or influencer.")

brand = st.text_input("Enter the brand or influencer name:", "FPT Telecom")

if st.button("Run Analysis"):
    with st.spinner("Running analysis..."):
        # Run the analysis (this might take a while)
        result = run_social_media_monitoring(brand)
        time.sleep(1)  # Optional: simulate processing delay
    if result:
        st.success("Analysis completed successfully!")
        st.subheader("Final Report")
        st.text(result)
    else:
        st.error("Failed to generate the report. Please try again later.")
