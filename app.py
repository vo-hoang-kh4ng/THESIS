# Enhanced app.py with better UI, caching, and visualization options
import streamlit as st
from my_utils import (
    run_social_media_monitoring, parse_negative_percentage,
    find_top_influencers, display_influencer_graph

)
import time
import os
import json
import networkx as nx
import pandas as pd
from datetime import datetime, timedelta

# Environment setup
os.environ["OTEL_SDK_DISABLED"] = "true"

# App configuration
st.set_page_config(
    page_title="Crisis Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS to improve appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4285F4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #5C5C5C;
        margin-bottom: 1rem;
    }
    .crisis-alert {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #FFE0E0;
        border: 1px solid #FF0000;
        margin: 1rem 0;
    }
    .normal-status {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #E0FFE0;
        border: 1px solid #00FF00;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #F0F2F6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown("<h1 class='main-header'>📊 Real-Time Crisis Sentiment Analysis Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Monitor social media sentiment in real-time, detect potential crises, and identify key influencers.</p>", unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.markdown("## Configuration")
    
    # Brand input
    brand = st.text_input("🔍 Brand or Influencer Name:", placeholder="e.g., Tesla, Apple, etc.")
    
    # Advanced settings as expandable section
    with st.expander("Advanced Settings"):
        crisis_threshold = st.slider("Crisis Threshold (%)", 30, 70, 50, 
                                   help="Percentage of negative sentiment that triggers a crisis alert")
        platforms = st.multiselect("Platforms to Monitor", 
                                  ["Twitter", "YouTube", "News", "Web"], 
                                  default=["Twitter", "YouTube", "News"])
        time_window = st.selectbox("Time Window", 
                                  ["Last 24 hours", "Last 7 days", "Last 30 days"], 
                                  index=0)
    
    # Analysis options
    st.markdown("## Analysis Options")
    include_influencer_analysis = st.checkbox("Include Influencer Network Analysis", value=True)
    include_response_suggestions = st.checkbox("Generate Response Suggestions", value=True)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
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
if "sentiment_trend" not in st.session_state:
    st.session_state.sentiment_trend = None
if "word_cloud" not in st.session_state:
    st.session_state.word_cloud = None
if "response_templates" not in st.session_state:
    st.session_state.response_templates = None
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = None
if "flow" not in st.session_state:
    st.session_state.flow = None

# Control buttons
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

with col1:
    run_analysis = st.button("⚡ Run Analysis", type="primary", disabled=not brand)

with col2:
    reset_analysis = st.button("🔄 Reset Analysis")

with col3:
    if st.session_state.last_refresh and st.session_state.result:
        auto_refresh = st.button("🔄 Auto-Refresh (5m)")
    else:
        auto_refresh = st.button("🔄 Auto-Refresh (5m)", disabled=True)

with col4:
    export_report = st.button("📥 Export Report", disabled=not st.session_state.result)

# Handle reset
if reset_analysis:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.history = []
    st.success("✅ Session state reset. Ready for a new analysis!")
    time.sleep(1)
    st.rerun()

# Auto-refresh functionality
if auto_refresh and st.session_state.last_refresh:
    if datetime.now() - st.session_state.last_refresh > timedelta(minutes=5):
        run_analysis = True
    else:
        st.info(f"Last refresh was {(datetime.now() - st.session_state.last_refresh).seconds // 60} minutes ago. Auto-refresh is set to 5 minute intervals.")

# Run analysis
if run_analysis and brand:
    with st.status("Running sentiment analysis...", expanded=True) as status:
        st.write("Fetching data from social media platforms...")
        time.sleep(1)
        
        st.write("Analyzing sentiment and detecting potential crises...")
        try:
            # Run analysis with configured parameters
            custom_params = {
                "crisis_threshold": crisis_threshold,
                "platforms": platforms,
                "time_window": time_window
            }
            
            result, crisis_detected, negative_percent = run_social_media_monitoring(brand)
            
            # Update session state
            st.session_state.result = result
            st.session_state.crisis_detected = crisis_detected
            st.session_state.negative_percent = negative_percent
            st.session_state.show_report = True
            st.session_state.last_refresh = datetime.now()
            
            # Add to history
            historical_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "brand": brand,
                "negative_percent": negative_percent,
                "crisis_detected": crisis_detected
            }
            st.session_state.history.append(historical_entry)
            
            # Generate additional insights
            if include_influencer_analysis:
                st.write("Analyzing influencer networks...")
                tweets = run_social_media_monitoring(brand)
                if tweets:
                    top_influencers, top_opposers, G = find_top_influencers(tweets)
                    st.session_state.top_influencers = top_influencers
                    st.session_state.top_opposers = top_opposers
                    st.session_state.graph = G
                    
            
            # Generate word cloud
            
     
            status.update(label="Analysis completed successfully!", state="complete")
            
        except Exception as e:
            status.update(label=f"⚠️ Error during analysis: {e}", state="error")
            st.error(f"Failed to complete analysis: {str(e)}")
if st.session_state.flow:
    flow = st.session_state.flow
    
    # Display summary metrics
    st.markdown("## 📊 Sentiment Summary")
    
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        st.metric("Negative Sentiment", f"{flow.state.negative_percent}%", 
                delta=f"{flow.state.negative_percent - 50}" if flow.state.negative_percent > 0 else None,
                delta_color="inverse")
    
    with metric_cols[1]:
        positive_percent = 100 - flow.state.negative_percent - 20  # Mock neutral at 20%
        st.metric("Positive Sentiment", f"{positive_percent}%")
    
    with metric_cols[2]:
        neutral_percent = 100 - flow.state.negative_percent - positive_percent
        st.metric("Neutral Sentiment", f"{neutral_percent}%")
    
    with metric_cols[3]:
        if st.session_state.last_refresh:
            st.metric("Last Updated", st.session_state.last_refresh.strftime("%H:%M:%S"))
    
    # Crisis Alert
    if flow.state.crisis_detected:
        st.markdown(f"""
        <div class="crisis-alert">
            <h3>🚨 CRISIS ALERT ��</h3>
            <p>Negative sentiment has reached {flow.state.negative_percent}%, exceeding the crisis threshold of {crisis_threshold}%.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="normal-status">
            <h3>✅ Normal Status</h3>
            <p>Current negative sentiment is {flow.state.negative_percent}%, below the crisis threshold of {crisis_threshold}%.</p>
        </div>
        """, unsafe_allow_html=True)

    # Display influencer network analysis
    if flow.state.top_influencers and include_influencer_analysis:
        st.markdown("## 🏆 Influencer Network Analysis")
        display_influencer_graph(flow.state.graph, flow.state.top_influencers, flow.state.top_opposers)

    # Display report
    if flow.state.report_data:
        st.markdown("## 📝 Analysis Report")
        st.markdown(flow.state.report_data)
# Display summary metrics
if st.session_state.result:
    st.markdown("## 📊 Sentiment Summary")
    
    # Create metrics in columns
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        st.metric("Negative Sentiment", f"{st.session_state.negative_percent}%", 
                delta=f"{st.session_state.negative_percent - 50}" if st.session_state.history and len(st.session_state.history) > 1 else None,
                delta_color="inverse")
    
    with metric_cols[1]:
        positive_percent = 100 - st.session_state.negative_percent - 20  # Mock neutral at 20%
        st.metric("Positive Sentiment", f"{positive_percent}%")
    
    with metric_cols[2]:
        neutral_percent = 100 - st.session_state.negative_percent - positive_percent
        st.metric("Neutral Sentiment", f"{neutral_percent}%")
    
    with metric_cols[3]:
        if st.session_state.last_refresh:
            st.metric("Last Updated", st.session_state.last_refresh.strftime("%H:%M:%S"))
    
    # Crisis Alert
    if st.session_state.crisis_detected:
        st.markdown(f"""
        <div class="crisis-alert">
            <h3>🚨 CRISIS ALERT 🚨</h3>
            <p>Negative sentiment has reached {st.session_state.negative_percent}%, exceeding the crisis threshold of {crisis_threshold}%.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="normal-status">
            <h3>✅ Normal Status</h3>
            <p>Current negative sentiment is {st.session_state.negative_percent}%, below the crisis threshold of {crisis_threshold}%.</p>
        </div>
        """, unsafe_allow_html=True)

# Display sentiment trend
if st.session_state.sentiment_trend is not None:
    st.markdown("## 📈 Sentiment Trend Analysis")
    st.plotly_chart(st.session_state.sentiment_trend, use_container_width=True)

# Display influencer network analysis
if st.session_state.top_influencers and include_influencer_analysis:
    st.markdown("## 🏆 Influencer Network Analysis")
    
    influencer_cols = st.columns(2)
    
    with influencer_cols[0]:
        st.markdown("### 🌟 Top Influencers:")
        for i, (influencer, score) in enumerate(st.session_state.top_influencers):
            st.markdown(f"{i+1}. **{influencer}** - Influence Score: {score}")
    
    with influencer_cols[1]:
        st.markdown("### 🔥 Top Opposers:")
        for i, (opposer, score) in enumerate(st.session_state.top_opposers):
            st.markdown(f"{i+1}. **{opposer}** - Opposition Score: {score}")
    
    # Display network graph
    if st.session_state.graph is not None:
        st.markdown("### 🕸️ Network Visualization")
        display_influencer_graph(st.session_state.graph)

# Display word cloud
if st.session_state.word_cloud is not None:
    st.markdown("## 🔤 Key Topics Word Cloud")
    st.pyplot(st.session_state.word_cloud)

# Display response templates if crisis detected
if st.session_state.response_templates and st.session_state.crisis_detected:
    st.markdown("## 🛠️ Crisis Response Suggestions")
    
    for i, template in enumerate(st.session_state.response_templates):
        with st.expander(f"Response Template {i+1}: {template['title']}"):
            st.markdown(template['content'])
            st.download_button(
                label="Download Template",
                data=template['content'],
                file_name=f"{brand.lower()}_crisis_response_{i+1}.md",
                mime="text/markdown"
            )

# Display full report
if st.session_state.show_report and os.path.exists("final_report.md"):
    with st.expander("📄 View Full Report"):
        with open("final_report.md", "r", encoding="utf-8") as file:
            report_content = file.read()
        st.markdown(report_content, unsafe_allow_html=True)
        
        # Add export options
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            st.download_button(
                label="Download Report as Markdown",
                data=report_content,
                file_name=f"{brand.lower()}_sentiment_report.md",
                mime="text/markdown"
            )
        with export_col2:
            st.download_button(
                label="Download Report as PDF",
                data=report_content,  # In production, convert to PDF
                file_name=f"{brand.lower()}_sentiment_report.pdf",
                mime="application/pdf"
            )

# Add footer
st.markdown("---")
st.markdown("### 📊 Real-Time Crisis Sentiment Analysis Tool")
st.markdown("Powered by CrewAI and Streamlit | © 2025")