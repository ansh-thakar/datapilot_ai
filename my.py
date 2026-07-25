# my.py
import streamlit as st
import pandas as pd

# Import tab modules from components folder
from components.tab_eda import render_eda_tab
from components.tab_clean import render_clean_tab
from components.tab_ml import render_ml_tab
from components.tab_ai import render_ai_tab

# Page Config
st.set_page_config(page_title="DataPilot AI by Ansh Thakar", layout="wide")
st.title("DataPilot AI 📊")
st.caption("🚀 Built by **Ansh Thakar** | Your clean pipeline for EDA, Preprocessing, ML, and AI Insights")
if st.button("🔄 Reset Processed Data"):
        if st.session_state.raw_df is not None:
            st.session_state.processed_df = st.session_state.raw_df.copy()
            st.session_state.outlier_report = None
            st.session_state.last_action = "Reset to original dataset."
            st.rerun()

# 1. Initialize Session States
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "processed_df" not in st.session_state:
    st.session_state.processed_df = None
if "ai_response" not in st.session_state:
    st.session_state.ai_response = ""
if "outlier_report" not in st.session_state:
    st.session_state.outlier_report = None

# 2. Sidebar Upload Logic
with st.sidebar:
    st.header("📁 Data Source")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        st.session_state.raw_df = pd.read_csv(uploaded_file)
        if st.session_state.processed_df is None:
            st.session_state.processed_df = st.session_state.raw_df.copy()

# 3. Main Navigation Tabs
tab_eda, tab_clean, tab_ml, tab_ai = st.tabs([
    "📈 Exploratory Data Analysis",
    "🧼 Preprocessing Pipeline",
    "🤖 ML Model Studio",
    "💬 GenAI Chatbot"
])


# 4. Render Tabs cleanly
with tab_eda:
    render_eda_tab()

with tab_clean:
    render_clean_tab()

with tab_ml:
    render_ml_tab()

with tab_ai:
    render_ai_tab()