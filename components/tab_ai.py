import streamlit as st
from ai_engine import query_gemini
from components.utils import make_summary  # or generate_df_context from your utils

def render_ai_tab():
    st.subheader("💬 GenAI Dataset Assistant")
    st.caption("Powered by Gemini | Created by **Ansh Thakar**")

    # 1. Grab whichever dataset is active
    active_df = st.session_state.get("processed_df") if st.session_state.get("processed_df") is not None else st.session_state.get("raw_df")

    if active_df is None:
        st.info("⚠️ Please upload a CSV dataset in the sidebar to activate the AI Assistant.")
        return

    # Prepare context summary
    df_summary_text = str(make_summary(active_df))

    st.markdown("### ⚡ Quick AI Action Reports")
    c1, c2, c3 = st.columns(3)
    prompt_to_run = None

    with c1:
        if st.button("📌 Explain Data Simply", use_container_width=True):
            prompt_to_run = "Explain what this dataset is about in 3 simple bullet points for a non-technical manager."

    with c2:
        if st.button("🧼 What Should I Clean?", use_container_width=True):
            prompt_to_run = "Tell me the top 3 data quality issues I need to fix, explained simply without jargon."

    with c3:
        if st.button("🎯 Key Drivers & Insights", use_container_width=True):
            prompt_to_run = "In simple terms, what are the top 3 factors that seem most important in this data and why?"

    st.divider()

    # 2. Input Form (Guaranteed to display)
    with st.form(key="ai_question_form"):
        user_question = st.text_input(
            "💬 Ask any question about your data:",
            placeholder="e.g., What are the main trends or outliers in this dataset?"
        )
        submit_button = st.form_submit_button("Submit Question")

        if submit_button and user_question.strip():
            prompt_to_run = user_question

    # Clear Report Button
    if st.button("🗑️ Clear AI Answer"):
        st.session_state.ai_response = ""
        st.rerun()

    # 3. Handle API Query
    if prompt_to_run:
        with st.spinner("🤖 DataPilot AI is analyzing your dataset..."):
            st.session_state.ai_response = query_gemini(prompt_to_run, df_summary_text)
        st.rerun()

    # 4. Display AI Response
    if st.session_state.get("ai_response"):
        st.divider()
        st.markdown("### 🤖 DataPilot AI Report")
        st.markdown(st.session_state.ai_response)