# ai_engine.py
import os
import requests
import streamlit as st

def query_gemini(prompt_text: str, dataset_context: str) -> str:
    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    except Exception:
        api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return "⚠️ **API Key Missing**: Please add `GEMINI_API_KEY` inside `.streamlit/secrets.toml`."

    candidate_models = [
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-pro"
    ]

    friendly_system_instruction = (
        "You are DataPilot AI, an easy-to-understand AI Data Assistant created by Ansh Thakar. "
        "When greeting the user, mention you were created by Ansh Thakar. "
        "Avoid heavy math jargon. Use plain English, simple analogies, bullet points, and emojis."
    )

    payload = {
        "system_instruction": {"parts": [{"text": friendly_system_instruction}]},
        "contents": [{"parts": [{"text": f"Dataset Context:\n{dataset_context}\n\nUser Question:\n{prompt_text}"}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2048}
    }

    last_error = ""
    for model_name in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        try:
            response = requests.post(url, json=payload, timeout=15)
            if response.status_code == 200:
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                last_error = f"{model_name} (HTTP {response.status_code}): {response.text}"
        except requests.exceptions.RequestException as e:
            last_error = f"{model_name}: {str(e)}"

    return f"❌ **Gemini Error**: Could not connect to Gemini models.\n\n`Details: {last_error}`"
