import os

import requests
import streamlit as st


API_BASE = os.getenv("VIDQUERY_API_BASE", "http://127.0.0.1:8000")


st.set_page_config(page_title="VidQuery", page_icon="VQ", layout="wide")
st.title("VidQuery - Semantic Video Search")

query = st.text_input("Ask about your videos", placeholder="Find where the professor writes on the board")

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Search") and query.strip():
        try:
            response = requests.get(f"{API_BASE}/search", params={"q": query}, timeout=120)
            response.raise_for_status()
            payload = response.json()

            st.subheader("Generated Cypher")
            st.code(payload.get("cypher", ""), language="cypher")

            st.subheader("Results")
            results = payload.get("results", [])
            if not results:
                st.info("No results found.")
            else:
                for item in results:
                    st.write(
                        f"Video: {item.get('video_id')} | "
                        f"Timestamp: {item.get('timecode')} "
                        f"({item.get('timestamp')}s)"
                    )
        except Exception as exc:
            st.error(f"Search failed: {exc}")

with col2:
    st.subheader("Pipeline")
    if st.button("Run Ingestion Pipeline"):
        try:
            response = requests.post(f"{API_BASE}/upload", json={}, timeout=30)
            response.raise_for_status()
            st.success(response.json().get("message", "Pipeline started"))
        except Exception as exc:
            st.error(f"Upload failed: {exc}")
