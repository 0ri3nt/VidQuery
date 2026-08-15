"""Optional Streamlit client for users who prefer it to the built-in UI."""

import os

import requests
import streamlit as st

API_BASE = os.getenv("VIDQUERY_API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="VidQuery", page_icon="VQ", layout="wide")
st.title("VidQuery - Semantic Video Search")
st.caption("The primary browser UI is served directly by the API at its root URL.")

upload = st.file_uploader("Upload an MP4", type=["mp4"])
if upload and st.button("Upload and process"):
    response = requests.post(
        f"{API_BASE}/api/videos",
        files={"file": (upload.name, upload.getvalue(), "video/mp4")},
        timeout=300,
    )
    if response.ok:
        st.success(response.json())
    else:
        st.error(response.json().get("detail", response.text))

query = st.text_input(
    "Ask about your videos",
    placeholder="Find where someone discusses architecture near a whiteboard",
)
if st.button("Search") and query.strip():
    response = requests.post(
        f"{API_BASE}/api/search",
        json={"query": query, "video_ids": [], "limit": 10},
        timeout=120,
    )
    if not response.ok:
        st.error(response.json().get("detail", response.text))
    else:
        payload = response.json()
        st.json(payload["parsed_query"])
        if not payload["results"]:
            st.info("No matching indexed moments.")
        for result in payload["results"]:
            st.subheader(
                f"{result['video_title']} · {result['start_time']:.1f}s · {result['score']:.2f}"
            )
            st.write(result["match_reason"])
            st.write(result["transcript"] or "No speech in this segment.")
            st.video(
                f"{API_BASE}{result['stream_url']}",
                start_time=int(result["start_time"]),
            )
