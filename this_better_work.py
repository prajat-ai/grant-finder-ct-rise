# Final Capstone Project (Grant Matcher for CT RISE)
# --------------------------------------------------
# Uses the public Grants.gov v1 /api/search2 endpoint (POST, no auth).

import os, time, json, requests, re
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import openai

# ─────────── CONFIG ───────────
MISSION = (
    "The Connecticut RISE Network empowers public high schools with data-driven strategies "
    "and personalized support to improve student outcomes and promote postsecondary success, "
    "especially for Black, Latinx, and low-income youth."
)
ROWS      = 40      # how many grants to request
SHOW_TOP  = 15
API_URL   = "https://api.grants.gov/v1/api/search2"
EMB_MODEL = "text-embedding-ada-002"
RETRIES   = 4
PAUSE     = 2       # seconds between RateLimit retries

# ─────────── OPENAI KEY ─────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ─────────── HELPERS ────────────
def embed(text: str):
    for a in range(RETRIES):
        try:
            return openai.Embedding.create(input=text, model=EMB_MODEL)["data"][0]["embedding"]
        except openai.error.RateLimitError:
            time.sleep(PAUSE * (a + 1))
    st.error("Rate-limited by OpenAI. Try again later."); st.stop()

def fetch_grants(rows=ROWS) -> pd.DataFrame:
    """POST to search2; no API key needed."""
    payload = {
        "rows": rows,
        "keyword": "education high school youth \"college readiness\"",
        "oppStatuses": "forecasted|posted"
    }
    r = requests.post(API_URL, json=payload, timeout=40,
                      headers={"Content-Type": "application/json"})
    r.raise_for_status()
    data = r.json().get("data", {})
    hits = data.get("oppHits", [])
    rows_out = []
    for h in hits:
        rows_out.append(
            {
                "title":    h.get("title", "N/A"),
                "sponsor":  h.get("agencyName", "N/A"),
                "amount":   h.get("awardCeiling", "N/A"),
                "deadline": h.get("closeDate", "N/A"),
                "summary":  h.get("synopsis", "")[:1600],
                "url":      f"https://www.grants.gov/search-results-detail/{h.get('id')}",
            }
        )
    return pd.DataFrame(rows_out)

@st.cache_data(show_spinner=False, ttl=43200)   # 12 h
def rank_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame()
    mvec = embed(MISSION)
    df_raw["%Match"] = (
        df_raw.summary.apply(lambda s:
            cosine_similarity([embed(s)], [mvec])[0][0] * 100).round(1)
    )
    top = (
        df_raw.sort_values("%Match", ascending=False)
        .head(SHOW_TOP)
        .reset_index(drop=True)
    )
    top.index = top.index + 1
    top.insert(0, "Rank", top.index)
    return top

# ─────────── UI ────────────────
st.title("Final Capstone Project (Grant Matcher for CT RISE)")
st.write("> **Mission:**", MISSION)

if st.button("🔄 Fetch & Rank 15 Grants", type="primary"):
    with st.spinner("Retrieving opportunities & computing similarity …"):
        try:
            raw_df = fetch_grants()
            st.session_state["tbl"] = rank_table(raw_df)
            st.success("Done!")
        except requests.HTTPError as e:
            st.error(f"Grants.gov API error ({e.response.status_code}). Try again later.")

if "tbl" in st.session_state and not st.session_state["tbl"].empty:
    st.dataframe(
        st.session_state["tbl"][[
            "Rank", "title", "sponsor", "amount", "deadline", "%Match", "url", "summary"
        ]],
        use_container_width=True
    )
elif "tbl" in st.session_state:
    st.info("Grants.gov returned no data — please retry.")
else:
    st.caption("Click the button to generate the ranked list.")
