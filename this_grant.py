# Final Capstone Project (Grant Matcher for CT RISE)

import os, time, requests
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import openai

# ───────── CONFIG ─────────
PULL_N     = 40
SHOW_TOP   = 15
SEARCH_Q   = "education AND (high school OR college readiness OR youth)"
API_URL    = "https://www.grants.gov/grantsws/rest/opportunities/search"
EMBED_MOD  = "text-embedding-ada-002"
RETRIES    = 4
DELAY      = 2

MISSION = (
    "The Connecticut RISE Network empowers public high schools with data-driven strategies "
    "and personalized support to improve student outcomes and promote postsecondary success, "
    "especially for Black, Latinx, and low-income youth."
)

# ────── OPENAI KEY ────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ────── FUNCTIONS ─────────
def embed(txt: str):
    for a in range(RETRIES):
        try:
            return openai.Embedding.create(input=txt, model=EMBED_MOD)["data"][0]["embedding"]
        except openai.error.RateLimitError:
            time.sleep(DELAY * (a+1))
    st.error("OpenAI rate-limited; try later."); st.stop()

def fetch_grants(n=PULL_N) -> pd.DataFrame:
    """POST search request to Grants.gov"""
    payload = {
        "keywords": SEARCH_Q,
        "oppStatuses": ["posted", "forecasted"],
        "sortField": "openDate",
        "sortOrder": "desc",
        "pageSize": n,
        "startRecordNum": 0,
    }
    r = requests.post(API_URL, json=payload, timeout=30,
                      headers={"Content-Type": "application/json",
                               "Accept": "application/json"})
    r.raise_for_status()
    hits = r.json().get("oppHits", [])
    rows=[]
    for h in hits:
        rows.append({
            "title":    h.get("oppTitle","N/A"),
            "sponsor":  h.get("agency","N/A"),
            "amount":   h.get("awardCeiling","N/A"),
            "summary":  h.get("synopsis","")[:1800],
            "deadline": h.get("closeDate","N/A"),
            "url":      h.get("oppLink",""),
        })
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False, ttl=43200)
def rank_table(df_raw: pd.DataFrame):
    if df_raw.empty: return pd.DataFrame()
    mvec = embed(MISSION)
    df_raw["%Match"] = (df_raw.summary.apply(lambda s:
        float(cosine_similarity([embed(s)],[mvec])[0][0])*100)).round(1)
    top = df_raw.sort_values("%Match", ascending=False).head(SHOW_TOP).reset_index(drop=True)
    top.index = top.index + 1
    top.insert(0, "Rank", top.index)
    return top

# ────── UI ────────────────
st.title("Final Capstone Project (Grant Matcher for CT RISE)")
st.write("> **Mission:**", MISSION)

if st.button("🔄 Fetch & Rank 15 Grants", type="primary"):
    with st.spinner("Contacting Grants.gov and OpenAI… please wait ≈1 min"):
        raw = fetch_grants()
        ranked = rank_table(raw)
        st.session_state["tbl"] = ranked
        st.success("Done!")

if "tbl" in st.session_state and not st.session_state["tbl"].empty:
    st.dataframe(
        st.session_state["tbl"][["Rank","title","sponsor","amount",
                                 "deadline","%Match","url","summary"]],
        use_container_width=True)
elif "tbl" in st.session_state:
    st.info("No data returned — try again in a few minutes.")
else:
    st.caption("Ready — click the button to generate the list.")
