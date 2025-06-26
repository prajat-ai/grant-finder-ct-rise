# Final Capstone Project (Grant Matcher for CT RISE)
# --------------------------------------------------
# Uses the official Grants.gov Search API (GET) and OpenAI embeddings.

import os, time, requests, urllib.parse
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import openai

# ╭─ CONFIG ───────────────────────────────────────────
MISSION = (
    "The Connecticut RISE Network empowers public high schools with data-driven strategies "
    "and personalized support to improve student outcomes and promote postsecondary success, "
    "especially for Black, Latinx, and low-income youth."
)
QUERY       = "education \"high school\" OR \"college readiness\" OR youth"
PULL_N      = 40      # how many opportunities to request
SHOW_TOP    = 15      # how many rows to display
API_URL     = "https://www.grants.gov/grantsws/rest/opportunities/search"
EMBED_MODEL = "text-embedding-ada-002"
RETRIES     = 4
DELAY       = 2       # seconds between retries

# ╭─ KEYS ─────────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ╭─ HELPERS ──────────────────────────────────────────
def embed(text: str):
    """Get embedding with simple exponential back-off on RateLimit."""
    for a in range(RETRIES):
        try:
            r = openai.Embedding.create(input=text, model=EMBED_MODEL)
            return r["data"][0]["embedding"]
        except openai.error.RateLimitError:
            time.sleep(DELAY * (a + 1))
    st.error("OpenAI rate-limited – try again later."); st.stop()

def fetch_grants(max_rows=PULL_N) -> pd.DataFrame:
    """Call Grants.gov Search API (GET) per official spec."""
    params = {
        "keywords": QUERY,
        "oppStatuses": "posted,forecasted",
        "sortField": "openDate",
        "sortOrder": "desc",
        "pageSize": max_rows,
        "startRecordNum": 0,
    }
    url = f"{API_URL}?{urllib.parse.urlencode(params, safe=',')}"
    r = requests.get(url, timeout=30, headers={"Accept": "application/json"})
    r.raise_for_status()          # HTTP 4xx/5xx → exception
    hits = r.json().get("oppHits", [])
    rows = []
    for h in hits:
        rows.append(
            {
                "title":    h.get("oppTitle", "N/A"),
                "sponsor":  h.get("agency", "N/A"),
                "amount":   h.get("awardCeiling", "N/A"),
                "deadline": h.get("closeDate", "N/A"),
                "summary":  h.get("synopsis", "")[:1600],
                "url":      h.get("oppLink", ""),
            }
        )
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False, ttl=43200)   # 12 h cache
def rank_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Compute %Match and return Top 15 ranked 1-15."""
    if df_raw.empty:
        return pd.DataFrame()
    mission_vec = embed(MISSION)
    df_raw["%Match"] = (
        df_raw.summary.apply(lambda s: cosine_similarity([embed(s)], [mission_vec])[0][0] * 100)
        .round(1)
    )
    top = (
        df_raw.sort_values("%Match", ascending=False)
        .head(SHOW_TOP)
        .reset_index(drop=True)
    )
    top.index = top.index + 1        # make Rank start at 1
    top.insert(0, "Rank", top.index)
    return top

# ╭─ UI ───────────────────────────────────────────────
st.title("Final Capstone Project (Grant Matcher for CT RISE)")
st.write("> **Mission:**", MISSION)

if st.button("🔄 Fetch & Rank 15 Grants", type="primary"):
    with st.spinner("Querying Grants.gov and computing similarity …"):
        try:
            raw = fetch_grants()
            st.session_state["tbl"] = rank_table(raw)
            st.success("Done!")
        except requests.HTTPError as e:
            st.error(f"Grants.gov error {e.response.status_code}. Please retry later.")

if "tbl" in st.session_state and not st.session_state["tbl"].empty:
    st.dataframe(
        st.session_state["tbl"][
            ["Rank", "title", "sponsor", "amount", "deadline", "%Match", "url", "summary"]
        ],
        use_container_width=True
    )
elif "tbl" in st.session_state:
    st.info("No data returned – try again later.")
else:
    st.caption("Ready – click the button to generate your ranked list.")
