# Final Capstone Project (Grant Matcher for CT RISE)
# -- Streamlit dashboard pulling verified data from Grants.gov and ranking by mission match.

import os, time, requests
from urllib.parse import urlencode
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import openai

# ───────── CONFIG ─────────
PULL_N       = 40      # grants to fetch from API
SHOW_TOP     = 15      # show this many
SEARCH_QUERY = "education high school youth college readiness"
CHAT_MODEL   = "text-embedding-ada-002"
MAX_RETRIES  = 4
DELAY        = 2       # seconds between retries
GRANTS_API   = "https://www.grants.gov/grantsws/rest/opportunities/search"

# Mission statement
MISSION = (
    "The Connecticut RISE Network empowers public high schools with data-driven strategies "
    "and personalized support to improve student outcomes and promote postsecondary success, "
    "especially for Black, Latinx, and low-income youth."
)

# ───────── API KEY ─────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ───────── FUNCTIONS ───────
def fetch_grants(n=PULL_N) -> pd.DataFrame:
    """Query Grants.gov Search API and return DataFrame."""
    params = {
        "keywords": SEARCH_QUERY,
        "oppStatuses": "posted,forecasted",
        "sortField": "openDate",
        "sortOrder": "desc",
        "pageSize": n,
        "startRecordNum": 0,
    }
    url = f"{GRANTS_API}?{urlencode(params)}"
    r = requests.get(url, timeout=30, headers={"Accept": "application/json"})
    r.raise_for_status()
    hits = r.json().get("oppHits", [])
    rows = []
    for h in hits:
        rows.append(
            {
                "title": h.get("oppTitle", "N/A"),
                "sponsor": h.get("agency", "N/A"),
                "amount": h.get("awardCeiling", "N/A"),
                "summary": h.get("synopsis", "")[:1800],
                "deadline": h.get("closeDate", "N/A"),
                "url": h.get("oppLink", ""),
            }
        )
    return pd.DataFrame(rows)

def embed(text: str):
    """Get embedding with basic retry."""
    for a in range(MAX_RETRIES):
        try:
            resp = openai.Embedding.create(input=text, model=CHAT_MODEL)
            return resp["data"][0]["embedding"]
        except openai.error.RateLimitError:
            time.sleep(DELAY * (a + 1))
    st.error("OpenAI rate-limited; please try again later.")
    st.stop()

@st.cache_data(show_spinner=False, ttl=43200)
def rank_grants(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame()
    mission_vec = embed(MISSION)
    sims = []
    for txt in df_raw.summary:
        sims.append(float(cosine_similarity([embed(txt)], [mission_vec])[0][0]))
    df_raw["%Match"] = (pd.Series(sims) * 100).round(1)
    ranked = df_raw.sort_values("%Match", ascending=False).head(SHOW_TOP).reset_index(drop=True)
    ranked.index = ranked.index + 1   # make Rank start at 1
    ranked.insert(0, "Rank", ranked.index)
    return ranked

# ───────── UI ─────────
st.title("Final Capstone Project (Grant Matcher for CT RISE)")
st.write("> **Mission:**", MISSION)

if st.button("🔄 Fetch & Rank 15 Grants", type="primary"):
    with st.spinner("Contacting Grants.gov and generating embeddings…"):
        raw_df  = fetch_grants()
        table   = rank_grants(raw_df)
        st.session_state["tbl"] = table
        st.success("Finished!")

if "tbl" in st.session_state and not st.session_state["tbl"].empty:
    st.dataframe(
        st.session_state["tbl"][
            ["Rank", "title", "sponsor", "amount", "deadline", "%Match", "url", "summary"]
        ],
        use_container_width=True,
    )
elif "tbl" in st.session_state:
    st.info("No results from Grants.gov. Try again in a minute.")
else:
    st.caption("Press the button to generate the ranked list.")
