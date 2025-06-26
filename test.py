# Final Capstone Project (Grant Matcher for CT RISE) — Candid Grants API

import os, time, requests, urllib.parse
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import openai

# ── CONFIG ──────────────────────────────────────────
MISSION = (
    "The Connecticut RISE Network empowers public high schools with data-driven "
    "strategies and personalized support to improve student outcomes and promote "
    "postsecondary success, especially for Black, Latinx, and low-income youth."
)
CANDID_API   = "https://api.candid.org/grants/v1/transactions"   # doc endpoint:contentReference[oaicite:3]{index=3}
KEYWORDS     = "education high school youth \"college readiness\""
ROWS         = 40            # records to request
SHOW_TOP     = 15            # rows to display
EMBED_MODEL  = "text-embedding-ada-002"
RETRIES      = 4
SLEEP        = 2             # sec between OpenAI retries

# ── KEYS ────────────────────────────────────────────
load_dotenv()
openai.api_key    = os.getenv("OPENAI_API_KEY")
CANDID_KEY        = os.getenv("CANDID_API_KEY")     # must be set

# ── HELPERS ─────────────────────────────────────────
def embed(text: str):
    """Get OpenAI embedding with simple back-off."""
    for a in range(RETRIES):
        try:
            return openai.Embedding.create(input=text, model=EMBED_MODEL)["data"][0]["embedding"]
        except openai.error.RateLimitError:
            time.sleep(SLEEP * (a + 1))
    st.error("OpenAI rate-limit—try again later."); st.stop()

def fetch_grants(max_rows=ROWS) -> pd.DataFrame:
    """Query Candid Grants /transactions (verified data)."""
    params = {
        "keyword": KEYWORDS,
        "rows": max_rows,
        "sort_by": "award_amount",
        "sort_order": "desc",
    }
    headers = {"X-API-Key": CANDID_KEY, "Accept": "application/json"}  # auth header:contentReference[oaicite:4]{index=4}
    url = f"{CANDID_API}?{urllib.parse.urlencode(params)}"
    r = requests.get(url, headers=headers, timeout=40)
    r.raise_for_status()
    data = r.json().get("data", {}).get("transactions", [])           # field per docs :contentReference[oaicite:5]{index=5}
    rows = []
    for g in data:
        rows.append(
            {
                "title":    g.get("title", "N/A"),
                "sponsor":  g.get("funder_name", "N/A"),
                "amount":   g.get("amount", "N/A"),
                "deadline": g.get("year", "N/A"),       # Candid has year, not close date
                "summary":  g.get("description", "")[:1600],
                "url":      g.get("url", "https://candid.org"),        # if missing
            }
        )
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False, ttl=43200)        # 12 h
def rank_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame()
    mvec = embed(MISSION)
    df_raw["%Match"] = (
        df_raw.summary.apply(lambda s: cosine_similarity([embed(s)], [mvec])[0][0] * 100)
    ).round(1)
    top = (
        df_raw.sort_values("%Match", ascending=False)
        .head(SHOW_TOP)
        .reset_index(drop=True)
    )
    top.index = top.index + 1
    top.insert(0, "Rank", top.index)
    return top

# ── UI ──────────────────────────────────────────────
st.title("Final Capstone Project (Grant Matcher for CT RISE)")
st.write("> **Mission:**", MISSION)

if st.button("🔄 Fetch & Rank 15 Grants", type="primary"):
    with st.spinner("Contacting Candid & OpenAI … this may take ~1 min"):
        try:
            raw = fetch_grants()
            st.session_state["tbl"] = rank_table(raw)
            st.success("Done!")
        except requests.HTTPError as e:
            st.error(f"Candid API error {e.response.status_code} — check key & retry.")

if "tbl" in st.session_state and not st.session_state["tbl"].empty:
    st.dataframe(
        st.session_state["tbl"][
            ["Rank", "title", "sponsor", "amount", "deadline", "%Match", "url", "summary"]
        ],
        use_container_width=True,
    )
elif "tbl" in st.session_state:
    st.info("No data returned — try a different keyword or later.")
else:
    st.caption("Press the button to generate your ranked list.")
