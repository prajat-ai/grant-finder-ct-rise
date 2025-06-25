# app.py ──────────────────────────────────────────────────────────
"""
CT RISE Network – Smart Grant Finder
Pulls grants from Grants.gov, ranks by semantic similarity to CT RISE’s mission,
adds a GPT feasibility label, and displays everything in a Streamlit dashboard.
"""

import os, time, json, logging, requests, pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import openai

# ── CONFIG ───────────────────────────────────────────────────────
MAX_RESULTS   = 25        # grants pulled from API each refresh
TOP_N_GPT     = 10        # only run GPT on top-N matches
SLEEP_SECONDS = 2         # pause between OpenAI calls (rate-limit safety)
RETRIES       = 3         # OpenAI retry attempts

# ── KEYS ─────────────────────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ── CT RISE MISSION ──────────────────────────────────────────────
CT_RISE_MISSION = (
    "The Connecticut RISE Network empowers public high schools with "
    "data-driven strategies and personalized support to improve student outcomes "
    "and promote postsecondary success, especially for Black, Latinx, "
    "and low-income youth."
)

# ── OPENAI HELPERS ───────────────────────────────────────────────
def get_embedding(text: str, model: str = "text-embedding-ada-002") -> list[float]:
    for attempt in range(RETRIES):
        try:
            resp = openai.Embedding.create(input=text, model=model)
            return resp["data"][0]["embedding"]
        except openai.error.RateLimitError:
            time.sleep(5 * (attempt + 1))
    st.error("OpenAI rate-limit hit repeatedly.")
    return [0.0] * 1536

def gpt_brief(mission: str, grant_row: pd.Series) -> tuple[str, str]:
    prompt = (
        f"Nonprofit mission:\n{mission}\n\n"
        f"Grant title: {grant_row.title}\n"
        f"Grant description: {grant_row.summary}\n\n"
        'Respond ONLY in JSON like {"feasibility":"High","why":"<one sentence>"}'
    )
    for attempt in range(RETRIES):
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=60,
                temperature=0.3,
            )
            data = json.loads(resp.choices[0].message.content.strip())
            return data.get("feasibility", "Unknown"), data.get("why", "")
        except (openai.error.RateLimitError, json.JSONDecodeError):
            time.sleep(5 * (attempt + 1))
    return "Unknown", "Could not parse GPT response."

# ── GRANTS.GOV FETCHER ───────────────────────────────────────────
@st.cache_data(ttl=86400)
def fetch_grants(max_results: int = 25) -> pd.DataFrame:
    base = "https://www.grants.gov/grantsws/rest/opportunities/search"
    params = {
        "keywords": "education high school youth college success",
        "oppStatuses": "forecasted,posted",
        "sortField": "openDate",
        "sortOrder": "desc",
        "pageSize": max_results,
        "startRecordNum": 0,
    }
    try:
        r = requests.get(base, params=params, timeout=30, headers={"Accept": "application/json"})
        r.raise_for_status()
        hits = r.json().get("oppHits", [])
    except Exception as e:
        logging.warning(f"Grants.gov error: {e}")
        return pd.DataFrame()

    rows = []
    for h in hits:
        rows.append(
            {
                "title": h.get("oppTitle", "N/A"),
                "agency": h.get("agency", ""),
                "summary": h.get("synopsis", "")[:2000],
                "deadline": h.get("closeDate", "N/A"),
                "url": h.get("oppLink", ""),
            }
        )
    return pd.DataFrame(rows)

# ── RANK + GPT ANNOTATE ──────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def process_grants(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    mission_vec = get_embedding(CT_RISE_MISSION)
    sims = []
    for txt in df["summary"]:
        vec = get_embedding(txt)
        sims.append(float(cosine_similarity([vec], [mission_vec])[0][0]))
        time.sleep(SLEEP_SECONDS)
    df["similarity"] = sims

    df = df.sort_values("similarity", ascending=False).head(TOP_N_GPT).reset_index(drop=True)

    feasibilities, whys = [], []
    for _, row in df.iterrows():
        f, w = gpt_brief(CT_RISE_MISSION, row)
        feasibilities.append(f)
        whys.append(w)
        time.sleep(SLEEP_SECONDS)
    df["feasibility"] = feasibilities
    df["why_fit"]    = whys
    return df

# ── STREAMLIT UI ─────────────────────────────────────────────────
st.title("CT RISE Network — Smart Grant Finder")

st.markdown(
    "> **Mission**: The Connecticut RISE Network empowers public high schools with data-driven "
    "strategies and personalized support to improve student outcomes and promote post-secondary success, "
    "especially for Black, Latinx, and low-income youth."
)

if st.button("🔄 Refresh grants & rank"):
    with st.spinner("Contacting Grants.gov and OpenAI… this can take ~1 min"):
        raw_df   = fetch_grants(MAX_RESULTS)
        ranked   = process_grants(raw_df)
        st.session_state["grants"] = ranked
        st.success("Updated!")

if "grants" in st.session_state and not st.session_state["grants"].empty:
    st.subheader("Top matched grants")
    st.dataframe(
        st.session_state["grants"][["title", "similarity", "feasibility", "why_fit", "deadline", "url"]],
        use_container_width=True,
    )
elif "grants" in st.session_state:
    st.info("No grants returned from API — try again later.")
else:
    st.info("Click **Refresh grants & rank** to generate matches.")
