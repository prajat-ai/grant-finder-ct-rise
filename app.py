# app.py  ──────────────────────────────────────────────────────────
import os, time, requests, json
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import openai
import logging

# ── 0. Configure & load key ──────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Grant-search parameters
MAX_RESULTS   = 25        # how many grants to pull each refresh
SLEEP_SECONDS = 1         # pause between OpenAI calls
RETRIES       = 3         # retry attempts on RateLimit error

# ── 1. CT RISE profile ───────────────────────────────────────────
CT_RISE_MISSION = (
    "The Connecticut RISE Network empowers public high schools with "
    "data-driven strategies and personalized support to improve student outcomes "
    "and promote postsecondary success, especially for Black, Latinx, and low-income youth."
)

# ── 2. OpenAI helper functions ───────────────────────────────────
def get_embedding(text: str, model: str = "text-embedding-ada-002"):
    """Return embedding vector with retry + back-off."""
    for attempt in range(RETRIES):
        try:
            resp = openai.Embedding.create(input=text, model=model)
            return resp["data"][0]["embedding"]
        except openai.error.RateLimitError as e:
            wait = (attempt + 1) * 5
            logging.warning(f"Rate-limited, retrying in {wait}s …")
            time.sleep(wait)
    st.error("OpenAI rate-limit hit repeatedly. Try again later.")
    return [0.0] * 1536  # return dummy vector to keep code alive

def gpt_brief(mission: str, grant_row: pd.Series) -> tuple[str, str]:
    """Return feasibility (High/Medium/Low) + 1-sentence rationale."""
    prompt = (
        f"Nonprofit mission:\n{mission}\n\n"
        f"Grant title: {grant_row.title}\n"
        f"Grant description: {grant_row.summary}\n\n"
        "Answer ONLY with JSON like {\"feasibility\":\"High\",\"why\":\"<one sentence>\"}"
    )
    for attempt in range(RETRIES):
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # free-tier friendly; swap to gpt-4o if account allows
                messages=[{"role": "user", "content": prompt}],
                max_tokens=60,
                temperature=0.3,
            )
            # try to parse JSON safely
            raw = resp.choices[0].message.content.strip()
            data = json.loads(raw)
            return data.get("feasibility", "Unknown"), data.get("why", "")
        except (openai.error.RateLimitError, json.JSONDecodeError):
            wait = (attempt + 1) * 5
            time.sleep(wait)
    return "Unknown", "Could not parse GPT response."

# ── 3. Grants.gov fetcher (robust) ───────────────────────────────
@st.cache_data(ttl=86400)
def fetch_grants(max_results: int = 25) -> pd.DataFrame:
    """
    Returns a DataFrame of grant listings or an empty DF if the API
    fails. We catch JSON errors and log the raw response.
    """
    url = "https://www.grants.gov/grantsws/rest/opportunities/search"
    payload = {
        "keywords": "education college readiness high school",
        "oppStatuses": ["posted","forecasted"],   # documented field
        "eligibilityCodes": ["25"],               # 25 = nonprofits without 501c3 (broadest)
        "pageSize": max_results,
        "sortField": "openDate",
        "sortOrder": "desc"
    }
    try:
        r = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        r.raise_for_status()            # HTTP errors → exception
        data = r.json()
        hits = data.get("oppHits", [])
    except Exception as e:
        st.error("Grants.gov API problem - showing placeholder data.")
        st.caption(f"(debug info: {e})")
        return pd.DataFrame()           # empty → handled downstream

    records = []
    for hit in hits:
        records.append(
            {
                "title": hit.get("oppTitle", "N/A"),
                "agency": hit.get("agency", ""),
                "summary": hit.get("synopsis", "")[:2000],
                "deadline": hit.get("closeDate", "N/A"),
                "url": hit.get("oppLink", ""),
            }
        )
    return pd.DataFrame(records)

# ── 4. Rank & annotate grants ────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)  # 1-hour cache
def process_grants(df: pd.DataFrame) -> pd.DataFrame:
    # embed mission once
    mission_vec = get_embedding(CT_RISE_MISSION)
    # embed each grant + compute similarity
    vecs, sims = [], []
    for txt in df["summary"]:
        vec = get_embedding(txt)
        vecs.append(vec)
        sims.append(float(cosine_similarity([vec], [mission_vec])[0][0]))
        time.sleep(SLEEP_SECONDS)
    df["similarity"] = sims

    # pick top 15 by similarity
    df = df.sort_values("similarity", ascending=False).head(15).reset_index(drop=True)

    # GPT feasibility + rationale
    feas, whys = [], []
    for _, row in df.iterrows():
        f, w = gpt_brief(CT_RISE_MISSION, row)
        feas.append(f)
        whys.append(w)
        time.sleep(SLEEP_SECONDS)
    df["feasibility"] = feas
    df["why_fit"] = whys
    return df

# ── 5. Streamlit UI ──────────────────────────────────────────────
st.title("CT RISE Network — Smart Grant Finder")

st.markdown(
    """
    **Mission:**  
    > The Connecticut RISE Network empowers public high schools with data-driven strategies  
    > and personalized support to improve student outcomes and promote post-secondary success.
    """
)

if st.button("🔄 Refresh grants & rank"):
    with st.spinner("Contacting Grants.gov and OpenAI… please wait≈1 min"):
        raw_df   = fetch_grants(MAX_RESULTS)
        final_df = process_grants(raw_df)
        st.session_state["grants"] = final_df
        st.success("Done!")

if "grants" in st.session_state:
    st.subheader("Top grant matches for CT RISE")
    st.dataframe(
        st.session_state["grants"][
            ["title", "similarity", "feasibility", "why_fit", "deadline", "url"]
        ],
        use_container_width=True,
    )
else:
    st.info("Click the refresh button to fetch and rank grants.")
# ─────────────────────────────────────────────────────────────────
