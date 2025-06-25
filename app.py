import os, requests, pandas as pd
import streamlit as st
from dotenv import load_dotenv
import openai
from sklearn.metrics.pairwise import cosine_similarity

# ---------- 1. Keys ----------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------- 2. CT RISE mission ----------
CT_RISE_MISSION = (
    "The Connecticut RISE Network empowers public high schools with "
    "data-driven strategies and personalized support to improve student outcomes "
    "and promote postsecondary success, especially for Black, Latinx, and low-income youth."
)

# ---------- 3. Helper: Get OpenAI embedding ----------
def get_embedding(text, model="text-embedding-ada-002"):
    resp = openai.Embedding.create(input=text, model=model)
    return resp["data"][0]["embedding"]

mission_vec = get_embedding(CT_RISE_MISSION)

# ---------- 4. Fetch grants (Grants.gov API) ----------
@st.cache_data(show_spinner=False)
def fetch_grants(max_results=25):
    url = "https://www.grants.gov/grantsws/rest/opportunities/search"
    payload = {
        "keywords": "education",
        "eligibilities": "Nonprofits",
        "opportunityStatus": "forecasted,posted",
        "startRecordNum": 0,
        "sortBy": "openDate|desc",
        "resultsPerPage": max_results,
    }
    r = requests.post(url, json=payload, timeout=20)
    data = r.json().get("oppHits", [])
    records = []
    for hit in data:
        records.append({
            "title": hit.get("oppTitle", "N/A"),
            "agency": hit.get("agency",""),
            "summary": hit.get("synopsis","")[:2000],  # truncate
            "deadline": hit.get("closeDate","N/A"),
            "url": hit.get("oppLink","")
        })
    return pd.DataFrame(records)

# ---------- 5. Rank by similarity ----------
def rank_grants(df):
    df["embedding"] = df["summary"].apply(get_embedding)
    sims = cosine_similarity(df["embedding"].tolist(), [mission_vec]).flatten()
    df["similarity"] = sims
    return df.sort_values("similarity", ascending=False).head(15).reset_index(drop=True)

# ---------- 6. GPT-4o analysis ----------
def gpt_feasibility(row):
    prompt = (
      f"Nonprofit mission: {CT_RISE_MISSION}\n\n"
      f"Grant: {row.title}\n{row.summary}\n\n"
      "Answer in JSON with keys feasibility (High/Medium/Low) and why (one sentence)."
    )
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # free-tier model name may differ; fallback to gpt-4o if allowed
        messages=[{"role":"user","content":prompt}],
        max_tokens=60,
        temperature=0.3,
    )
    return resp.choices[0].message.content

def apply_gpt(df):
    feasibilities, whys = [], []
    for _, row in df.iterrows():
        try:
            result = gpt_feasibility(row)
            # simple parse:
            feasibilities.append(result.split('"feasibility":"')[1].split('"')[0])
            whys.append(result.split('"why":"')[1].split('"')[0])
        except Exception:
            feasibilities.append("N/A")
            whys.append("Could not parse")
    df["feasibility"] = feasibilities
    df["why_fit"] = whys
    return df

# ---------- 7. Streamlit UI ----------
st.title("CT RISE – Smart Grant Finder")

if st.button("🔄 Refresh grant list"):
    st.info("Fetching and ranking grants… this takes ~1 min.")
    raw = fetch_grants()
    ranked = rank_grants(raw)
    final = apply_gpt(ranked)
    st.session_state["grants"] = final

if "grants" in st.session_state:
    st.subheader("Top matched grants")
    st.dataframe(
        st.session_state["grants"][["title","similarity","feasibility","why_fit","deadline","url"]],
        use_container_width=True
    )
else:
    st.write("Click **Refresh grant list** to generate matches.")
