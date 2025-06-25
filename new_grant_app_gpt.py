# grant_app_gpt.py  – CT RISE Smart Grant Finder v2-JSON
# Generates 15 education-equity grants with GPT-3.5-turbo-1106 (JSON mode),
# ranks them by similarity to the mission, labels feasibility, shows table.

import os, time, json, random
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import openai

# ───────────────────── CONFIG ─────────────────────
NUM_GRANTS   = 15          # tell GPT to return this many
TOP_N        = 8           # we annotate only the top-N matches
BASE_DELAY   = 2           # seconds before retry back-off
MAX_RETRIES  = 5
CHAT_MODEL   = "gpt-3.5-turbo-1106"    # supports JSON mode
EMBED_MODEL  = "text-embedding-ada-002"

# ───────────────────── KEY ─────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ─────────────────── MISSION ───────────────────────
MISSION = (
    "The Connecticut RISE Network empowers public high schools with "
    "data-driven strategies and personalized support to improve student outcomes "
    "and promote postsecondary success, especially for Black, Latinx, "
    "and low-income youth."
)

# ─────────────────── HELPERS ───────────────────────
def chat(messages, maxtok=800, **extra):
    """Chat w/ exponential back-off & JSON mode when requested."""
    for a in range(MAX_RETRIES):
        try:
            r = openai.ChatCompletion.create(
                model=CHAT_MODEL,
                messages=messages,
                max_tokens=maxtok,
                **extra,
            )
            return r.choices[0].message.content
        except openai.error.RateLimitError:
            time.sleep(BASE_DELAY * (2 ** a) + random.uniform(0, 1))
    st.error("OpenAI still rate-limited after retries."); st.stop()

def embed(txt: str):
    for a in range(MAX_RETRIES):
        try:
            v = openai.Embedding.create(input=txt, model=EMBED_MODEL)
            return v["data"][0]["embedding"]
        except openai.error.RateLimitError:
            time.sleep(BASE_DELAY * (2 ** a))
    return [0.0] * 1536

# ─── GPT: GENERATE GRANTS LIST (strict JSON mode) ────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def gpt_grants():
    sys = {"role": "system", "content": "You are a concise grants researcher."}
    usr = {"role": "user", "content":
        f"Provide exactly {NUM_GRANTS} CURRENT (2024-2025) U.S. grant opportunities for nonprofits "
        "working on high-school education, college readiness, or youth equity. "
        "Return nothing except a JSON array; each element must include keys: "
        "title, sponsor, summary, deadline, url."
    }
    raw_json = chat([sys, usr],
                    response_format={"type": "json_object"},  # forces valid JSON
                    maxtok=900)

    data = json.loads(raw_json)
    # If GPT wraps the list in an object, pull it out:
    if isinstance(data, dict):
        data = next(iter(data.values()))
    return data[:NUM_GRANTS]

# ─── RANK + FEASIBILITY ──────────────────────────────────────────
def rank_and_score(raw):
    if not raw: return pd.DataFrame()
    df = pd.DataFrame(raw)
    mission_vec = embed(MISSION)
    df["sim"] = [
        float(cosine_similarity([embed(s)], [mission_vec])[0][0])
        for s in df.summary
    ]
    df = df.sort_values("sim", ascending=False).head(TOP_N).reset_index(drop=True)

    feas, why = [], []
    for _, row in df.iterrows():
        prompt = (
            f'Mission: "{MISSION}"\n'
            f'Grant: "{row.title}" – {row.summary}\n\n'
            'Return JSON {"feasibility":"High|Medium|Low","why":"<one sentence>"}'
        )
        ans = chat([{"role": "user", "content": prompt}],
                   response_format={"type": "json_object"},
                   maxtok=60)
        j = json.loads(ans)
        feas.append(j.get("feasibility", "?")); why.append(j.get("why", ""))
    df["feasibility"] = feas
    df["why_fit"]     = why
    return df

# ─── UI ──────────────────────────────────────────────────────────
st.title("CT RISE Network – Smart Grant Finder (v2-JSON)")

st.markdown(f"> **Mission:** {MISSION}")

if st.button("🚀 Find grants for CT RISE", type="primary"):
    with st.spinner("GPT is generating & ranking grants… please wait 60-90 s"):
        st.session_state["tbl"] = rank_and_score(gpt_grants())
        st.success("Done!")

if "tbl" in st.session_state and not st.session_state["tbl"].empty:
    st.dataframe(
        st.session_state["tbl"][
            ["title", "sponsor", "sim", "feasibility", "why_fit", "deadline", "url"]
        ],
        use_container_width=True,
    )
elif "tbl" in st.session_state:
    st.info("GPT returned no grants. Click again or try later.")
else:
    st.caption("Build: v2-JSON — click the rocket to start.")
