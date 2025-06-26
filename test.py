# Final Capstone Project (Grant Matcher for CT RISE) – GPT-only version

import os, json, time, random, re
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import openai

# ── CONFIG ─────────────────────────────────────────
MISSION = (
    "The Connecticut RISE Network empowers public high schools with data-driven strategies "
    "and personalized support to improve student outcomes and promote postsecondary success, "
    "especially for Black, Latinx, and low-income youth."
)
NUM_GRANTS  = 10
EMB_MODEL   = "text-embedding-ada-002"
CHAT_MODEL  = "gpt-4o"         # use 3.5 if quota limited
MAX_TRIES   = 3
OPENAI_RETRIES = 4
DELAY_SEC   = 2

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ── OPENAI HELPERS ─────────────────────────────────
def chat(messages, maxtok=1200):
    """GPT chat with exponential backoff on rate-limit."""
    for attempt in range(OPENAI_RETRIES):
        try:
            r = openai.ChatCompletion.create(
                model=CHAT_MODEL,
                messages=messages,
                max_tokens=maxtok,
                temperature=0.3,
                response_format={"type": "json_object"},   # forces valid JSON
            )
            return r.choices[0].message.content
        except openai.error.RateLimitError:
            time.sleep(DELAY_SEC * (2 ** attempt))
    st.error("OpenAI rate-limit persisted."); st.stop()

def embed(text: str):
    """Get Ada-002 embedding with retry."""
    for a in range(OPENAI_RETRIES):
        try:
            v = openai.Embedding.create(input=text, model=EMB_MODEL)
            return v["data"][0]["embedding"]
        except openai.error.RateLimitError:
            time.sleep(DELAY_SEC * (a + 1))
    st.error("Embedding rate-limit."); st.stop()

# ── GPT: Get 10 complete grants (retry until every field present) ─
def get_grants():
    sys = {"role": "system", "content": "You are a grants researcher."}

    prompt = (
        f"Return exactly {NUM_GRANTS} CURRENT grant opportunities (public or private) that serve "
        "U.S. nonprofits focused on high-school education, youth equity, or college readiness. "
        "Return ONLY valid JSON with this schema:\n"
        '[{"title":"…","sponsor":"…","amount":"…","deadline":"YYYY-MM-DD or rolling","url":"…","summary":"…"}]'
    )

    for _ in range(MAX_TRIES):
        raw = chat([sys, {"role":"user","content":prompt}])
        try:
            data = json.loads(raw)
            # ensure structure
            if (
                isinstance(data, list) and
                len(data) == NUM_GRANTS and
                all(all(k in g for k in ("title","sponsor","amount","deadline","url","summary")) for g in data)
            ):
                return data
        except json.JSONDecodeError:
            pass
        prompt = "Your previous response was invalid. Return ONLY the JSON array with all required keys."
    st.error("Could not get a complete grant list from GPT."); st.stop()

# ── Rank & annotate grants ─────────────────────────
def process(df_raw: pd.DataFrame):
    mvec = embed(MISSION)
    df_raw["%Match"] = (
        df_raw.summary.apply(lambda s: cosine_similarity([embed(s)], [mvec])[0][0] * 100)
        .round(1)
    )

    # GPT explanation of fit
    whys = []
    for _, row in df_raw.iterrows():
        q = (
            f'Nonprofit mission: "{MISSION}"\n\nGrant: "{row.title}" – {row.summary}\n'
            "Explain in one sentence why this grant aligns with the mission."
        )
        whys.append(chat([{"role":"user","content":q}], maxtok=60))
    df_raw["Why Fit"] = whys

    ranked = df_raw.sort_values("%Match", ascending=False).reset_index(drop=True)
    ranked.index = ranked.index + 1
    ranked.insert(0, "Rank", ranked.index)
    return ranked

# ── UI ─────────────────────────────────────────────
st.title("Final Capstone Project (Grant Matcher for CT RISE using ChatGPT Search)")
st.write("> **Mission:**", MISSION)

if st.button("🚀 Generate & Rank 10 Grants", type="primary"):
    with st.spinner("GPT searching web, compiling grants, computing similarity…"):
        grants = get_grants()
        table  = process(pd.DataFrame(grants))
        st.session_state["tbl"] = table
        st.success("Completed!")

if "tbl" in st.session_state:
    st.dataframe(
        st.session_state["tbl"][
            ["Rank","title","sponsor","amount","deadline","%Match","Why Fit","url","summary"]
        ],
        use_container_width=True
    )
else:
    st.caption("Press the rocket to generate your ranked grant list (GPT-powered).")
