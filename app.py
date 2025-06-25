# app.py ──────────────────────────────────────────────────────────
"""
CT RISE – Smart Grant Finder (GPT-powered version, no external APIs)
• GPT-3.5 returns 25 education-focused grants in JSON.
• OpenAI embeddings rank by similarity to CT RISE mission.
• GPT adds a feasibility label + 1-sentence rationale on the top N.
"""

import os, time, json, logging
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import openai

# ── CONFIG ───────────────────────────────────────────────────────
NUM_GRANTS    = 25       # GPT generates this many grants
TOP_N_GPT     = 10       # run GPT feasibility on top-N
SLEEP_SECONDS = 1        # delay per API call (rate-limit safety)
RETRIES       = 3
EMBED_MODEL   = "text-embedding-ada-002"
CHAT_MODEL    = "gpt-3.5-turbo"          # free-tier friendly

# ── LOAD KEY ─────────────────────────────────────────────────────
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
def call_openai_chat(messages, model=CHAT_MODEL, max_tokens=800):
    for attempt in range(RETRIES):
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return resp.choices[0].message.content
        except openai.error.RateLimitError:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("OpenAI rate-limit persisted.")

def get_embedding(text, model=EMBED_MODEL):
    for attempt in range(RETRIES):
        try:
            resp = openai.Embedding.create(input=text, model=model)
            return resp["data"][0]["embedding"]
        except openai.error.RateLimitError:
            time.sleep(5 * (attempt + 1))
    return [0.0] * 1536

# ── 1. GPT: GENERATE GRANTS LIST ─────────────────────────────────
def gpt_generate_grants(n=NUM_GRANTS):
    sys = {"role": "system", "content": "You are a grants researcher."}
    usr = {
        "role": "user",
        "content": (
            f"Provide {n} CURRENT (2024-2025) grant opportunities for US nonprofits "
            "focused on high-school education, college readiness, or youth equity. "
            "Return ONLY valid JSON list like:\n"
            '[{"title":"...", "sponsor":"...", "summary":"...", "deadline":"...", "url":"..."}]'
        ),
    }
    raw = call_openai_chat([sys, usr])
    # Safe JSON load
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logging.error("GPT JSON parse failure.")
        return []
    return data[:n]

# ── 2. RANK + FEASIBILITY ────────────────────────────────────────
def rank_and_score(grants_json):
    if not grants_json:
        return pd.DataFrame()
    df = pd.DataFrame(grants_json)
    mission_vec = get_embedding(CT_RISE_MISSION)
    sims = []
    for descr in df["summary"]:
        sims.append(float(cosine_similarity([get_embedding(descr)], [mission_vec])[0][0]))
        time.sleep(SLEEP_SECONDS)
    df["similarity"] = sims
    df = df.sort_values("similarity", ascending=False).head(TOP_N_GPT).reset_index(drop=True)

    feas, why = [], []
    for _, row in df.iterrows():
        prompt = (
            f'Nonprofit mission: "{CT_RISE_MISSION}"\n\n'
            f'Grant: "{row.title}" – {row.summary}\n\n'
            'Answer ONLY JSON like {"feasibility":"High","why":"<one sentence>"}'
        )
        j = call_openai_chat([{"role": "user", "content": prompt}], max_tokens=60)
        try:
            parsed = json.loads(j)
            feas.append(parsed.get("feasibility", "Unknown"))
            why.append(parsed.get("why", ""))
        except json.JSONDecodeError:
            feas.append("Unknown")
            why.append("Could not parse")
        time.sleep(SLEEP_SECONDS)
    df["feasibility"] = feas
    df["why_fit"]    = why
    return df

# ── STREAMLIT UI ─────────────────────────────────────────────────
st.title("CT RISE Network — Smart Grant Finder (GPT version)")
st.markdown(
    "> **Mission**: The Connecticut RISE Network empowers public high schools with data-driven "
    "strategies and personalized support to improve student outcomes, especially for "
    "Black, Latinx, and low-income youth."
)

if st.button("🔄 Generate & rank grants"):
    with st.spinner("Talking to GPT… please wait ≈1-2 min"):
        grants_raw = gpt_generate_grants()
        table      = rank_and_score(grants_raw)
        st.session_state["grants"] = table
        st.success("Done!")

if "grants" in st.session_state and not st.session_state["grants"].empty:
    st.subheader("Top matched grants")
    st.dataframe(
        st.session_state["grants"][["title", "sponsor", "similarity", "feasibility", "why_fit", "deadline", "url"]],
        use_container_width=True,
    )
elif "grants" in st.session_state:
    st.info("GPT returned no usable grants — click the button again.")
else:
    st.info("Click **Generate & rank grants** to get matches.")
