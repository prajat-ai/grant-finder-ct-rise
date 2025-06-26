# Final Capstone Project (Grant Matcher for CT RISE using ChatGPT Search)

import os, json, re, time, random
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
NUM_NEEDED   = 10           # table rows
GPT_REQUEST  = 12           # ask for a few extras, drop invalids
CHAT_MODEL   = "gpt-3.5-turbo-1106"   # JSON mode support
EMB_MODEL    = "text-embedding-ada-002"
OPENAI_RETRY = 4
BACKOFF      = 2            # sec
PROMPT_RETRY = 5

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ── OPENAI HELPERS ────────────────────────────────
def chat_json(msgs, maxtok=1000):
    """Return content from GPT with JSON mode + back-off."""
    for a in range(OPENAI_RETRY):
        try:
            r = openai.ChatCompletion.create(
                model=CHAT_MODEL,
                messages=msgs,
                response_format={"type": "json_object"},
                max_tokens=maxtok,
                temperature=0.3,
            )
            return r.choices[0].message.content
        except openai.error.RateLimitError:
            time.sleep(BACKOFF * (a + 1))
    st.error("OpenAI rate-limited; try again later."); st.stop()

def embed(txt):
    for a in range(OPENAI_RETRY):
        try:
            return openai.Embedding.create(input=txt, model=EMB_MODEL)["data"][0]["embedding"]
        except openai.error.RateLimitError:
            time.sleep(BACKOFF * (a + 1))
    st.error("Embedding rate-limited."); st.stop()

# ── 1 · Fetch grant list (robust) ──────────────────
def fetch_grants():
    sys = {"role": "system", "content": "You are a grants researcher."}
    user_prompt = (
        f"Provide {GPT_REQUEST} CURRENT (2024-2025) grant opportunities for US nonprofits "
        "focused on high-school education, youth equity, or college readiness. "
        "Return ONLY a JSON array where each element has keys: "
        "title, sponsor, amount, deadline, url, summary."
    )
    msgs = [sys, {"role":"user", "content": user_prompt}]

    for _ in range(PROMPT_RETRY):
        raw = chat_json(msgs)
        # salvage JSON array even if GPT wraps it
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", raw, re.S)
            data = json.loads(match.group()) if match else []

        # fill missing keys with "N/A"
        cleaned=[]
        for d in data:
            row={k: d.get(k,"N/A") for k in ("title","sponsor","amount","deadline","url","summary")}
            cleaned.append(row)
        if len(cleaned) >= NUM_NEEDED:
            return cleaned[:NUM_NEEDED]
        msgs.append({"role":"user","content":"Please try again. Remember: JSON array only."})

    st.error("Could not obtain enough grant data from GPT.")
    st.stop()

# ── 2 · Rank & annotate ───────────────────────────
def rank(df: pd.DataFrame):
    mvec = embed(MISSION)
    df["%Match"] = (
        df.summary.apply(lambda s: cosine_similarity([embed(s)], [mvec])[0][0] * 100)
        .round(1)
    )
    df = df.sort_values("%Match", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.insert(0, "Rank", df.index)

    whys=[]
    for _, r in df.iterrows():
        q = (f'Mission: "{MISSION}"\nGrant: "{r.title}" – {r.summary}\n'
             "Explain briefly (1 sentence) why it aligns.")
        whys.append(openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":q}],
            max_tokens=60,
            temperature=0.3,
        ).choices[0].message.content.strip())
    df["Why Fit"] = whys
    return df

# ── UI ─────────────────────────────────────────────
st.title("Final Capstone Project (Grant Matcher for CT RISE using ChatGPT Search)")
st.write("> **Mission:**", MISSION)

if st.button("🚀 Generate & Rank 10 Grants", type="primary"):
    with st.spinner("GPT searching web, compiling grants, computing similarity…"):
        g
