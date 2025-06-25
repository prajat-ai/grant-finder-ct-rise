# CT RISE Smart Grant Finder  – v2-robust
import os, time, json, random, re
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import openai

# ── CONFIG ──────────────────────────────────────────────────────
NUM, TOP = 15, 8
DELAY, RETRIES = 2, 5
EMBED_MODEL = "text-embedding-ada-002"
CHAT_MODEL  = "gpt-3.5-turbo"

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

MISSION = ("The Connecticut RISE Network empowers public high schools with "
           "data-driven strategies and personalized support to improve student outcomes "
           "and promote postsecondary success, especially for Black, Latinx, "
           "and low-income youth.")

# ── OpenAI helpers ──────────────────────────────────────────────
def chat(msgs, maxtok=800):
    for a in range(RETRIES):
        try:
            r = openai.ChatCompletion.create(
                model=CHAT_MODEL, messages=msgs,
                max_tokens=maxtok, temperature=0.7
            )
            return r.choices[0].message.content
        except openai.error.RateLimitError:
            time.sleep(DELAY * (2**a) + random.uniform(0,1))
    st.error("OpenAI still rate-limited."); st.stop()

def embed(txt):
    for a in range(RETRIES):
        try:
            return openai.Embedding.create(input=txt, model=EMBED_MODEL)["data"][0]["embedding"]
        except openai.error.RateLimitError:
            time.sleep(DELAY * (2**a))
    return [0.0]*1536

# ── GPT → grants JSON (with fallback) ───────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def gpt_grants():
    sys = {"role":"system","content":"You are a concise grants researcher."}
    user = {"role":"user","content":
        f"Return {NUM} CURRENT (2024-2025) US grant opportunities for nonprofits "
        "focused on high-school education or youth equity. "
        "Respond ONLY with a JSON array of objects having keys: title, sponsor, summary, deadline, url."}
    raw = chat([sys, user], maxtok=900)

    # --- salvage first JSON array in raw text ---
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            snippet = re.search(r"\[.*\]", raw, re.S).group()
            return json.loads(snippet)
        except Exception:
            return []            # fail silently → UI shows “try again”

# ── rank + feasibility ──────────────────────────────────────────
def rank_and_score(js):
    if not js: return pd.DataFrame()
    df = pd.DataFrame(js)[:NUM]
    mvec = embed(MISSION)
    df["sim"] = [float(cosine_similarity([embed(x)], [mvec])[0][0]) for x in df.summary]

    df = df.sort_values("sim", ascending=False).head(TOP).reset_index(drop=True)

    feas, why = [], []
    for _, row in df.iterrows():
        prompt = (f"Mission: {MISSION}\nGrant: {row.title} – {row.summary}\n\n"
                  "Return JSON {'feasibility':'High|Medium|Low','why':'<one sentence>'}")
        try:
            j = json.loads(chat([{"role":"user","content":prompt}], maxtok=60))
            feas.append(j.get("feasibility","?")); why.append(j.get("why",""))
        except Exception:
            feas.append("?"); why.append("parse error")
    df["feasibility"], df["why_fit"] = feas, why
    return df

# ── UI ───────────────────────────────────────────────────────────
st.title("CT RISE Network — Smart Grant Finder (v2-robust)")
st.write("> **Mission:**", MISSION)

if st.button("🚀 Find grants for CT RISE"):
    with st.spinner("GPT thinking… ≈60 s"):
        st.session_state["tbl"] = rank_and_score(gpt_grants())
        st.success("Done!")

if "tbl" in st.session_state and not st.session_state["tbl"].empty:
    st.dataframe(
        st.session_state["tbl"][["title","sponsor","sim","feasibility","why_fit","deadline","url"]],
        use_container_width=True)
elif "tbl" in st.session_state:
    st.info("GPT returned no grants — click again.")
else:
    st.caption("Build tag: v2-robust  |  click the rocket to start.") 
