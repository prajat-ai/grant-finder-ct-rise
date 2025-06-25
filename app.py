import os, time, json, random
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import openai
import os

st.sidebar.write("KEY loaded:", bool(os.getenv("OPENAI_API_KEY")))

# ── settings ────────────────────────────────────────────────────
NUM_GRANTS, TOP_N = 15, 8          # fewer calls → fewer 429s
EMBED_MODEL  = "text-embedding-ada-002"
CHAT_MODEL   = "gpt-3.5-turbo"
BASE_DELAY   = 2                   # seconds
MAX_RETRIES  = 5

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

MISSION = ("The Connecticut RISE Network empowers public high schools with "
           "data-driven strategies and personalized support to improve student outcomes "
           "and promote postsecondary success, especially for Black, Latinx, "
           "and low-income youth.")

# ── helper wrappers ─────────────────────────────────────────────
def chat(messages, maxtok=700):
    """Robust call with exponential back-off."""
    for a in range(MAX_RETRIES):
        try:
            r = openai.ChatCompletion.create(model=CHAT_MODEL,
                                             messages=messages,
                                             max_tokens=maxtok)
            return r.choices[0].message.content
        except openai.error.RateLimitError:
            wait = BASE_DELAY * (2 ** a) + random.uniform(0, 1)
            time.sleep(wait)
    st.error("OpenAI still rate-limited after several tries."); st.stop()

def embed(txt):
    for a in range(MAX_RETRIES):
        try:
            return openai.Embedding.create(input=txt, model=EMBED_MODEL)["data"][0]["embedding"]
        except openai.error.RateLimitError:
            time.sleep(BASE_DELAY * (2 ** a))
    return [0.0] * 1536

# ── GPT: generate grants list ───────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def gpt_grants():
    sys = {"role":"system","content":"You are a grants researcher."}
    usr = {"role":"user","content":
        f"Provide {NUM_GRANTS} CURRENT (2024-2025) US grant opportunities for nonprofits "
        "focused on high-school education or youth equity. Return ONLY JSON list like "
        '[{\"title\":\"...\",\"sponsor\":\"...\",\"summary\":\"...\",\"deadline\":\"...\",\"url\":\"...\"}]'}
    try:
        return json.loads(chat([sys, usr]))
    except json.JSONDecodeError:
        return []

# ── ranking + feasibility ───────────────────────────────────────
def rank_and_score(raw):
    if not raw: return pd.DataFrame()
    df = pd.DataFrame(raw)[:NUM_GRANTS]
    mvec = embed(MISSION)
    sims=[]
    for s in df.summary:
        sims.append(float(cosine_similarity([embed(s)], [mvec])[0][0]))
    df["sim"]=sims
    df=df.sort_values("sim",ascending=False).head(TOP_N).reset_index(drop=True)

    feas, why = [], []
    for _, row in df.iterrows():
        prompt = (f'Mission: "{MISSION}"\nGrant: "{row.title}" – {row.summary}\n\n'
                  'ONLY JSON {"feasibility":"High|Medium|Low","why":"<one sentence>"}')
        try:
            j = json.loads(chat([{"role":"user","content":prompt}], maxtok=60))
            feas.append(j.get("feasibility","?")); why.append(j.get("why",""))
        except json.JSONDecodeError:
            feas.append("?"); why.append("GPT parse error")
    df["feasibility"]=feas; df["why"]=why
    return df

# ── UI ──────────────────────────────────────────────────────────
st.title("CT RISE — Smart Grant Finder (GPT)")
st.write("> **Mission:**", MISSION)

if st.button("🔄 Generate & rank grants"):
    with st.spinner("Generating grant list… please wait ≈1 min"):
        st.session_state["tbl"] = rank_and_score(gpt_grants())
        st.success("Done!")

if "tbl" in st.session_state and not st.session_state["tbl"].empty:
    st.dataframe(
        st.session_state["tbl"][["title","sponsor","sim","feasibility","why","deadline","url"]],
        use_container_width=True)
elif "tbl" in st.session_state:
    st.info("GPT returned no grants; click again in a few minutes.")
else:
    st.info("Click the button to generate matches.")
