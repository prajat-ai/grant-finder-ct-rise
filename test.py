# Final Capstone Project – CT RISE Grant Matcher (GPT-4o Search, stable)

import os, json, re, time, pandas as pd, streamlit as st, openai
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# ── CONFIG ─────────────────────────────────────────
SEARCH_MODEL = "gpt-4o-mini-search-preview"
CHAT_MODEL   = "gpt-3.5-turbo"
EMB_MODEL    = "text-embedding-ada-002"
ROWS         = 10
RETRY        = 4
DELAY        = 2

MISSION = (
    "The Connecticut RISE Network empowers public high schools with data-driven strategies "
    "and personalized support to improve student outcomes and promote postsecondary success, "
    "especially for Black, Latinx, and low-income youth."
)

# ── KEYS ───────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ── HELPERS ───────────────────────────────────────
def backoff(fn):
    def wrap(*a, **k):
        for i in range(RETRY):
            try:
                return fn(*a, **k)
            except openai.RateLimitError:
                time.sleep(DELAY * (i + 1))
        st.error("OpenAI rate-limit; try later."); st.stop()
    return wrap

@backoff
def chat(model, messages, **kw):
    return openai.chat.completions.create(model=model, messages=messages, **kw)

@backoff
def embed(txt):
    return openai.embeddings.create(model=EMB_MODEL, input=txt).data[0].embedding

# ── STEP 1 – one search call ──────────────────────
def fetch_grants():
    user_msg = (
        f"Search the web and return exactly {ROWS} CURRENT US grant opportunities suitable for "
        "non-profits working on high-school education, youth equity, or college readiness. "
        "Respond ONLY with a JSON array; each object MUST include keys: "
        "title, sponsor, amount, deadline, url, summary."
    )
    resp = chat(SEARCH_MODEL, [{"role":"user","content": user_msg}], max_tokens=1500, temperature=0.3)
    txt  = resp.choices[0].message.content

    # Robust JSON extraction
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", txt, re.S)
        data = json.loads(m.group()) if m else []

    cleaned = [
        {k: d.get(k, "N/A") for k in
         ("title","sponsor","amount","deadline","url","summary")}
        for d in data
    ]
    return cleaned[:ROWS]

# ── STEP 2 – rank & “Why Fit” ─────────────────────
def build(df):
    m_vec = embed(MISSION)
    df["%Match"] = (
        df.summary.apply(lambda s: cosine_similarity([embed(s)], [m_vec])[0][0]*100)
        .round(1)
    )
    df = df.sort_values("%Match", ascending=False).reset_index(drop=True)
    df.index += 1
    df.insert(0, "Rank", df.index)

    why = []
    for _, r in df.iterrows():
        q = (f'In one sentence: why does the grant "{r.title}" help achieve the mission '
             f'"{MISSION}"?')
        a = chat(CHAT_MODEL, [{"role":"user","content": q}], max_tokens=60,
                 temperature=0.3).choices[0].message.content.strip()
        why.append(a)
    df["Why Fit"] = why
    return df

# ── STREAMLIT UI ──────────────────────────────────
st.title("Final Capstone Project (Grant Matcher for CT RISE using GPT-4o Search)")
st.write("> **Mission:**", MISSION)

if st.button("🚀 Generate & Rank 10 Grants", type="primary"):
    with st.spinner("GPT-4o is finding grants and computing similarity …"):
        rows = fetch_grants()
        if len(rows) < ROWS:
            st.error("Model returned fewer than 10 grants; click again.")
        else:
            st.session_state["tbl"] = build(pd.DataFrame(rows))
            st.success("Done!")

if "tbl" in st.session_state:
    st.dataframe(
        st.session_state["tbl"][
            ["Rank","title","sponsor","amount","deadline",
             "%Match","Why Fit","url","summary"]
        ],
        use_container_width=True
    )
else:
    st.caption("Press the rocket to generate your ranked list.")
