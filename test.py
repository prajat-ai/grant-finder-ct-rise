# Final Capstone Project (Grant Matcher for CT RISE — GPT-4o Search, tool-enabled)

import os, time, json, re, pandas as pd, streamlit as st, openai
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# ─── CONFIG ───────────────────────────────────────────
SEARCH_MODEL = "gpt-4o-mini-search-preview"   # supports built-in search tool
CHAT_MODEL   = "gpt-3.5-turbo"
EMB_MODEL    = "text-embedding-ada-002"
NEEDED       = 10
RETRIES      = 4
PAUSE        = 2

MISSION = (
    "The Connecticut RISE Network empowers public high schools with data-driven strategies "
    "and personalized support to improve student outcomes and promote postsecondary success, "
    "especially for Black, Latinx, and low-income youth."
)

# ─── KEYS ─────────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ─── HELPERS ─────────────────────────────────────────
def chat(model, messages, **kw):
    for a in range(RETRIES):
        try:
            resp = openai.ChatCompletion.create(model=model, messages=messages, **kw)
            return resp
        except openai.error.RateLimitError:
            time.sleep(PAUSE * (a + 1))
    st.error("OpenAI rate-limited."); st.stop()

def embed(txt):
    for a in range(RETRIES):
        try:
            return openai.Embedding.create(input=txt, model=EMB_MODEL)["data"][0]["embedding"]
        except openai.error.RateLimitError:
            time.sleep(PAUSE * (a + 1))
    st.error("Embedding rate-limited."); st.stop()

# ─── STEP 1: one call to SEARCH model ───────────────
def fetch_grants():
    prompt = (
        f"Search the web and return exactly {NEEDED} current US grant opportunities for "
        "nonprofits working on high-school education, youth equity, or college readiness. "
        "Output ONLY a JSON array where each element has keys: "
        "title, sponsor, amount, deadline, url, summary."
    )
    resp = chat(
        SEARCH_MODEL,
        messages=[{"role":"user","content":prompt}],
        tools=[{"type":"search"}],          # allow the internal search tool
        tool_choice="auto",
        temperature=0.3,
        max_tokens=1400
    )
    content = resp.choices[0].message.content

    # robust parse
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", content, re.S)
        data = json.loads(m.group()) if m else []
    cleaned = [{k: d.get(k, "N/A") for k in
               ("title","sponsor","amount","deadline","url","summary")} for d in data]
    return cleaned[:NEEDED]

# ─── STEP 2: rank + “Why Fit” ───────────────────────
def rank(df):
    mvec = embed(MISSION)
    df["%Match"] = (
        df.summary.apply(lambda s: cosine_similarity([embed(s)], [mvec])[0][0]*100).round(1)
    )
    df = df.sort_values("%Match", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.insert(0,"Rank", df.index)

    whys=[]
    for _, row in df.iterrows():
        q = (f'In one sentence: why does "{row.title}" align with the mission "{MISSION}"?')
        ans = chat(CHAT_MODEL, [{"role":"user","content":q}], max_tokens=60).choices[0].message.content.strip()
        whys.append(ans)
    df["Why Fit"] = whys
    return df

# ─── UI ─────────────────────────────────────────────
st.title("Final Capstone Project (Grant Matcher for CT RISE using ChatGPT Search)")
st.write("> **Mission:**", MISSION)

if st.button("🚀 Generate & Rank 10 Grantz", type="primary"):
    with st.spinner("GPT-4o is searching and compiling grants…"):
        rows = fetch_grants()
        if len(rows) < NEEDED:
            st.error("Model returned fewer than 10 grants. Click again.")
        else:
            st.session_state["tbl"] = rank(pd.DataFrame(rows))
            st.success("Done!")

if "tbl" in st.session_state:
    st.dataframe(
        st.session_state["tbl"][
            ["Rank","title","sponsor","amount","deadline","%Match","Why Fit","url","summary"]
        ],
        use_container_width=True
    )
else:
    st.caption("Press the rocket to generate a ranked list.")
