# Final Capstone Project – CT RISE Grant Matcher (GPT-4o Search, v2)

import os, json, re, time, pandas as pd, streamlit as st, openai, datetime as dt
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# ── CONFIG ─────────────────────────────────────────
SEARCH_MODEL = "gpt-4o-mini-search-preview"
CHAT_MODEL   = "gpt-3.5-turbo"
EMB_MODEL    = "text-embedding-ada-002"
NEEDED       = 10
RETRY        = 4
WAIT         = 2

MISSION = (
    "The Connecticut RISE Network empowers public high schools with data-driven strategies "
    "and personalized support to improve student outcomes and promote post-secondary success, "
    "especially for Black, Latinx, and low-income youth."
)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ── HELPERS ─────────────────────────────────────────
def backoff(fn):
    def wrap(*a, **k):
        for i in range(RETRY):
            try:
                return fn(*a, **k)
            except openai.RateLimitError:
                time.sleep(WAIT * (i + 1))
        st.error("OpenAI rate-limit; try again later."); st.stop()
    return wrap

@backoff
def embed(text):
    return openai.embeddings.create(model=EMB_MODEL, input=text).data[0].embedding

@backoff
def chat(model, messages):
    return openai.chat.completions.create(model=model, messages=messages)

# ── 1 · FETCH GRANTS VIA SEARCH MODEL ──────────────
def fetch_grants():
    prompt = (
        f"search: Provide exactly {NEEDED} CURRENT grant opportunities for U.S. nonprofits "
        "focused on high-school education, youth equity, or college readiness. "
        "For each grant return JSON with keys: "
        "title, sponsor, amount, deadline (YYYY-MM-DD or 'rolling'), "
        "url (direct *Apply Now* link), summary."
    )
    raw = chat(SEARCH_MODEL, [{"role":"user","content":prompt}]).choices[0].message.content
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, re.S)
        data = json.loads(m.group()) if m else []

    # normalize & keep only future deadlines
    today = dt.date.today()
    cleaned = []
    for d in data:
        g = {k: d.get(k, "N/A") for k in
             ("title","sponsor","amount","deadline","url","summary")}
        # parse deadline
        dl = g["deadline"].lower()
        future = True
        if dl != "rolling":
            try:
                future = dt.datetime.strptime(dl[:10], "%Y-%m-%d").date() >= today
            except Exception:
                future = False
        if future:
            cleaned.append(g)
        if len(cleaned) == NEEDED:
            break
    return cleaned

# ── 2 · RANK + WHY-FIT ─────────────────────────────
def build_table(rows):
    df = pd.DataFrame(rows)
    base_vec = embed(MISSION)
    df["Match%"] = (
        df.summary.apply(lambda s: cosine_similarity([embed(s)], [base_vec])[0][0]*100)
        .round(1)
    )
    df = df.sort_values("Match%", ascending=False).reset_index(drop=True)

    # Why Fit sentence
    whys=[]
    for _, r in df.iterrows():
        q = (f'Why does the grant "{r.title}" align with the mission '
             f'"{MISSION}"? Respond with one sentence.')
        a = chat(CHAT_MODEL, [{"role":"user","content":q}]).choices[0].message.content.strip()
        whys.append(a)
    df["Why It Fits"] = whys

    # reorder columns
    return df[["title","Match%","amount","deadline","sponsor","summary","url","Why It Fits"]] \
        .rename(columns={
            "title":"Title","amount":"Amount",
            "deadline":"Deadline","sponsor":"Sponsor",
            "summary":"Grant Summary","url":"URL"
        })

# ── STREAMLIT UI ───────────────────────────────────
st.title("Final Capstone Project (Grant Matcher for CT RISE – GPT-4o Search)")
st.write("> **Mission:**", MISSION)

if st.button("🚀 Generate & Rank 10 Grants", type="primary"):
    with st.spinner("GPT-4o searching web and compiling grants…"):
        grants = fetch_grants()
        if len(grants) < NEEDED:
            st.error("Model returned fewer than 10 future-deadline grants – click again.")
        else:
            st.session_state["tbl"] = build_table(grants)
            st.success("Done!")

if "tbl" in st.session_state:
    st.dataframe(st.session_state["tbl"], use_container_width=True)
else:
    st.caption("Press the rocket to generate your ranked grant list.")
