# Final Capstone Project – CT RISE Grant Matcher (GPT-4o Search, future deadlines only)

import os, json, re, time, datetime as dt, pandas as pd, streamlit as st, openai
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# ── CONFIG ──────────────────────────────────
SEARCH_MODEL = "gpt-4o-mini-search-preview"
CHAT_MODEL   = "gpt-3.5-turbo"
EMB_MODEL    = "text-embedding-ada-002"
TARGET       = 10           # want 10 rows
ASK_FOR      = 20           # request 20, keep ≥10
MAX_TRY      = 6
RETRY_API    = 4
BACKOFF      = 2            # sec

MISSION = (
    "The Connecticut RISE Network empowers public high schools with data-driven strategies "
    "and personalized support to improve student outcomes and promote post-secondary success, "
    "especially for Black, Latinx, and low-income youth."
)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ── UTILS ─────────────────────────────────────
def retry(fn):
    def wrap(*a, **k):
        for i in range(RETRY_API):
            try:
                return fn(*a, **k)
            except openai.RateLimitError:
                time.sleep(BACKOFF * (i + 1))
        st.error("OpenAI rate-limit; try later."); st.stop()
    return wrap

@retry
def embed(txt):  # embedding call
    return openai.embeddings.create(model=EMB_MODEL, input=txt).data[0].embedding

@retry
def chat(model, msgs):  # chat call
    return openai.chat.completions.create(model=model, messages=msgs)

# ── 1 · fetch grants via search-preview ───────
def fetch_grants():
    today = dt.date.today().isoformat()
    base_prompt = (
        f"search: Provide {ASK_FOR} CURRENT grant opportunities for US nonprofits focused on "
        "high-school education, youth equity, or college readiness. EXCLUDE grants whose "
        f"deadline is before {today}. For each, return JSON with keys: "
        "title, sponsor, amount, deadline (YYYY-MM-DD or 'rolling'), "
        "url (direct apply link), summary."
    )
    for _ in range(MAX_TRY):
        txt = chat(SEARCH_MODEL, [{"role":"user","content":base_prompt}]).choices[0].message.content
        try:
            data = json.loads(txt)
        except json.JSONDecodeError:
            m = re.search(r"\[.*\]", txt, re.S)
            data = json.loads(m.group()) if m else []
        # validate & filter future deadlines
        rows=[]
        for d in data:
            g={k: d.get(k,"N/A") for k in
               ("title","sponsor","amount","deadline","url","summary")}
            dl=g["deadline"].lower()
            keep=True
            if dl!="rolling":
                try:
                    keep = dt.datetime.strptime(dl[:10], "%Y-%m-%d").date() >= dt.date.today()
                except Exception:
                    keep = False
            if keep: rows.append(g)
            if len(rows)==TARGET: break
        if len(rows)>=TARGET: return rows
    return rows  # may be <10 after retries

# ── 2 · rank & add “Why It Fits” ───────────────
def rank_table(rows):
    df=pd.DataFrame(rows)
    base_vec=embed(MISSION)
    df["Match%"]=(df.summary.apply(lambda s:
        cosine_similarity([embed(s)], [base_vec])[0][0]*100).round(1))
    df=df.sort_values("Match%",ascending=False).head(TARGET).reset_index(drop=True)

    # why-fit
    whys=[]
    for _,r in df.iterrows():
        q=(f'In one sentence, why does the grant "{r.title}" align with the mission '
           f'"{MISSION}"?')
        whys.append(chat(CHAT_MODEL,[{"role":"user","content":q}])
                    .choices[0].message.content.strip())
    df["Why It Fits"]=whys

    # reorder columns
    return df[["title","Match%","amount","deadline",
               "sponsor","summary","url","Why It Fits"]].rename(columns={
        "title":"Title","amount":"Amount","deadline":"Deadline",
        "sponsor":"Sponsor","summary":"Grant Summary","url":"URL"
    })

# ── UI ──────────────────────────────────────────
st.title("Final Capstone Project (Grant Matcher for CT RISE – GPT-4o Search)")
st.write("> **Mission:**", MISSION)

if st.button("🚀 Generate & Rank 10 Grants", type="primary"):
    with st.spinner("GPT-4o searching and compiling grants…"):
        grants=fetch_grants()
        if len(grants)<TARGET:
            st.error("Couldn’t gather 10 future-deadline grants. Click again.")
        else:
            st.session_state["tbl"]=rank_table(grants)
            st.success("Done!")

if "tbl" in st.session_state:
    st.dataframe(st.session_state["tbl"], use_container_width=True)
else:
    st.caption("Press the rocket to generate your ranked list.")
