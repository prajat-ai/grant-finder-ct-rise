# CT RISE — Persistent Grant Fit Analyzer
# Saves every analysis to grants_history.csv so data survive page reloads & redeploys.

import os, json, re, time, datetime as dt, pandas as pd, streamlit as st, openai
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# ─── CONFIG ─────────────────────────────────────────
SEARCH_MODEL = "gpt-4o-mini-search-preview"
CHAT_MODEL   = "gpt-3.5-turbo"
EMB_MODEL    = "text-embedding-ada-002"
CSV_PATH     = "grants_history.csv"       # persistence file
API_RETRY    = 4
BACKOFF      = 2

MISSION = (
    "The Connecticut RISE Network empowers public high schools with data-driven strategies "
    "and personalized support to improve student outcomes and promote post-secondary success, "
    "especially for Black, Latinx, and low-income youth."
)

# ─── KEYS ───────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ─── HELPERS (retry wrappers) ───────────────────────
def retry(fn):
    def wrap(*a, **k):
        for i in range(API_RETRY):
            try:
                return fn(*a, **k)
            except openai.RateLimitError:
                time.sleep(BACKOFF * (i + 1))
        st.error("OpenAI rate-limit—try again later."); st.stop()
    return wrap

@retry
def chat(model, messages):  # generic chat
    return openai.chat.completions.create(model=model, messages=messages)

@retry
def get_embed(text):
    return openai.embeddings.create(model=EMB_MODEL, input=text).data[0].embedding

# ─── LOAD / SAVE TABLE PERSISTENCE ──────────────────
def load_history() -> pd.DataFrame:
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    return pd.DataFrame(columns=[
        "Title","Match%","Amount","Deadline","Sponsor",
        "Grant Summary","URL","Recommendation"
    ])

def save_history(df: pd.DataFrame):
    df.to_csv(CSV_PATH, index=False)

# ─── LLM SCRAPE OF SINGLE GRANT URL ─────────────────
def scrape_grant(url: str) -> dict | None:
    prompt = (
        f"search: Visit {url} and return JSON with keys "
        "{title, sponsor, amount, deadline (YYYY-MM-DD or 'rolling'), summary}. "
        "If any field missing use 'N/A'."
    )
    raw = chat(SEARCH_MODEL, [{"role":"user","content":prompt}]).choices[0].message.content
    match = re.search(r"\{.*\}", raw, re.S)
    try:
        data = json.loads(match.group() if match else raw)
    except Exception:
        return None
    data["url"] = url
    return {k: data.get(k,"N/A") for k in
            ("title","sponsor","amount","deadline","summary","url")}

def future_deadline(dl: str) -> bool:
    if dl.lower() == "rolling": return True
    try: return dt.datetime.strptime(dl[:10], "%Y-%m-%d").date() >= dt.date.today()
    except Exception: return False

# ─── STREAMLIT UI ───────────────────────────────────
st.title("Grant Fit Analyzer for CT RISE (persistent)")

# load history into session_state once
if "table" not in st.session_state:
    st.session_state["table"] = load_history()

st.write("> **Mission:**", MISSION)
url = st.text_input("Paste a grant application URL")

if st.button("Analyze Grant") and url.strip():
    with st.spinner("Analyzing…"):
        grant = scrape_grant(url.strip())
        if not grant:
            st.error("Could not parse that URL.")
        elif not future_deadline(grant["deadline"]):
            st.warning("Deadline already passed—ignored.")
        else:
            # duplicate check
            if ((st.session_state["table"]["URL"].str.lower() == grant["url"].lower()).any()
                    or (st.session_state["table"]["Title"].str.lower() == grant["title"].lower()).any()):
                st.info("Grant already in the table.")
            else:
                # similarity + recommendation
                sim = cosine_similarity(
                    [get_embed(grant["summary"])],
                    [get_embed(MISSION)]
                )[0][0] * 100
                rec_prompt = (
                    f'Mission: "{MISSION}"\n\nGrant: "{grant["title"]}" – {grant["summary"]}\n'
                    "In 1-2 sentences, say if this is a strong fit and why."
                )
                rec = chat(CHAT_MODEL,[{"role":"user","content":rec_prompt}]).\
                      choices[0].message.content.strip()

                new_row = {
                    "Title": grant["title"],
                    "Match%": round(sim,1),
                    "Amount": grant["amount"],
                    "Deadline": grant["deadline"],
                    "Sponsor": grant["sponsor"],
                    "Grant Summary": grant["summary"],
                    "URL": grant["url"],
                    "Recommendation": rec
                }
                st.session_state["table"] = pd.concat(
                    [st.session_state["table"], pd.DataFrame([new_row])],
                    ignore_index=True
                )
                save_history(st.session_state["table"])   # persist to CSV
                st.success("Grant added & saved!")

st.subheader("Analyzed Grants (saved)")
st.dataframe(st.session_state["table"], use_container_width=True)
