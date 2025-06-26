# grant_analyzer.py  –  Persistent Grant-Fit Dashboard for CT RISE

import os, json, re, time, datetime as dt
import pandas as pd, streamlit as st, openai
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# ── APP CONFIG ───────────────────────────────────────────────────
SEARCH_MODEL = "gpt-4o-mini-search-preview"
CHAT_MODEL   = "gpt-3.5-turbo"
EMB_MODEL    = "text-embedding-ada-002"
CSV_PATH     = "grants_history.csv"
API_RETRY    = 4
BACKOFF      = 2
COLS = [
    "Title","Match%","Amount","Deadline","Sponsor",
    "Grant Summary","URL","Recommendation"
]
MISSION = (
    "The Connecticut RISE Network empowers public high schools with data-driven strategies "
    "and personalized support to improve student outcomes and promote post-secondary success, "
    "especially for Black, Latinx, and low-income youth."
)

# ── OPENAI KEY ──────────────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ── RETRY DECORATOR ─────────────────────────────────────────────
def retry(fn):
    def wrapper(*a, **k):
        for i in range(API_RETRY):
            try:
                return fn(*a, **k)
            except openai.error.RateLimitError:
                time.sleep(BACKOFF * (i + 1))
        st.error("OpenAI rate-limit; try later."); st.stop()
    return wrapper

@retry
def chat(model, msgs):
    return openai.chat.completions.create(model=model, messages=msgs)

@retry
def get_embed(text):
    return openai.embeddings.create(model=EMB_MODEL, input=text).data[0].embedding

# ── CSV LOAD / SAVE ─────────────────────────────────────────────
def load_history():
    return pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else pd.DataFrame(columns=COLS)

def save_history(df):
    df.to_csv(CSV_PATH, index=False)

# ── GRANT SCRAPER (robust JSON parser) ──────────────────────────
def scrape_grant(url: str):
    prompt = (
        f"search: Visit {url} and return JSON with keys "
        "{title, sponsor, amount, deadline (YYYY-MM-DD or 'rolling'), summary}. "
        "If a field is missing use 'N/A'. Respond ONLY with JSON."
    )
    raw = chat(SEARCH_MODEL, [{"role":"user","content":prompt}]).choices[0].message.content
    m = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", raw, re.S) or \
        re.search(r"(\{.*?\}|\[.*?\])", raw, re.S)
    if not m: return None
    try:
        data = json.loads(m.group(1))
        if isinstance(data, list): data = data[0]
    except Exception:
        return None
    data["url"] = url
    return {k: data.get(k,"N/A") for k in
            ("title","sponsor","amount","deadline","summary","url")}

def future_deadline(dl: str) -> bool:
    if dl.lower() == "rolling": return True
    try:
        return dt.datetime.strptime(dl[:10], "%Y-%m-%d").date() >= dt.date.today()
    except Exception:
        return False

# ── STREAMLIT UI ────────────────────────────────────────────────
st.set_page_config(page_title="CT RISE Grant Analyzer", layout="wide")
st.title("CT RISE – Grant Fit Analyzer (persistent)")
st.markdown(f"**Mission:** {MISSION}")

if "tbl" not in st.session_state:
    st.session_state["tbl"] = load_history()

url = st.text_input("Paste a grant application URL")

if st.button("Analyze Grant") and url.strip():
    with st.spinner("Analyzing…"):
        g = scrape_grant(url.strip())
        if not g:
            st.error("Could not parse that URL.")
        elif not future_deadline(g["deadline"]):
            st.warning("Deadline already passed – skipped.")
        else:
            df = st.session_state["tbl"]
            if ((df["URL"].str.lower() == g["url"].lower()).any() or
                (df["Title"].str.lower() == g["title"].lower()).any()):
                st.info("Grant already in the table.")
            else:
                sim = cosine_similarity(
                    [get_embed(g["summary"])],
                    [get_embed(MISSION)]
                )[0][0]*100
                rec_prompt = (
                    f'Mission: "{MISSION}"\n\nGrant: "{g["title"]}" – {g["summary"]}\n'
                    "In 1-2 sentences, say if this is a strong fit and why."
                )
                rec = chat(CHAT_MODEL,[{"role":"user","content":rec_prompt}])\
                      .choices[0].message.content.strip()

                new_row = {
                    "Title": g["title"],
                    "Match%": round(sim,1),
                    "Amount": g["amount"],
                    "Deadline": g["deadline"],
                    "Sponsor": g["sponsor"],
                    "Grant Summary": g["summary"],
                    "URL": g["url"],
                    "Recommendation": rec
                }
                st.session_state["tbl"] = pd.concat(
                    [df, pd.DataFrame([new_row])],
                    ignore_index=True
                )
                save_history(st.session_state["tbl"])
                st.success("Grant added and saved!")

st.subheader("Analyzed Grants (saved across sessions)")
st.dataframe(st.session_state["tbl"], use_container_width=True)

# ---------- CLEAR TABLE BUTTON ----------
if st.button("🗑️ Clear table"):
    st.session_state["tbl"] = pd.DataFrame(columns=COLS)
    save_history(st.session_state["tbl"])
    # immediate refresh so user sees empty table without a second click
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()
