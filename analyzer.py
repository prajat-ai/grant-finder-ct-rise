# grant_analyzer.py  –  Persistent Grant-Fit Dashboard for CT RISE
# Adds: in-depth analysis of the most-recently added grant + download button.

import os, json, re, time, datetime as dt, io
import pandas as pd, streamlit as st, openai
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# ── CONFIG ─────────────────────────────────────────────
SEARCH_MODEL = "gpt-4o-mini-search-preview"
CHAT_MODEL   = "gpt-3.5-turbo"
EMB_MODEL    = "text-embedding-ada-002"
CSV_PATH     = "grants_history.csv"
API_RETRY    = 4
BACKOFF      = 2
COLS = ["Title","Match%","Amount","Deadline","Sponsor",
        "Grant Summary","URL","Recommendation"]
MISSION = (
    "The Connecticut RISE Network empowers public high schools with data-driven strategies "
    "and personalized support to improve student outcomes and promote post-secondary success, "
    "especially for Black, Latinx, and low-income youth."
)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ── RETRY DECORATOR ──────────────────────────────────
def retry(fn):
    def wrap(*a, **k):
        for i in range(API_RETRY):
            try: return fn(*a, **k)
            except openai.error.RateLimitError: time.sleep(BACKOFF*(i+1))
        st.error("OpenAI rate-limit; try later."); st.stop()
    return wrap

@retry
def chat(model, msgs): return openai.chat.completions.create(model=model, messages=msgs)

@retry
def embed(text): return openai.embeddings.create(model=EMB_MODEL, input=text).data[0].embedding

def load_history():  return pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else pd.DataFrame(columns=COLS)
def save_history(df): df.to_csv(CSV_PATH, index=False)

# ── QUICK GRANT SCRAPER ──────────────────────────────
def scrape_grant(url:str):
    prompt = (f"search: Visit {url} and return JSON with keys "
              "{title,sponsor,amount,deadline (YYYY-MM-DD or 'rolling'), summary}. "
              "Use 'N/A' for unknown. Respond ONLY with JSON.")
    raw = chat(SEARCH_MODEL,[{"role":"user","content":prompt}]).choices[0].message.content
    m = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", raw, re.S) or re.search(r"(\{.*?\}|\[.*?\])", raw, re.S)
    if not m: return None
    data = json.loads(m.group(1))
    if isinstance(data, list): data = data[0]
    data["url"] = url
    return {k: data.get(k,"N/A") for k in ("title","sponsor","amount","deadline","summary","url")}

def deadline_ok(dl:str):
    if dl.lower()=="rolling": return True
    try: return dt.datetime.strptime(dl[:10],"%Y-%m-%d").date() >= dt.date.today()
    except: return False

def sort_save(df): df.sort_values("Match%", ascending=False, ignore_index=True).to_csv(CSV_PATH, index=False)

# ── STREAMLIT UI ─────────────────────────────────────
st.set_page_config("CT RISE Grant Analyzer", layout="wide")
st.title("CT RISE — Grant Fit Analyzer (persistent)")
st.write("**Mission:**", MISSION)

if "tbl" not in st.session_state: st.session_state.tbl = load_history()

url = st.text_input("Paste grant application URL")

# ---------- ANALYZE BUTTON ----------
if st.button("Analyze Grant") and url.strip():
    with st.spinner("Analyzing…"):
        g = scrape_grant(url.strip())
        if not g:
            st.error("Could not parse that URL.")
        elif not deadline_ok(g["deadline"]):
            st.warning("Deadline passed — skipped.")
        else:
            df = st.session_state.tbl
            if ((df["URL"].str.lower()==g["url"].lower()).any() or
                (df["Title"].str.lower()==g["title"].lower()).any()):
                st.info("Grant already in table.")
            else:
                sim = cosine_similarity([embed(g["summary"])],[embed(MISSION)])[0][0]*100
                # short rec for table
                rec_short = chat(CHAT_MODEL,[{
                    "role":"user",
                    "content":f'In one sentence, why is "{g["title"]}" a fit (or not) for this mission: {MISSION}'
                }]).choices[0].message.content.strip()
                # detailed analysis
                analysis_prompt = (
                    f'Mission: "{MISSION}"\n\nGrant details:\nTitle: {g["title"]}\n'
                    f'Sponsor: {g["sponsor"]}\nAmount: {g["amount"]}\nDeadline: {g["deadline"]}\n'
                    f'Summary: {g["summary"]}\n\nWrite a concise (~250-word) analysis covering:\n'
                    "- Alignment with mission and underserved populations\n"
                    "- Strengths / opportunities\n- Potential challenges or missing criteria\n"
                    "- Overall recommendation"
                )
                full_report = chat(CHAT_MODEL,[{"role":"user","content":analysis_prompt}])\
                              .choices[0].message.content.strip()
                # save table row
                new = pd.DataFrame([{
                    "Title": g["title"], "Match%": round(sim,1), "Amount": g["amount"],
                    "Deadline": g["deadline"], "Sponsor": g["sponsor"],
                    "Grant Summary": g["summary"], "URL": g["url"], "Recommendation": rec_short
                }])
                st.session_state.tbl = pd.concat([df,new], ignore_index=True)
                st.session_state.tbl = st.session_state.tbl.sort_values("Match%",ascending=False,ignore_index=True)
                save_history(st.session_state.tbl)
                # store latest report in session for display & download
                st.session_state.latest_report = full_report
                st.session_state.latest_title  = g["title"]
                st.success("Grant added, table saved, analysis generated!")

# ---------- SHOW LATEST ANALYSIS ----------
if "latest_report" in st.session_state:
    st.subheader(f"Detailed Analysis: {st.session_state.latest_title}")
    st.write(st.session_state.latest_report)
    st.download_button("Download analysis (.txt)",
                       data=st.session_state.latest_report.encode(),
                       file_name=f"{st.session_state.latest_title}_analysis.txt",
                       mime="text/plain")

# ---------- TABLE ----------
st.subheader("Analyzed Grants (saved across sessions)")
st.dataframe(st.session_state.tbl, use_container_width=True)

# ---------- CLEAR TABLE ----------
if st.button("🗑️ Clear table"):
    st.session_state.tbl = pd.DataFrame(columns=COLS)
    save_history(st.session_state.tbl)
    st.session_state.pop("latest_report", None)
    st.session_state.pop("latest_title",  None)
    st.success("Table cleared.")
