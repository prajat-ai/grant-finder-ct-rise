# grant_analyzer.py  – Persistent Grant-Fit Dashboard for CT RISE  
# • Keeps history in grants_history.csv  
# • Adds one-click “Clear table”  
# • Generates a frank ±250-word analysis of the **most-recent** grant and lets
#   the user download it as a .txt file

import os, json, re, time, datetime as dt, io
import pandas as pd, streamlit as st, openai
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# ───────── CONFIG
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
    "and personalised support to improve student outcomes and promote post-secondary success, "
    "especially for Black, Latinx, and low-income youth."
)

# ───────── OPENAI KEY
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ───────── RETRY WRAPPER
def retry(fn):
    def wrapper(*a, **k):
        for i in range(API_RETRY):
            try:   return fn(*a, **k)
            except openai.error.RateLimitError: time.sleep(BACKOFF*(i+1))
        st.error("OpenAI rate-limit; try again later."); st.stop()
    return wrapper

@retry
def chat(model, msgs, **kw): return openai.chat.completions.create(model=model, messages=msgs, **kw)
@retry
def embed(txt):              return openai.embeddings.create(model=EMB_MODEL, input=txt).data[0].embedding

# ───────── CSV PERSISTENCE
def load_history():  return pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else pd.DataFrame(columns=COLS)
def save_history(df): df.to_csv(CSV_PATH, index=False)

# ───────── GRANT SCRAPER
def scrape_grant(url:str):
    prmpt = (f"search: Visit {url} and return JSON with keys "
             "{title,sponsor,amount,deadline (YYYY-MM-DD or 'rolling'), summary}. "
             "If unknown, use 'N/A'. Respond ONLY with JSON.")
    raw   = chat(SEARCH_MODEL,[{"role":"user","content":prmpt}]).choices[0].message.content
    m     = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", raw, re.S) or re.search(r"(\{.*?\}|\[.*?\])", raw, re.S)
    if not m: return None
    data  = json.loads(m.group(1))
    if isinstance(data, list): data = data[0]
    data["url"] = url
    return {k:data.get(k,"N/A") for k in ("title","sponsor","amount","deadline","summary","url")}

def deadline_ok(dl:str):
    if dl.lower() == "rolling": return True
    try: return dt.datetime.strptime(dl[:10],"%Y-%m-%d").date() >= dt.date.today()
    except: return False

# ───────── STREAMLIT UI
st.set_page_config("CT RISE Grant Analyzer", layout="wide")
st.title("CT RISE — Grant Fit Analyzer")
st.write("**Mission:**", MISSION)

if "tbl" not in st.session_state:    st.session_state.tbl = load_history()
if "latest_report" not in st.session_state: st.session_state.latest_report = None
if "latest_title"  not in st.session_state: st.session_state.latest_title  = None

url = st.text_input("Paste grant application URL")

# ---------- ANALYZE BUTTON ----------
if st.button("Analyze Grant") and url.strip():
    with st.spinner("Analyzing…"):
        g = scrape_grant(url.strip())
        if not g:
            st.error("Could not parse that URL.")
        elif not deadline_ok(g["deadline"]):
            st.warning("Deadline already passed — skipped.")
        else:
            df = st.session_state.tbl
            if ((df["URL"].str.lower()==g["url"].lower()).any() or
                (df["Title"].str.lower()==g["title"].lower()).any()):
                st.info("Grant already in table.")
            else:
                sim = cosine_similarity([embed(g["summary"])],[embed(MISSION)])[0][0]*100
                # short one-sentence summary for table
                short = chat(
                    CHAT_MODEL,
                    [{"role":"user",
                      "content":f'One sentence: why is "{g["title"]}" a fit—or not—for {MISSION}?'}],
                    temperature=0.3).choices[0].message.content.strip()
                # honest full analysis
                analysis_prompt = (
                    f"You are an objective grant advisor.\n\nMission:\n{MISSION}\n\n"
                    f"Grant details:\n"
                    f"- Title: {g['title']}\n- Sponsor: {g['sponsor']}\n"
                    f"- Amount: {g['amount']}\n- Deadline: {g['deadline']}\n"
                    f"- Summary: {g['summary']}\n\n"
                    "Write ±250 words assessing TRUE fit. Be blunt if fit is weak. Cover:\n"
                    "1. Alignment with mission & population\n2. Strengths/opportunities\n"
                    "3. Major gaps/disqualifiers\n4. Verdict: **Strong / Moderate / Poor fit**."
                )
                full = chat(CHAT_MODEL,[{"role":"user","content":analysis_prompt}],temperature=0.7)\
                       .choices[0].message.content.strip()
                new = pd.DataFrame([{
                    "Title": g["title"], "Match%": round(sim,1), "Amount": g["amount"],
                    "Deadline": g["deadline"], "Sponsor": g["sponsor"],
                    "Grant Summary": g["summary"], "URL": g["url"], "Recommendation": short
                }])
                st.session_state.tbl = pd.concat([df,new], ignore_index=True)\
                                          .sort_values("Match%",ascending=False,ignore_index=True)
                save_history(st.session_state.tbl)
                st.session_state.latest_report = full
                st.session_state.latest_title  = g["title"]
                st.success("Grant added & detailed analysis generated!")

# ---------- SHOW LATEST ANALYSIS ----------
if st.session_state.latest_report:
    st.subheader(f"Detailed Analysis — {st.session_state.latest_title}")
    st.write(st.session_state.latest_report)
    st.download_button(
        "Download analysis (.txt)",
        data=st.session_state.latest_report.encode(),
        file_name=f"{st.session_state.latest_title}_analysis.txt",
        mime="text/plain"
    )

# ---------- TABLE ----------
st.subheader("Analyzed Grants (saved across sessions)")
st.dataframe(st.session_state.tbl, use_container_width=True)

# ---------- CLEAR TABLE ----------
if st.button("🗑️ Clear table"):
    st.session_state.tbl = pd.DataFrame(columns=COLS)
    save_history(st.session_state.tbl)
    st.session_state.latest_report = None
    st.session_state.latest_title  = None
    st.rerun()          # refresh immediately
