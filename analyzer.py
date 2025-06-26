# grant_analyzer.py  – Grant-Fit Dashboard for CT RISE
# • Feasibility (High/Med/Low) column right after Match %
# • Honest 250-word analysis (no “verdict” line) + feasibility noted
# • Analysis downloadable as PDF
# • One-click Clear table

import os, json, re, time, datetime as dt, io
import pandas as pd, streamlit as st, openai
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ───────── CONFIG
SEARCH_MODEL = "gpt-4o-mini-search-preview"
CHAT_MODEL   = "gpt-3.5-turbo"
EMB_MODEL    = "text-embedding-ada-002"
CSV_PATH     = "grants_history.csv"
API_RETRY    = 4
BACKOFF      = 2
COLS = ["Title","Match%","Feasibility","Amount","Deadline","Sponsor",
        "Grant Summary","URL","Recommendation"]

MISSION = (
    "The Connecticut RISE Network empowers public high schools with data-driven strategies "
    "and personalised support to improve student outcomes and promote post-secondary success, "
    "especially for Black, Latinx, and low-income youth."
)

# ───────── OPENAI
load_dotenv(); openai.api_key = os.getenv("OPENAI_API_KEY")

def retry(fn):
    def wrap(*a, **k):
        for i in range(API_RETRY):
            try:   return fn(*a, **k)
            except openai.error.RateLimitError: time.sleep(BACKOFF*(i+1))
        st.error("OpenAI rate-limit; try later."); st.stop()
    return wrap

@retry
def chat(model,msgs,**kw): return openai.chat.completions.create(model=model,messages=msgs,**kw)
@retry
def embed(txt):             return openai.embeddings.create(model=EMB_MODEL,input=txt).data[0].embedding

# ───────── CSV I/O
def load_hist():  return pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else pd.DataFrame(columns=COLS)
def save_hist(df): df.to_csv(CSV_PATH,index=False)

# ───────── SCRAPE
def scrape(url:str):
    prm=(f"search: Visit {url} and return JSON with keys "
         "{title,sponsor,amount,deadline (YYYY-MM-DD or 'rolling'), summary}. "
         "Return ONLY JSON; use 'N/A' for unknown.")
    raw=chat(SEARCH_MODEL,[{"role":"user","content":prm}]).choices[0].message.content
    m=re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```",raw,re.S) or re.search(r"(\{.*?\}|\[.*?\])",raw,re.S)
    if not m: return None
    obj=json.loads(m.group(1)); obj=obj[0] if isinstance(obj,list) else obj
    obj["url"]=url
    return {k:obj.get(k,"N/A") for k in ("title","sponsor","amount","deadline","summary","url")}

def deadline_ok(dl:str):
    if dl.lower()=="rolling": return True
    try: return dt.datetime.strptime(dl[:10],"%Y-%m-%d").date()>=dt.date.today()
    except: return False

# ───────── PDF MAKER
def analysis_pdf(title:str,text:str)->bytes:
    buf=io.BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=letter,leftMargin=40,rightMargin=40,topMargin=40,bottomMargin=40)
    sty=getSampleStyleSheet()
    story=[Paragraph(f"<b>{title}</b>",sty["Title"]),Spacer(1,12),
           Paragraph(text.replace("\n","<br/>"),sty["BodyText"])]
    doc.build(story); return buf.getvalue()

def feasibility(match:float)->str:
    return "High" if match>=75 else "Medium" if match>=50 else "Low"

# ───────── UI
st.set_page_config("CT RISE Grant Analyzer",layout="wide")
st.title("CT RISE — Grant Fit Analyzer")
st.write("**Mission:**",MISSION)

if "tbl" not in st.session_state:           st.session_state.tbl=load_hist()
if "latest_title" not in st.session_state:  st.session_state.latest_title=None
if "latest_report" not in st.session_state: st.session_state.latest_report=None
if "latest_pdf" not in st.session_state:    st.session_state.latest_pdf=None

url=st.text_input("Paste grant application URL")

# --- Analyze button
if st.button("Analyze Grant") and url.strip():
    with st.spinner("Analyzing…"):
        g=scrape(url.strip())
        if not g: st.error("Could not parse that URL.")
        elif not deadline_ok(g["deadline"]): st.warning("Deadline passed — skipped.")
        else:
            df=st.session_state.tbl
            if ((df["URL"].str.lower()==g["url"].lower()).any() or
                (df["Title"].str.lower()==g["title"].lower()).any()):
                st.info("Grant already in table.")
            else:
                mscore=cosine_similarity([embed(g["summary"])],[embed(MISSION)])[0][0]*100
                feas=feasibility(mscore)
                short=chat(CHAT_MODEL,[{"role":"user",
                    "content":f'One sentence: why is "{g["title"]}" a fit (or not) for {MISSION}?'}],
                    temperature=0.3).choices[0].message.content.strip()
                # detailed analysis (no verdict line)
                long_prompt=(
                    f"You are an objective grant advisor.\n\nMission:\n{MISSION}\n\n"
                    f"Grant details:\nTitle: {g['title']}\nSponsor: {g['sponsor']}\n"
                    f"Amount: {g['amount']}\nDeadline: {g['deadline']}\nSummary: {g['summary']}\n\n"
                    "Write ~250 words covering:\n"
                    "1. Alignment with mission & population\n"
                    "2. Strengths/opportunities\n"
                    "3. Gaps/disqualifiers (be blunt)\n"
                    f"4. Feasibility rating (High/Medium/Low) — use '{feas}'."
                )
                full=chat(CHAT_MODEL,[{"role":"user","content":long_prompt}],temperature=0.7)\
                     .choices[0].message.content.strip()
                pdf=analysis_pdf(g["title"],full)
                new=pd.DataFrame([{
                    "Title":g["title"],"Match%":round(mscore,1),"Feasibility":feas,
                    "Amount":g["amount"],"Deadline":g["deadline"],"Sponsor":g["sponsor"],
                    "Grant Summary":g["summary"],"URL":g["url"],"Recommendation":short
                }])
                st.session_state.tbl=pd.concat([df,new],ignore_index=True)\
                                         .sort_values("Match%",ascending=False,ignore_index=True)
                save_hist(st.session_state.tbl)
                st.session_state.latest_title=g["title"]
                st.session_state.latest_report=full
                st.session_state.latest_pdf=pdf
                st.success("Grant added & analysis ready!")

# --- Latest analysis
if st.session_state.latest_report:
    st.subheader(f"Detailed Analysis — {st.session_state.latest_title}")
    st.write(st.session_state.latest_report)
    st.download_button("Download analysis (PDF)",
                       st.session_state.latest_pdf,
                       f"{st.session_state.latest_title}_analysis.pdf",
                       mime="application/pdf")

# --- Table
st.subheader("Analyzed Grants (saved across sessions)")
st.dataframe(st.session_state.tbl,use_container_width=True)

# --- Clear table
if st.button("🗑️ Clear table"):
    st.session_state.tbl=pd.DataFrame(columns=COLS)
    save_hist(st.session_state.tbl)
    st.session_state.latest_title=st.session_state.latest_report=st.session_state.latest_pdf=None
    st.rerun()
