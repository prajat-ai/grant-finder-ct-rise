# grant_analyzer.py  –  CT RISE persistent dashboard (edit-mode fix)

import os, json, re, time, io, datetime as dt
import pandas as pd, streamlit as st, openai
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from reportlab.lib.pagesizes import landscape, letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors

# ── CONFIG ─────────────────────
SEARCH_MODEL="gpt-4o-mini-search-preview"
CHAT_MODEL  ="gpt-3.5-turbo"
EMB_MODEL   ="text-embedding-ada-002"
CSV_PATH    ="grants_history.csv"
MISSION=(
  "The Connecticut RISE Network empowers public high schools with data-driven strategies "
  "and personalized support to improve student outcomes and promote postsecondary success, "
  "especially for Black, Latinx, and low-income youth."
)

# ── KEYS ───────────────────────
load_dotenv(); openai.api_key=os.getenv("OPENAI_API_KEY")

def retry(fn):
  def wrap(*a,**k):
    for i in range(4):
      try:return fn(*a,**k)
      except openai.RateLimitError: time.sleep(2*(i+1))
    st.error("OpenAI rate-limit."); st.stop()
  return wrap
@retry
def chat(m,msg): return openai.chat.completions.create(model=m,messages=msg)
@retry
def emb(t):      return openai.embeddings.create(model=EMB_MODEL,input=t).data[0].embedding

COLS=["Title","Match%","Amount","Deadline","Sponsor",
      "Grant Summary","URL","Recommendation"]

def load_hist(): return pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else pd.DataFrame(columns=COLS)
def save_hist(df): df.to_csv(CSV_PATH,index=False)

def scrape(u):
  p=(f"search: Visit {u} and return JSON {{title,sponsor,amount,deadline,summary}} "
     "or 'N/A' for unknown.")
  raw=chat(SEARCH_MODEL,[{"role":"user","content":p}]).choices[0].message.content
  m=re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```",raw,re.S) or re.search(r"(\{.*?\}|\[.*?\])",raw,re.S)
  try:d=json.loads(m.group(1) if m else raw); d=d[0] if isinstance(d,list) else d
  except: return None
  d["url"]=u
  return {k:d.get(k,"N/A") for k in("title","sponsor","amount","deadline","summary","url")}
def future(dl):
  if dl.lower()=="rolling":return True
  try:return dt.datetime.strptime(dl[:10],"%Y-%m-%d").date()>=dt.date.today()
  except:return False

def df_to_pdf(df):
  buf=io.BytesIO()
  doc=SimpleDocTemplate(buf,pagesize=landscape(letter),leftMargin=20,rightMargin=20)
  data=[["#"]+df.columns.tolist()]+[[i+1,*row] for i,row in df.iterrows()]
  t=Table(data,repeatRows=1); t.setStyle(TableStyle([
    ('GRID',(0,0),(-1,-1),0.4,colors.grey),
    ('BACKGROUND',(0,0),(-1,0),colors.lightgrey),
    ('FONT',(0,0),(-1,0),"Helvetica-Bold")
  ]))
  doc.build([t]); return buf.getvalue()

# ── APP ────────────────────────
st.set_page_config("CT RISE Grant Analyzer",layout="wide")
st.title("CT RISE — Grant Fit Analyzer (persistent)")
st.write("**Mission:**",MISSION)

if "tbl" not in st.session_state: st.session_state.tbl=load_hist()
url=st.text_input("Paste grant URL")

if st.button("Analyze Grant") and url.strip():
  with st.spinner("Analyzing…"):
    g=scrape(url.strip())
    if not g: st.error("Couldn’t parse URL.")
    elif not future(g["deadline"]): st.warning("Deadline passed — skipped.")
    else:
      df=st.session_state.tbl
      if (df["URL"].str.lower()==g["url"].lower()).any() or \
         (df["Title"].str.lower()==g["title"].lower()).any():
        st.info("Already in table.")
      else:
        sim=cosine_similarity([emb(g["summary"])],[emb(MISSION)])[0][0]*100
        rec=chat(CHAT_MODEL,[{"role":"user","content":
              f'Mission: "{MISSION}"\nGrant: "{g["title"]}" – {g["summary"]}\n'
              "In 1–2 sentences explain fit."}]).choices[0].message.content.strip()
        newrow=pd.DataFrame([{
          "Title":g["title"],"Match%":round(sim,1),"Amount":g["amount"],
          "Deadline":g["deadline"],"Sponsor":g["sponsor"],
          "Grant Summary":g["summary"],"URL":g["url"],"Recommendation":rec
        }])
        st.session_state.tbl=pd.concat([df,newrow],ignore_index=True)\
                                .sort_values("Match%",ascending=False,ignore_index=True)
        save_hist(st.session_state.tbl); st.success("Added & saved!")

st.divider(); st.subheader("Analyzed Grants")
edit=st.toggle("✏️ Edit mode")
df_disp=st.session_state.tbl.copy(); df_disp.index+=1

if edit:
  df_disp["Delete?"]=False
  edited=st.data_editor(df_disp,use_container_width=True,num_rows="dynamic",key="editor")
  del_rows=edited[edited.get("Delete?",False)].index if "Delete?" in edited else []
  col1,col2=st.columns(2)
  if col1.button("Delete selected"):
    edited=edited.drop(del_rows).drop(columns="Delete?",errors="ignore").reset_index(drop=True)
    st.session_state.tbl=edited.sort_values("Match%",ascending=False,ignore_index=True)
    save_hist(st.session_state.tbl); st.success("Rows deleted & saved.")
  if col2.button("Save edits"):
    edited=edited.drop(columns="Delete?",errors="ignore").reset_index(drop=True)
    st.session_state.tbl=edited.sort_values("Match%",ascending=False,ignore_index=True)
    save_hist(st.session_state.tbl); st.success("Edits saved.")
  show=edited.drop(columns="Delete?",errors="ignore")
else:
  show=df_disp

st.dataframe(show,use_container_width=True)

if not show.empty:
  st.download_button("📄 Export as PDF",data=df_to_pdf(show),
                     file_name="CT_RISE_grants.pdf",mime="application/pdf")
