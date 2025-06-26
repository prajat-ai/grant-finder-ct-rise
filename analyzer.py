import os, json, re, time, io, datetime as dt
import pandas as pd, streamlit as st, openai
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from reportlab.lib.pagesizes import landscape, letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors

# ── CONFIG ─────────────────────────────────────────
SEARCH_MODEL = "gpt-4o-mini-search-preview"
CHAT_MODEL   = "gpt-3.5-turbo"
EMB_MODEL    = "text-embedding-ada-002"
CSV_PATH     = "grants_history.csv"
API_RETRY    = 4
BACKOFF      = 2
MISSION = (
    "The Connecticut RISE Network empowers public high schools with data-driven strategies "
    "and personalized support to improve student outcomes and promote postsecondary success, "
    "especially for Black, Latinx, and low-income youth."
)

# ── OPENAI KEY ─────────────────────────────────────
load_dotenv(); openai.api_key = os.getenv("OPENAI_API_KEY")

# ── RETRY DECORATOR ────────────────────────────────
def retry(fn):
    def wrap(*a, **k):
        for i in range(API_RETRY):
            try: return fn(*a, **k)
            except openai.RateLimitError: time.sleep(BACKOFF*(i+1))
        st.error("OpenAI rate-limit; try later."); st.stop()
    return wrap
@retry
def chat(model, msgs): return openai.chat.completions.create(model=model, messages=msgs)
@retry
def emb(txt): return openai.embeddings.create(model=EMB_MODEL,input=txt).data[0].embedding

# ── PERSISTENCE ────────────────────────────────────
COLS = ["Title","Match%","Amount","Deadline","Sponsor","Grant Summary","URL","Recommendation"]
def load_hist():
    df = pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else pd.DataFrame(columns=COLS)
    return df
def save_hist(df): df.to_csv(CSV_PATH, index=False)

# ── GRANT SCRAPER ──────────────────────────────────
def scrape(url:str):
    prompt=(f"search: Visit {url} and return JSON {{title,sponsor,amount,deadline,summary}}. "
            "If unknown put 'N/A'.")
    raw=chat(SEARCH_MODEL,[{"role":"user","content":prompt}]).choices[0].message.content
    m=re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```",raw,re.S) or re.search(r"(\{.*?\}|\[.*?\])",raw,re.S)
    try:
        data=json.loads(m.group(1) if m else raw); data=data[0] if isinstance(data,list) else data
    except Exception: return None
    data["url"]=url
    return {k:data.get(k,"N/A") for k in ("title","sponsor","amount","deadline","summary","url")}
def future(dl): 
    if dl.lower()=="rolling": return True
    try: return dt.datetime.strptime(dl[:10],"%Y-%m-%d").date()>=dt.date.today()
    except Exception: return False

# ── PDF EXPORT ─────────────────────────────────────
def df_to_pdf(df:pd.DataFrame)->bytes:
    buf=io.BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=landscape(letter),leftMargin=20,rightMargin=20)
    data=[["#"]+df.columns.tolist()] + [[idx+1,*row] for idx,row in df.iterrows()]
    table=Table(data,repeatRows=1,colWidths=None)
    table.setStyle(TableStyle([
        ('GRID',(0,0),(-1,-1),0.4,colors.grey),
        ('BACKGROUND',(0,0),(-1,0),colors.lightgrey),
        ('FONT',(0,0),(-1,0),"Helvetica-Bold")
    ]))
    doc.build([table]); return buf.getvalue()

# ── STREAMLIT UI ───────────────────────────────────
st.set_page_config("CT RISE Grant Analyzer",layout="wide")
st.title("CT RISE — Grant Fit Analyzer  (persistent & editable)")
st.write(f"**Mission:** {MISSION}")

# init session state
if "tbl" not in st.session_state: st.session_state.tbl=load_hist()

url=st.text_input("Paste grant Application URL")
if st.button("Analyze Grant") and url.strip():
    with st.spinner("Analyzing…"):
        g=scrape(url.strip())
        if not g: st.error("Couldn’t parse URL.")
        elif not future(g["deadline"]): st.warning("Deadline passed — skipped.")
        else:
            df=st.session_state.tbl
            if (df["URL"].str.lower()==g["url"].lower()).any() or \
               (df["Title"].str.lower()==g["title"].lower()).any():
                st.info("Grant already in table.")
            else:
                sim=cosine_similarity([emb(g["summary"])],[emb(MISSION)])[0][0]*100
                rec_prompt=(f'Mission: "{MISSION}"\nGrant: "{g["title"]}" – {g["summary"]}\n'
                            "In 1–2 sentences explain fit.")
                rec=chat(CHAT_MODEL,[{"role":"user","content":rec_prompt}]).choices[0].message.content.strip()
                new_row=pd.DataFrame([{
                    "Title":g["title"],"Match%":round(sim,1),"Amount":g["amount"],
                    "Deadline":g["deadline"],"Sponsor":g["sponsor"],
                    "Grant Summary":g["summary"],"URL":g["url"],"Recommendation":rec
                }])
                st.session_state.tbl=pd.concat([df,new_row],ignore_index=True)\
                                        .sort_values("Match%",ascending=False,ignore_index=True)
                save_hist(st.session_state.tbl)
                st.success("Added & saved!")

st.divider()
st.subheader("Analyzed Grants")

edit=st.toggle("✏️ Enter edit mode")
df_display=st.session_state.tbl.copy()
df_display.index+=1   # start at 1 for display

if edit:
    df_display["Delete?"]=False
    edited=st.data_editor(df_display,use_container_width=True,num_rows="dynamic",key="editor")
    del_rows=edited[edited["Delete?"]].index
    col1,col2=st.columns(2)
    if col1.button("Delete selected rows"):
        edited=edited.drop(del_rows).drop(columns="Delete?").reset_index(drop=True)
        st.session_state.tbl=edited.sort_values("Match%",ascending=False,ignore_index=True)
        save_hist(st.session_state.tbl); st.success("Rows deleted & saved.")
    if col2.button("Save edits"):
        edited=edited.drop(columns="Delete?").reset_index(drop=True)
        st.session_state.tbl=edited.sort_values("Match%",ascending=False,ignore_index=True)
        save_hist(st.session_state.tbl); st.success("Edits saved.")
    show_df=edited.drop(columns="Delete?")
else:
    show_df=df_display

st.dataframe(show_df,use_container_width=True)

if not show_df.empty:
    pdf_bytes=df_to_pdf(show_df)
    st.download_button("📄 Export as PDF",data=pdf_bytes,file_name="CT_RISE_grants.pdf",
                       mime="application/pdf")
