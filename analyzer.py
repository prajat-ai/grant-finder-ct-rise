import os, json, re, time, datetime as dt, io
import pandas as pd, streamlit as st, openai
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors

# ─── CONFIG ─────────────────────────────────────────
SEARCH_MODEL = "gpt-4o-mini-search-preview"
CHAT_MODEL   = "gpt-3.5-turbo"
EMB_MODEL    = "text-embedding-ada-002"
CSV_PATH     = "grants_history.csv"
MISSION = (
    "The Connecticut RISE Network empowers public high schools with data-driven strategies "
    "and personalized support to improve student outcomes and promote postsecondary success, "
    "especially for Black, Latinx, and low-income youth."
)

# ─── OPENAI KEY ─────────────────────────────────────
load_dotenv(); openai.api_key = os.getenv("OPENAI_API_KEY")

# ─── RETRY WRAPPERS ─────────────────────────────────
def retry(fn):
    def wrap(*a, **k):
        for i in range(4):
            try: return fn(*a, **k)
            except openai.RateLimitError: time.sleep(2*(i+1))
        st.error("OpenAI rate-limit."); st.stop()
    return wrap

@retry
def chat(model, msgs): return openai.chat.completions.create(model=model, messages=msgs)

@retry
def emb(t): return openai.embeddings.create(model=EMB_MODEL, input=t).data[0].embedding

# ─── PERSISTENCE ────────────────────────────────────
def load_hist():
    return pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else pd.DataFrame(columns=[
        "Title","Match%","Amount","Deadline","Sponsor",
        "Grant Summary","URL","Recommendation"
    ])
def save_hist(df): df.to_csv(CSV_PATH, index=False)

# ─── SCRAPE + PARSE ─────────────────────────────────
def scrape(url):
    p=(f"search: Visit {url} and return JSON {{title,sponsor,amount,deadline,summary}}.")
    raw=chat(SEARCH_MODEL,[{"role":"user","content":p}]).choices[0].message.content
    m=re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```",raw,re.S) or re.search(r"(\{.*?\}|\[.*?\])",raw,re.S)
    try:
        obj=json.loads(m.group(1) if m else raw); obj=obj[0] if isinstance(obj,list) else obj
    except Exception: return None
    obj["url"]=url
    return {k:obj.get(k,"N/A") for k in("title","sponsor","amount","deadline","summary","url")}

def future(dl): 
    if dl.lower()=="rolling": return True
    try: return dt.datetime.strptime(dl[:10],"%Y-%m-%d").date()>=dt.date.today()
    except: return False

# ─── PDF EXPORT ────────────────────────────────────
def df_to_pdf(df: pd.DataFrame) -> bytes:
    buff=io.BytesIO()
    doc=SimpleDocTemplate(buff,pagesize=letter)
    data=[df.columns.tolist()]+df.values.tolist()
    table=Table(data,repeatRows=1)
    table.setStyle(TableStyle([
        ('GRID',(0,0),(-1,-1),0.5,colors.grey),
        ('BACKGROUND',(0,0),(-1,0),colors.lightgrey),
        ('FONT', (0,0),(-1,0), "Helvetica-Bold")
    ]))
    doc.build([table])
    return buff.getvalue()

# ─── STREAMLIT UI ───────────────────────────────────
st.set_page_config(page_title="CT RISE Grant Analyzer", layout="wide")
st.title("CT RISE — Grant Fit Analyzer")
st.write("**Mission:**", MISSION)

# init session
if "tbl" not in st.session_state: st.session_state.tbl=load_hist()

url=st.text_input("Paste grant URL")

if st.button("Analyze Grant") and url.strip():
    with st.spinner("Analyzing…"):
        g=scrape(url.strip())
        if not g: st.error("Couldn’t parse URL.")
        elif not future(g["deadline"]): st.warning("Deadline passed — skipped.")
        elif (st.session_state.tbl["URL"].str.lower()==g["url"].lower()).any(): st.info("Already in table.")
        else:
            sim=cosine_similarity([emb(g["summary"])],[emb(MISSION)])[0][0]*100
            rec_prompt=(f'Mission: "{MISSION}"\nGrant: "{g["title"]}" – {g["summary"]}\n'
                        "In 1–2 sentences, say if this is a strong fit and why.")
            rec=chat(CHAT_MODEL,[{"role":"user","content":rec_prompt}]).choices[0].message.content.strip()
            new_row=pd.DataFrame([{
                "Title":g["title"],"Match%":round(sim,1),"Amount":g["amount"],
                "Deadline":g["deadline"],"Sponsor":g["sponsor"],
                "Grant Summary":g["summary"],"URL":g["url"],"Recommendation":rec
            }])
            st.session_state.tbl=pd.concat([st.session_state.tbl,new_row],ignore_index=True)
            save_hist(st.session_state.tbl); st.success("Added & saved!")

st.divider()
st.subheader("Analyzed Grants")

edit_mode=st.toggle("✏️ Edit table")

if edit_mode:
    edited=st.data_editor(st.session_state.tbl, num_rows="dynamic", use_container_width=True,
                          key="editor")
    del_rows=st.multiselect("Select rows to delete (by index)", edited.index.tolist())
    col1,col2=st.columns(2)
    if col1.button("Delete selected"):
        edited=edited.drop(del_rows).reset_index(drop=True)
        st.session_state.tbl=edited; save_hist(edited); st.success("Rows deleted.")
    if col2.button("Save changes"):
        st.session_state.tbl=edited; save_hist(edited); st.success("Table saved.")
    show_df=edited
else:
    show_df=st.session_state.tbl

st.dataframe(show_df,use_container_width=True)

# PDF export
if not show_df.empty:
    pdf_bytes=df_to_pdf(show_df)
    st.download_button("📄 Export table as PDF", data=pdf_bytes,
                       file_name="CT_RISE_grants.pdf", mime="application/pdf")
