import os, json, re, time, io, datetime as dt
import pandas as pd, streamlit as st, openai
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from reportlab.lib.pagesizes import landscape, letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors

# ───────────────────── CONFIG
SEARCH_MODEL = "gpt-4o-mini-search-preview"
CHAT_MODEL   = "gpt-3.5-turbo"
EMB_MODEL    = "text-embedding-ada-002"
CSV_PATH     = "grants_history.csv"
COLS = ["Title","Match%","Amount","Deadline","Sponsor",
        "Grant Summary","URL","Recommendation"]

MISSION = (
    "The Connecticut RISE Network empowers public high schools with data-driven strategies "
    "and personalized support to improve student outcomes and promote post-secondary success, "
    "especially for Black, Latinx, and low-income youth."
)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ───────────────────── HELPERS
def retry(fn):
    def wrapper(*a, **k):
        for i in range(4):
            try: return fn(*a, **k)
            except openai.RateLimitError: time.sleep(2*(i+1))
        st.error("OpenAI rate-limited; try again later."); st.stop()
    return wrapper

@retry
def chat(model, msgs): return openai.chat.completions.create(model=model, messages=msgs)

@retry
def embed(txt): return openai.embeddings.create(model=EMB_MODEL, input=txt).data[0].embedding

def load_table(): return pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else pd.DataFrame(columns=COLS)
def save_table(df): df.to_csv(CSV_PATH, index=False)

# ── robust grant scraper (2-step) ──────────────────
def scrape_grant(url: str) -> dict | None:
    base_keys = (
        "{title,sponsor,amount,deadline (YYYY-MM-DD or 'rolling'), summary}. "
        "Use 'N/A' for unknown."
    )
    prompts = [
        f"search: Visit {url} and return JSON with keys {base_keys}",
        f"search: Visit {url} and respond ONLY with a single JSON object with keys {base_keys}"
    ]

    for prmpt in prompts:
        raw = chat(SEARCH_MODEL,[{"role":"user","content":prmpt}]).choices[0].message.content
        raw = re.sub(r"```[\s\S]*?```", "", raw)                         # strip ``` blocks
        m   = re.search(r"(\{.*?\}|\[.*?\])", raw, re.S)                 # non-greedy
        if not m: continue
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, list): obj = obj[0]
            if not isinstance(obj, dict): continue
        except Exception: continue
        obj["url"] = url
        return {k: obj.get(k, "N/A") for k in
                ("title","sponsor","amount","deadline","summary","url")}
    return None

def deadline_ok(dl:str):
    if dl.lower()=="rolling": return True
    try: return dt.datetime.strptime(dl[:10],"%Y-%m-%d").date() >= dt.date.today()
    except: return False

def pdf_bytes_from_df(df):
    buf=io.BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=landscape(letter),leftMargin=20,rightMargin=20)
    data=[["#"]+df.columns.tolist()]+[[i+1]+list(r) for i,r in df.iterrows()]
    t=Table(data,repeatRows=1); t.setStyle(TableStyle([
        ('GRID',(0,0),(-1,-1),0.4,colors.grey),
        ('BACKGROUND',(0,0),(-1,0),colors.lightgrey),
        ('FONT',(0,0),(-1,0),"Helvetica-Bold")]))
    doc.build([t]); return buf.getvalue()

def sort_save():
    st.session_state.table = st.session_state.table.sort_values(
        "Match%", ascending=False, ignore_index=True)
    save_table(st.session_state.table)

# ───────────────────── STREAMLIT
st.set_page_config("CT RISE Grant Analyzer", layout="wide")
st.title("CT RISE — Grant Fit Analyzer (persistent)")
st.write("**Mission:**", MISSION)

if "table" not in st.session_state:
    st.session_state.table = load_table()

# ---------- ADD NEW GRANT ----------
url = st.text_input("Paste grant application URL")
if st.button("Analyze Grant") and url.strip():
    with st.spinner("Analyzing…"):
        g = scrape_grant(url.strip())
        if not g:
            st.error("Could not parse that URL.")
        elif not deadline_ok(g["deadline"]):
            st.warning("Deadline passed — skipped.")
        else:
            df = st.session_state.table
            if ((df["URL"].str.lower()==g["url"].lower()).any() or
                (df["Title"].str.lower()==g["title"].lower()).any()):
                st.info("Grant already in table.")
            else:
                sim = cosine_similarity([embed(g["summary"])],[embed(MISSION)])[0][0]*100
                rec_prompt=(f'Mission: "{MISSION}"\nGrant: "{g["title"]}" – {g["summary"]}\n'
                            "In 1-2 sentences, explain fit.")
                rec = chat(CHAT_MODEL,[{"role":"user","content":rec_prompt}]).choices[0].message.content.strip()
                new=pd.DataFrame([{
                    "Title":g["title"],"Match%":round(sim,1),"Amount":g["amount"],
                    "Deadline":g["deadline"],"Sponsor":g["sponsor"],
                    "Grant Summary":g["summary"],"URL":g["url"],"Recommendation":rec
                }])
                st.session_state.table=pd.concat([df,new], ignore_index=True)
                sort_save(); st.success("Added & saved!")

st.divider(); st.subheader("Analyzed Grants")

# ---------- EDIT MODE ----------
edit_mode = st.toggle("✏️ Edit mode")
df_view   = st.session_state.table.copy()
df_view.index = range(1, len(df_view)+1)   # display index starts at 1

if edit_mode:
    edited = st.data_editor(df_view, use_container_width=True, num_rows="dynamic", key="editor")
    st.session_state.table = edited.reset_index(drop=True)
    sort_save()
    show_df = edited
else:
    show_df = df_view

st.dataframe(show_df, use_container_width=True)

# ---------- PDF EXPORT ----------
if not show_df.empty:
    st.download_button("📄 Export table as PDF",
                       pdf_bytes_from_df(show_df),
                       "CT_RISE_grants.pdf",
                       "application/pdf")
