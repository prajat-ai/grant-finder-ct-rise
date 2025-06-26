# CT RISE – Grant Fit Analyzer (upload-by-URL)

import os, json, re, time, datetime as dt
import pandas as pd, streamlit as st, openai
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# ── CONFIG ─────────────────────────────────────────
SEARCH_MODEL = "gpt-4o-mini-search-preview"    # web-search capable
CHAT_MODEL   = "gpt-3.5-turbo"
EMB_MODEL    = "text-embedding-ada-002"
API_RETRY    = 4
BACKOFF      = 2  # seconds

MISSION = (
    "The Connecticut RISE Network empowers public high schools with data-driven strategies "
    "and personalized support to improve student outcomes and promote post-secondary success, "
    "especially for Black, Latinx, and low-income youth."
)

# ── KEYS ───────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ── HELPERS ───────────────────────────────────────
def retry(fn):
    def wrapper(*a, **k):
        for i in range(API_RETRY):
            try:
                return fn(*a, **k)
            except openai.RateLimitError:
                time.sleep(BACKOFF * (i + 1))
        st.error("OpenAI rate-limit; try again later."); st.stop()
    return wrapper

@retry
def chat(model: str, messages: list[dict]):
    return openai.chat.completions.create(model=model, messages=messages)

@retry
def embed(text: str):
    return openai.embeddings.create(model=EMB_MODEL, input=text).data[0].embedding

# ── GRANT SCRAPE & PARSE ──────────────────────────
def fetch_grant_data(url: str) -> dict | None:
    user_prompt = (
        f"search: Fetch the web page at {url} and extract the following as JSON:\n"
        "{title, sponsor, amount, deadline (YYYY-MM-DD or 'rolling'), summary}. "
        "If a field is unknown write 'N/A'. Respond ONLY with the JSON object."
    )
    txt = chat(SEARCH_MODEL, [{"role": "user", "content": user_prompt}]).choices[0].message.content
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", txt, re.S)
        data = json.loads(m.group()) if m else None
    if not data:
        return None
    data["url"] = url
    return {k: data.get(k, "N/A") for k in
            ("title", "sponsor", "amount", "deadline", "summary", "url")}

def is_future(deadline: str) -> bool:
    if deadline.lower() == "rolling":
        return True
    try:
        return dt.datetime.strptime(deadline[:10], "%Y-%m-%d").date() >= dt.date.today()
    except Exception:
        return False

# ── STREAMLIT UI ──────────────────────────────────
st.title("Grant Fit Analyzer for CT RISE")
st.write("> **Mission:**", MISSION)

# initialize storage
if "table" not in st.session_state:
    st.session_state["table"] = pd.DataFrame(columns=[
        "Title","Match%","Amount","Deadline","Sponsor",
        "Grant Summary","URL","Recommendation"
    ])

url_input = st.text_input("Paste a grant Application URL here and click **Analyze Grant**")
if st.button("Analyze Grant") and url_input.strip():
    with st.spinner("Analyzing grant page…"):
        grant = fetch_grant_data(url_input.strip())
        if not grant:
            st.error("Could not extract info from that URL – try another.")
        elif not is_future(grant["deadline"]):
            st.warning("Deadline has already passed – not added to table.")
        else:
            # compute similarity
            sim = cosine_similarity(
                [embed(grant["summary"])],
                [embed(MISSION)]
            )[0][0] * 100
            # recommendation
            rec_prompt = (
                f'Mission: "{MISSION}"\n\nGrant: "{grant["title"]}" – {grant["summary"]}\n'
                "In 1-2 sentences, explain whether this grant is a strong fit or not."
            )
            rec = chat(CHAT_MODEL, [{"role":"user","content":rec_prompt}]).choices[0].message.content.strip()
            # append to df
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
            st.success("Grant analyzed and added to table!")

st.subheader("Analyzed Grants")
st.dataframe(st.session_state["table"], use_container_width=True)
