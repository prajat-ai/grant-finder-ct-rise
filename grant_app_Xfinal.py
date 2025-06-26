# CT RISE – Smart Grant Finder  v4-retry
# GPT asked for 12 grants, retries up to 3× until JSON parses.

import os, time, json, random, re
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import openai

# ------------ CONFIG ------------
NUM_GRANTS      = 12         # lower token usage
TOP_N           = 8
BASE_DELAY      = 2          # seconds for back-off
OPENAI_RETRIES  = 5
PROMPT_RETRIES  = 3
CHAT_MODEL      = "gpt-3.5-turbo"
EMBED_MODEL     = "text-embedding-ada-002"

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

MISSION = ("The Connecticut RISE Network empowers public high schools with "
           "data-driven strategies and personalized support to improve student outcomes "
           "and promote postsecondary success, especially for Black, Latinx, "
           "and low-income youth.")

# ------------ OpenAI helpers ------------
def openai_chat(msgs, maxtok=800):
    for a in range(OPENAI_RETRIES):
        try:
            resp = openai.ChatCompletion.create(
                model=CHAT_MODEL,
                messages=msgs,
                max_tokens=maxtok,
                temperature=0.7,
            )
            return resp.choices[0].message.content
        except openai.error.RateLimitError:
            time.sleep(BASE_DELAY * 2 ** a + random.uniform(0, 1))
    st.error("Still rate-limited after several tries."); st.stop()

def embed(txt: str):
    for a in range(OPENAI_RETRIES):
        try:
            return openai.Embedding.create(input=txt, model=EMBED_MODEL)["data"][0]["embedding"]
        except openai.error.RateLimitError:
            time.sleep(BASE_DELAY * 2 ** a)
    return [0.0] * 1536

# ------------ 1 · get grants list (retry until parses) ----------
def get_grants_json():
    sys = {"role": "system", "content": "You are a concise grants researcher."}

    for attempt in range(PROMPT_RETRIES):
        user = {
            "role": "user",
            "content": (
                f"Provide exactly {NUM_GRANTS} CURRENT (2024-2025) U.S. grant opportunities for nonprofits "
                "focused on high-school education, college readiness, or youth equity. "
                "Return **nothing except** a valid JSON array; each object must have keys:\n"
                "title, sponsor, amount, summary, deadline, url.\n\n"
                "Example of one element:\n"
                '{"title":"High School Success Fund","sponsor":"Acme Foundation",'
                '"amount":"$50,000","summary":"Funds programs…","deadline":"2025-02-28","url":"https://example.com"}'
            ),
        }

        raw = openai_chat([sys, user], maxtok=900)

        # try direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # try to pull first [...] block
            match = re.search(r"\[[\s\S]*\]", raw)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    pass
            # give GPT feedback and retry
            sys["content"] = "Your previous response was not valid JSON. Return ONLY a JSON array."
    return []   # all retries failed

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_grants():
    return get_grants_json()[:NUM_GRANTS]

# ------------ 2 · rank + feasibility ---------------------------
def analyse(js):
    if not js:
        return pd.DataFrame()
    df = pd.DataFrame(js)
    mvec = embed(MISSION)
    df["sim"] = [float(cosine_similarity([embed(s)], [mvec])[0][0]) for s in df.summary]

    df = df.sort_values("sim", ascending=False).head(TOP_N).reset_index(drop=True)
    feas, why = [], []
    for _, row in df.iterrows():
        prompt = (
            f"Mission: {MISSION}\n\n"
            f"Grant: {row.title} – {row.summary}\n\n"
            "Answer ONLY: {\"feasibility\":\"High|Medium|Low\",\"why\":\"<one sentence>\"}"
        )
        try:
            j = json.loads(openai_chat([{"role":"user","content":prompt}], maxtok=60))
            feas.append(j.get("feasibility", "?")); why.append(j.get("why", ""))
        except Exception:
            feas.append("?"); why.append("parse error")
    df["feasibility"], df["why_fit"] = feas, why
    return df

# ------------ UI -----------------------------------------------
st.title("CT RISE Network — Smart Grant Finder (v4-retry)")
st.write("> **Mission:**", MISSION)

if st.button("🔄 Generate grants (retry-safe)", type="primary"):
    with st.spinner("GPT compiling grants… ~1 min"):
        tbl = analyse(fetch_grants())
        st.session_state["tbl"] = tbl
        st.success("Process finished!")

if "tbl" in st.session_state and not st.session_state["tbl"].empty:
    st.dataframe(
        st.session_state["tbl"][
            ["title", "sponsor", "amount", "deadline", "sim",
             "feasibility", "why_fit", "url"]
        ],
        use_container_width=True,
    )
elif "tbl" in st.session_state:
    st.info("Still couldn’t parse grants after retries — click again later.")
else:
    st.caption("Build tag: v4-retry — press the button to start.")
