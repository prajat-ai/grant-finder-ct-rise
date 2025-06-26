# CT RISE – Smart Grant Finder (v3-grants-amount)
# GPT generates 15 grant opportunities (title, sponsor, amount, summary, deadline, url),
# we rank by mission similarity, add feasibility & rationale, and display in Streamlit.

import os, time, json, random, re
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import openai

# ── CONFIG ──────────────────────────────────────────────────────
NUM, DELAY, RETRIES = 15, 2, 5     # 15 grants, 2-sec base delay
CHAT_MODEL  = "gpt-3.5-turbo"
EMBED_MODEL = "text-embedding-ada-002"

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

MISSION = ("The Connecticut RISE Network empowers public high schools with "
           "data-driven strategies and personalized support to improve student outcomes "
           "and promote postsecondary success, especially for Black, Latinx, "
           "and low-income youth.")

# ── OpenAI wrappers ─────────────────────────────────────────────
def chat(msgs, maxtok=900):
    for a in range(RETRIES):
        try:
            r = openai.ChatCompletion.create(
                model=CHAT_MODEL,
                messages=msgs,
                max_tokens=maxtok,
                temperature=0.7,
            )
            return r.choices[0].message.content
        except openai.error.RateLimitError:
            time.sleep(DELAY * (2**a) + random.uniform(0, 1))
    st.error("OpenAI still rate-limited."); st.stop()

def embed(text):
    for a in range(RETRIES):
        try:
            v = openai.Embedding.create(input=text, model=EMBED_MODEL)
            return v["data"][0]["embedding"]
        except openai.error.RateLimitError:
            time.sleep(DELAY * (2**a))
    return [0.0]*1536

# ── 1 · GPT → grant list  (robust JSON parse) ───────────────────
@st.cache_data(show_spinner=False, ttl=86400)
def fetch_grants():
    sys = {"role":"system","content":"You are a nonprofit grants researcher."}
    usr = {"role":"user","content":(
        f"Provide exactly {NUM} CURRENT (2024-2025) U.S. grant opportunities for nonprofits "
        "focused on high-school education, college readiness, or youth equity. "
        "Return ONLY a JSON array; every element must have keys:\n"
        "title, sponsor, amount (max USD), summary, deadline (YYYY-MM-DD or \"rolling\"), url."
    )}
    raw = chat([sys, usr])

    # extract first JSON array even if GPT surrounds it with text
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            snippet = re.search(r"\[[\s\S]*\]", raw).group()
            data = json.loads(snippet)
        except Exception:
            st.warning("Could not parse GPT grants JSON."); data = []
    return data[:NUM]

# ── 2 · Rank + feasibility ──────────────────────────────────────
def analyse(df_raw: pd.DataFrame):
    if df_raw.empty: return pd.DataFrame()

    mvec = embed(MISSION)
    df_raw["sim"] = [
        float(cosine_similarity([embed(s)], [mvec])[0][0]) for s in df_raw.summary
    ]
    df = df_raw.sort_values("sim", ascending=False).reset_index(drop=True)

    feas, why = [], []
    for _, row in df.iterrows():
        prompt = (
            f'Mission: "{MISSION}"\n'
            f'Grant: "{row.title}" – {row.summary}\n\n'
            'Return JSON {"feasibility":"High|Medium|Low","why":"<one sentence>"}'
        )
        try:
            j = json.loads(chat([{"role":"user","content":prompt}], maxtok=60))
            feas.append(j.get("feasibility","?")); why.append(j.get("why",""))
        except Exception:
            feas.append("?"); why.append("parse error")
    df["feasibility"], df["why_fit"] = feas, why
    return df

# ── UI ───────────────────────────────────────────────────────────
st.title("CT RISE Network — Smart Grant Finder (GPT v3)")
st.markdown(f"> **Mission:** {MISSION}")

if st.button("📚 Research & rank grants", type="primary"):
    with st.spinner("GPT is compiling real grants… please wait 60-90 s"):
        raw_list = fetch_grants()
        table    = analyse(pd.DataFrame(raw_list))
        st.session_state["tbl"] = table
        st.success("Done!")

if "tbl" in st.session_state and not st.session_state["tbl"].empty:
    st.subheader("Matched grants (ranked)")
    st.dataframe(
        st.session_state["tbl"][
            ["title","sponsor","amount","deadline","sim","feasibility","why_fit","url"]
        ],
        use_container_width=True,
    )
elif "tbl" in st.session_state:
    st.info("No usable grants returned — click again in a few minutes.")
else:
    st.caption("Build tag: v3-grants-amount  |  Ready to generate.")
