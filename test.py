# Final Capstone Project (Grant Matcher for CT RISE using ChatGPT search)

import os, json, re, time
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import openai

# ── CONSTANTS ─────────────────────────────────────
MISSION = (
    "The Connecticut RISE Network empowers public high schools with data-driven strategies "
    "and personalized support to improve student outcomes and promote post-secondary success, "
    "especially for Black, Latinx, and low-income youth."
)
ROWS_NEEDED   = 10               # final table size
GPT_REQUEST   = 14               # ask for a few extras
CHAT_MODEL    = "gpt-3.5-turbo"  # widely available
EMB_MODEL     = "text-embedding-ada-002"
MAX_API_RETRY = 4
SLEEP         = 2                # sec between retries
# ──────────────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ── OPENAI WRAPPERS ───────────────────────────────
def call_chat(msgs, max_tokens=1000):
    for a in range(MAX_API_RETRY):
        try:
            resp = openai.ChatCompletion.create(
                model=CHAT_MODEL,
                messages=msgs,
                max_tokens=max_tokens,
                temperature=0.4,
            )
            return resp.choices[0].message.content
        except openai.error.RateLimitError:
            time.sleep(SLEEP * (a + 1))
    st.error("OpenAI rate-limited; try later."); st.stop()

def get_embed(text):
    for a in range(MAX_API_RETRY):
        try:
            return openai.Embedding.create(input=text, model=EMB_MODEL)["data"][0]["embedding"]
        except openai.error.RateLimitError:
            time.sleep(SLEEP * (a + 1))
    st.error("Embedding rate-limited."); st.stop()

# ── STEP 1 – fetch grants list (robust loop) ──────
def fetch_grants():
    base_prompt = (
        f"Return {GPT_REQUEST} CURRENT (2024-2025) grant opportunities relevant to "
        "US nonprofits focused on high-school education, youth equity, or college readiness. "
        "Output ONLY a JSON array; each object MUST have the keys "
        "title, sponsor, amount, deadline, url, summary."
    )
    msgs = [{"role":"user","content":base_prompt}]
    for _ in range(5):                              # up to 5 prompt attempts
        raw = call_chat(msgs)
        # try to parse JSON array, even if surrounded by text
        match = re.search(r"\[.*\]", raw, re.S)
        try:
            data = json.loads(match.group() if match else raw)
        except Exception:
            msgs.append({"role":"user","content":"Please try again. JSON array only."})
            continue
        cleaned=[]
        for d in data:
            cleaned.append({k: d.get(k,"N/A") for k in
                           ("title","sponsor","amount","deadline","url","summary")})
        if len(cleaned) >= ROWS_NEEDED:
            return cleaned[:ROWS_NEEDED]
        msgs.append({"role":"user","content":
                     "Need complete data for at least 10 grants. Try again."})
    st.error("Could not obtain 10 complete grants from GPT."); st.stop()

# ── STEP 2 – rank & add 'Why Fit' ─────────────────
def build_table(rows):
    df = pd.DataFrame(rows)
    mission_vec = get_embed(MISSION)
    df["%Match"] = (
        df.summary.apply(lambda s:
            cosine_similarity([get_embed(s)], [mission_vec])[0][0] * 100)
        .round(1)
    )
    df = df.sort_values("%Match", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.insert(0,"Rank", df.index)

    whys=[]
    for _, r in df.iterrows():
        q = (f'Mission: "{MISSION}"\nGrant: "{r.title}" – {r.summary}\n'
             "Explain briefly (1 sentence) why this aligns.")
        whys.append(call_chat([{"role":"user","content":q}], max_tokens=60).strip())
    df["Why Fit"] = whys
    return df

# ── STREAMLIT UI ──────────────────────────────────
st.title("Final Capstone Project (Grant Matcher for CT RISE using ChatGPT Search)")
st.write("> **Mission:**", MISSION)

if st.button("🚀 Generate & Rank 10 Grants LETS GO", type="primary"):
    with st.spinner("GPT gathering grant information…"):
        grant_rows = fetch_grants()
        st.session_state["tbl"] = build_table(grant_rows)
        st.success("Done!")

if "tbl" in st.session_state:
    st.dataframe(
        st.session_state["tbl"][
            ["Rank","title","sponsor","amount","deadline",
             "%Match","Why Fit","url","summary"]
        ],
        use_container_width=True
    )
else:
    st.caption("Press the rocket to generate your ranked grant list.")
