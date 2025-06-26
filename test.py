# Final Capstone Project (Grant Matcher for CT RISE using ChatGPT Search) – stable build

import os, json, re, time
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import openai

# ── CONFIG ─────────────────────────────────────────
MISSION = (
    "The Connecticut RISE Network empowers public high schools with data-driven strategies "
    "and personalized support to improve student outcomes and promote postsecondary success, "
    "especially for Black, Latinx, and low-income youth."
)
NUM_ROWS   = 10           # final table size
ASK_FOR    = 14           # request a few extras
CHAT_MODEL = "gpt-3.5-turbo-1106"       # supports JSON mode
EMB_MODEL  = "text-embedding-ada-002"
MAX_RETRY  = 4            # OpenAI retries
SLEEP      = 2            # seconds between retries
PROMPT_TRY = 4            # attempts to get valid JSON

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ── OPENAI HELPERS ─────────────────────────────────
def ask_chat(messages, max_tokens=1000):
    for attempt in range(MAX_RETRY):
        try:
            resp = openai.ChatCompletion.create(
                model=CHAT_MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return resp.choices[0].message.content
        except openai.error.RateLimitError:
            time.sleep(SLEEP * (attempt + 1))
    st.error("OpenAI rate-limited; try again later."); st.stop()

def get_embed(text: str):
    for attempt in range(MAX_RETRY):
        try:
            vec = openai.Embedding.create(input=text, model=EMB_MODEL)
            return vec["data"][0]["embedding"]
        except openai.error.RateLimitError:
            time.sleep(SLEEP * (attempt + 1))
    st.error("Embedding rate-limited."); st.stop()

# ── STEP 1: Fetch grants list ──────────────────────
def fetch_grants() -> list[dict]:
    sys_msg = {"role": "system", "content": "You are a meticulous grants researcher."}
    user_base = (
        f"Provide {ASK_FOR} CURRENT (2024-2025) grant opportunities for US nonprofits "
        "focused on high-school education, youth equity, or college readiness. "
        "Return ONLY a JSON array; each element must include keys: "
        "title, sponsor, amount, deadline, url, summary."
    )

    messages = [sys_msg, {"role": "user", "content": user_base}]
    for _ in range(PROMPT_TRY):
        raw = ask_chat(messages)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # salvage [ ... ]
            m = re.search(r"\[.*\]", raw, re.S)
            data = json.loads(m.group()) if m else []
        cleaned = []
        for d in data:
            cleaned.append({k: d.get(k, "N/A") for k in
                            ("title", "sponsor", "amount", "deadline", "url", "summary")})
        if len(cleaned) >= NUM_ROWS:
            return cleaned[:NUM_ROWS]
        # retry prompt
        messages.append({"role":"user",
                         "content":"Please try again. Remember: strict JSON array only."})
    st.error("Unable to obtain complete grant list from GPT."); st.stop()

# ── STEP 2: Rank & add “Why Fit” ──────────────────
def rank_table(df: pd.DataFrame) -> pd.DataFrame:
    mission_vec = get_embed(MISSION)
    df["%Match"] = (
        df.summary.apply(lambda s: cosine_similarity([get_embed(s)], [mission_vec])[0][0] * 100)
        .round(1)
    )
    df = df.sort_values("%Match", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.insert(0, "Rank", df.index)

    whys = []
    for _, row in df.iterrows():
        q = (f'Mission: "{MISSION}"\n'
             f'Grant: "{row.title}" – {row.summary}\n'
             "One sentence on why this aligns.")
        ans = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":q}],
            max_tokens=60,
            temperature=0.3,
        ).choices[0].message.content.strip()
        whys.append(ans)
    df["Why Fit"] = whys
    return df

# ── UI ─────────────────────────────────────────────
st.title("Final Capstone Project (Grant Matcher for CT RISE using ChatGPT Search)")
st.write("> **Mission:**", MISSION)

if st.button("🚀 Generate & Rank 10 Grants", type="primary"):
    with st.spinner("GPT is gathering grants and computing similarity…"):
        grants = fetch_grants()
        table  = rank_table(pd.DataFrame(grants))
        st.session_state["tbl"] = table
        st.success("Done!")

if "tbl" in st.session_state:
    st.dataframe(
        st.session_state["tbl"][
            ["Rank","title","sponsor","amount","deadline","%Match","Why Fit","url","summary"]
        ],
        use_container_width=True
    )
else:
    st.caption("Press the rocket to generate your ranked grant list.")
