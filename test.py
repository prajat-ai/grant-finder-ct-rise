import os, json, re, time, pandas as pd, streamlit as st, openai
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# ── CONFIG ──────────────────────────────────────────
SEARCH_MODEL = "gpt-4o-mini-search-preview"
CHAT_MODEL   = "gpt-3.5-turbo"
EMB_MODEL    = "text-embedding-ada-002"
NEEDED       = 10
RETRY        = 4
PAUSE        = 2

MISSION = (
    "The Connecticut RISE Network empowers public high schools with data-driven strategies "
    "and personalized support to improve student outcomes and promote post-secondary success, "
    "especially for Black, Latinx, and low-income youth."
)

# ── KEYS ────────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ── HELPERS ─────────────────────────────────────────
def backoff(fn):
    def wrapper(*a, **k):
        for i in range(RETRY):
            try:
                return fn(*a, **k)
            except openai.RateLimitError:
                time.sleep(PAUSE * (i + 1))
        st.error("OpenAI rate-limit; try again later."); st.stop()
    return wrapper

@backoff
def embed(text: str):
    return openai.embeddings.create(model=EMB_MODEL, input=text).data[0].embedding

@backoff
def chat_base(model: str, messages: list[dict]):
    return openai.chat.completions.create(model=model, messages=messages)

# ── STEP 1 – single search call ─────────────────────
def fetch_grants():
    query = (
        f"search: Provide exactly {NEEDED} CURRENT grant opportunities for US non-profits "
        "focused on high-school education, youth equity, or college readiness. "
        "Respond ONLY with a JSON array. Each element must have keys "
        "title, sponsor, amount, deadline, url, summary."
    )
    resp = chat_base(SEARCH_MODEL, [{"role": "user", "content": query}])
    txt  = resp.choices[0].message.content

    # extract JSON array
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", txt, re.S)
        data = json.loads(m.group()) if m else []

    # sanitize
    result = [
        {k: d.get(k, "N/A") for k in
         ("title", "sponsor", "amount", "deadline", "url", "summary")}
        for d in data
    ]
    return result[:NEEDED]

# ── STEP 2 – rank & “Why Fit” sentence ─────────────
def build_table(rows):
    df = pd.DataFrame(rows)
    base_vec = embed(MISSION)
    df["%Match"] = (
        df.summary.apply(lambda s: cosine_similarity([embed(s)], [base_vec])[0][0] * 100)
        .round(1)
    )
    df = df.sort_values("%Match", ascending=False).reset_index(drop=True)
    df.index += 1
    df.insert(0, "Rank", df.index)

    whys = []
    for _, r in df.iterrows():
        q = (f'In one sentence, why does the grant "{r.title}" align with this mission: '
             f'"{MISSION}"?')
        ans = chat_base(CHAT_MODEL, [{"role": "user", "content": q}]).choices[0].message.content.strip()
        whys.append(ans)
    df["Why Fit"] = whys
    return df

# ── STREAMLIT UI ───────────────────────────────────
st.title("Final Capstone Project (Grant Matcher for CT RISE – GPT-4o Search)")
st.write("> **Mission:**", MISSION)

if st.button("🚀 Generate & Rank 10 Grants", type="primary"):
    with st.spinner("GPT-4o searching web and compiling grants …"):
        grants = fetch_grants()
        if len(grants) < NEEDED:
            st.error("Model returned fewer than 10 grants – click again.")
        else:
            st.session_state["tbl"] = build_table(grants)
            st.success("Done!")

if "tbl" in st.session_state:
    st.dataframe(
        st.session_state["tbl"][
            ["Rank","title","sponsor","amount","deadline","%Match","Why Fit","url","summary"]
        ],
        use_container_width=True
    )
else:
    st.caption("Press the rocket to generate a ranked grant list.")
